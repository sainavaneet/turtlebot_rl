import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from scipy.optimize import minimize
from sensor_msgs.msg import LaserScan

class TurtlebotController:

    def __init__(self):
        self.robot1_x = 0.0
        self.robot1_y = 0.0
        self.robot1_theta = 0.0
        self.state_leader = None
        self.state_follower = None
        self.distance = None
        self.laser_scan_data = None 

        rospy.init_node('mpc_follower', anonymous=True)

        self.pub_follower = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=10)
        self.pub_leader = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/robot2/odom', Odometry, self.callback_follower)
        rospy.Subscriber('/robot1/odom', Odometry, self.callback_leader)
        rospy.Subscriber('/robot2/scan', LaserScan, self.scan_callback)
        self.rate = rospy.Rate(10)  # 10Hz

    def scan_callback(self, msg):
        threshold = 0.5
        clusters = []
        cluster = []
        angles = []

        for i in range(len(msg.ranges)):
            if msg.ranges[i] == float('inf'):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if cluster:
                last_angle = angles[-1]
                angle_diff = angle - last_angle
                # Grouping continuous points with distance changes less than the threshold
                if angle_diff < msg.angle_increment + 1e-5 and abs(msg.ranges[i] - cluster[-1]) < threshold:
                    cluster.append(msg.ranges[i])
                    angles.append(angle)
                else:
                    clusters.append((cluster, angles))
                    cluster = [msg.ranges[i]]
                    angles = [angle]
            else:
                cluster = [msg.ranges[i]]
                angles = [angle]

        if cluster:  # Don't forget the last cluster
            clusters.append((cluster, angles))

        clusters.sort(key=lambda x: len(x[0]), reverse=True)

        if clusters:
            distance_to_obstacle = np.mean(clusters[0][0])  # mean distance
            angle_to_obstacle = np.mean(clusters[0][1])  # mean angle
            
            self.distance = distance_to_obstacle
            self.laser_scan_data = msg

            
        else:
            rospy.loginfo("no obsticle in range")
        


    def callback_leader(self, data):
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.state_leader = np.array([data.pose.pose.position.x, data.pose.pose.position.y, yaw])

    def callback_follower(self, data):
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.state_follower = np.array([data.pose.pose.position.x, data.pose.pose.position.y, yaw])

    def cost_function(self, u, *args):
        state_follower, state_leader, T, desired_distance = args
        N = len(u) // 2  # Number of control inputs
        x = state_follower.copy()
        cost = 0

        position_weight = 7.0
        orientation_weight = 1.0
        control_weight = 0.1

        for k in range(N):
            control = u[2 * k:2 * k + 2]
            theta = x[2]
            x[0] += control[0] * np.cos(theta) * T  # Update x-position
            x[1] += control[0] * np.sin(theta) * T  # Update y-position
            x[2] += control[1] * T  # Update orientation
            x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2]))

            position_diff = np.linalg.norm(x[:2] - state_leader[:2]) - desired_distance
            orientation_diff = x[2] - state_leader[2]

            orientation_error = np.arctan2(np.sin(orientation_diff), np.cos(orientation_diff))
            orientation_error_weight = 1.0  # Adjust the weight as needed

            control_effort = np.sum(np.square(control))

            cost += position_weight * np.square(position_diff)
            cost += orientation_weight * np.square(orientation_error) * orientation_error_weight
            cost += control_weight * control_effort

        return cost

    def calculate_mpc_control_input(self):
        T = 0.2  
        N = 20  
        u_dim = 2  
        v_max = 0.5  
        omega_max = np.pi  
        desired_distance = 0.5  

        u0 = np.zeros(N * u_dim)

        bounds = [(0, v_max) if i % u_dim == 0 else (-omega_max, omega_max) for i in range(N * u_dim)]

        args = (self.state_follower, self.state_leader, T, desired_distance)
        result = minimize(self.cost_function, u0, args=args, method='SLSQP', bounds=bounds)

        if result.success:
            optimal_control = result.x[:u_dim]  # We are interested only in the immediate next control input
            return optimal_control
        else:
            rospy.logwarn("MPC optimization failed.")
            return None
    def mpc_follower(self):
        while not rospy.is_shutdown():
            # self.scan_callback()

            
            rospy.loginfo(f"Distance to the nearest obstacle: {self.distance}")
            vel_cmd_leader = Twist()
            vel_cmd_leader.linear.x = 0.3
            vel_cmd_leader.angular.z = 0.3
            # self.pub_leader.publish(vel_cmd_leader)
            if self.state_leader is not None and self.state_follower is not None:
                u_optimal = self.calculate_mpc_control_input()
                if u_optimal is not None:
                    vel_cmd = Twist()
                    vel_cmd.linear.x = u_optimal[0]
                    vel_cmd.angular.z = u_optimal[1]
                    self.pub_follower.publish(vel_cmd)
                    
                else:
                    rospy.logwarn("No control command calculated. MPC optimization might have failed.")
            else:
                rospy.logwarn("No control command calculated. MPC optimization might have failed.")
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = TurtlebotController()
        controller.mpc_follower()
    except rospy.ROSInterruptException:
        pass
