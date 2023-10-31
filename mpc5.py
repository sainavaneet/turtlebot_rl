import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from scipy.optimize import minimize
from sensor_msgs.msg import LaserScan

class TurtlebotLMPC:

    def __init__(self):
        self.robot1_x = 0.0
        self.robot1_y = 0.0
        self.robot1_theta = 0.0
        self.state_leader = None
        self.state_follower = None
        self.distance = np.inf
        rospy.init_node('lmpc_follower', anonymous=True)
        self.pub_follower = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=10)
        self.pub_leader = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/robot2/odom', Odometry, self.callback_follower)
        rospy.Subscriber('/robot1/odom', Odometry, self.callback_leader)
        rospy.Subscriber('/robot2/scan', LaserScan, self.scan_callback)
        self.rate = rospy.Rate(10)  
        self.obstacle_close = False
        self.previous_trajectories = []
        

    def store_trajectory(self, state, control):
        self.previous_trajectories.append((state, control))

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

            self.obstacle_x = distance_to_obstacle * np.cos(angle_to_obstacle) - 1
            self.obstacle_y = distance_to_obstacle * np.sin(angle_to_obstacle)
            self.obstacle_theta = np.arctan2(self.obstacle_y, self.obstacle_x)

        distance_to_obstacle = np.linalg.norm(np.array([self.obstacle_x, self.obstacle_y]) - self.state_follower[:2])
        self.obstacle_close = distance_to_obstacle < 0.4

    def learned_cost(self, x):
            if not self.previous_trajectories:
                return 0
            cost = min(np.linalg.norm(np.array(x[:2]) - np.array(state[:2])) for state, _ in self.previous_trajectories)
            return cost * 10.0
            
        
    def callback_leader(self, data):
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot1_x = data.pose.pose.position.x
        self.robot1_y = data.pose.pose.position.y
        self.robot1_theta = yaw
        self.state_leader = np.array([self.robot1_x, self.robot1_y, self.robot1_theta])

    def callback_follower(self, data):
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        self.robot2_x = data.pose.pose.position.x
        self.robot2_y = data.pose.pose.position.y
        self.robot2_theta = yaw
        self.state_follower = np.array([self.robot2_x, self.robot2_y, self.robot2_theta])

    def get_desired_position(self, state_leader, distance_behind):
        desired_x = state_leader[0] - distance_behind * np.cos(state_leader[2]) 
        desired_y = state_leader[1] - distance_behind * np.sin(state_leader[2])
        return np.array([desired_x, desired_y])

    def obstacle_cost(self, x):
        if not hasattr(self, 'obstacle_x') or not hasattr(self, 'obstacle_y'):
            return 0

        # Calculate distance to the obstacle
        distance_to_obstacle = np.linalg.norm(np.array([self.obstacle_x, self.obstacle_y]) - x[:2])

        if distance_to_obstacle > 0.7:  # If distance is more than a threshold, return 0 cost
            return 0

        # Calculate the angle between robot's current direction and direction towards the obstacle
        robot_dir = np.array([np.cos(x[2]), np.sin(x[2])])
        obstacle_dir = np.array([self.obstacle_x, self.obstacle_y]) - x[:2]
        obstacle_dir /= np.linalg.norm(obstacle_dir)  # normalize

        dot_product = np.dot(robot_dir, obstacle_dir)
        relative_angle = np.arccos(dot_product)

        # Penalize configurations where the robot is facing the obstacle and it's close to the obstacle
        orientation_penalty = 1e5 * (1 + np.cos(relative_angle))
        proximity_penalty = 1e5 * (0.7 - distance_to_obstacle)

        return orientation_penalty + proximity_penalty




    def cost_function(self, u, *args):
        state_follower, state_leader, T, distance_behind = args
        N = len(u) // 2
        x = state_follower.copy()
        total_cost = 0
        
        for k in range(N):
            control = u[2*k : 2*k+2]
            theta = x[2]
            
            x[0] += control[0] * np.cos(theta) * T
            x[1] += control[0] * np.sin(theta) * T
            x[2] += control[1] * T

            tracking_cost = np.square(np.linalg.norm(x[:2] - self.get_desired_position(state_leader, distance_behind)))
            orientation_cost = np.square(x[2] - state_leader[2])
            control_cost = np.sum(np.square(control))
            
            total_cost += 10.0 * tracking_cost + 4 * orientation_cost + 0.4 * control_cost
            total_cost += self.obstacle_cost(x)
            total_cost += self.learned_cost(x)
        return total_cost

    def calculate_mpc_control_input(self):
        T = 0.1
        N = 10
        u_dim = 2
        u0 = np.zeros(N * u_dim)
        distance_behind = 1
        bounds = [(0, 0.4) if i % u_dim == 0 else (-np.pi, np.pi) for i in range(N * u_dim)]

        if self.obstacle_close:
            robot_dir = np.array([np.cos(self.state_follower[2]), np.sin(self.state_follower[2])])
            obstacle_dir = np.array([self.obstacle_x, self.obstacle_y]) - self.state_follower[:2]
            obstacle_dir /= np.linalg.norm(obstacle_dir)
            
            if np.dot(robot_dir, obstacle_dir) > 0.5:
                bounds[0] = (0, 0.1)

        args = (self.state_follower, self.state_leader, T, distance_behind)
        result = minimize(self.cost_function, u0, args=args, method='SLSQP', bounds=bounds)
        
        if result.success:
            self.store_trajectory(self.state_follower, result.x[:u_dim])
            return result.x[:u_dim]
        else:
            return None


    def lmpc_follower(self):
        while not rospy.is_shutdown():
            if self.state_leader is not None and self.state_follower is not None:
                u_optimal = self.calculate_mpc_control_input()
                if u_optimal is not None:
                    vel_cmd = Twist()
                    vel_cmd.linear.x = u_optimal[0]
                    vel_cmd.angular.z = u_optimal[1]
                    self.pub_follower.publish(vel_cmd)
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = TurtlebotLMPC()
        controller.lmpc_follower()
    except rospy.ROSInterruptException:
        pass
