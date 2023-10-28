import rospy
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from scipy.optimize import minimize

state_leader = None
state_follower = None

def callback_leader(data):
    global state_leader
    state_leader = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.twist.twist.linear.x, data.twist.twist.angular.z])

def callback_follower(data):
    global state_follower
    state_follower = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.twist.twist.linear.x, data.twist.twist.angular.z])
def get_desired_position(self, state_leader, distance_behind):
        desired_x = state_leader[0] - distance_behind * np.cos(state_leader[2]) 
        desired_y = state_leader[1] - distance_behind * np.sin(state_leader[2])
        return np.array([desired_x, desired_y])


def cost_function(self, u, *args):
        state_follower, state_leader, T, distance_behind = args
        N = len(u) // 2
        x = state_follower.copy()
        cost = 0

        position_weight = 7.0
        orientation_weight = 1.0
        control_weight = 0.4

        for k in range(N):
            control = u[2 * k:2 * k + 2]
            theta = x[2]
            x[0] += control[0] * np.cos(theta) * T
            x[1] += control[0] * np.sin(theta) * T
            x[2] += control[1] * T
            x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2]))

            desired_position = self.get_desired_position(state_leader, distance_behind)
            position_diff = np.linalg.norm(x[:2] - desired_position)
            orientation_diff = x[2] - state_leader[2]

            orientation_error = np.arctan2(np.sin(orientation_diff), np.cos(orientation_diff))

            control_effort = np.sum(np.square(control))

            cost += position_weight * np.square(position_diff)
            cost += orientation_weight * np.square(orientation_error)
            cost += control_weight * control_effort

        return cost

    

def calculate_mpc_control_input(self):
        T = 0.2
        N = 20
        u_dim = 2
        v_max = 0.4
        omega_max = np.pi
        distance_behind = 1

        u0 = np.zeros(N * u_dim)

        bounds = [(0, v_max) if i % u_dim == 0 else (-omega_max, omega_max) for i in range(N * u_dim)]

        args = (self.state_follower, self.state_leader, T, distance_behind)

        result = minimize(self.cost_function, u0, args=args, method='SLSQP', bounds=bounds)

        if result.success:
            return result.x[:u_dim]
        else:
            rospy.logwarn("MPC optimization failed.")
            return None





def mpc_controller():
    global state_leader, state_follower

    # Initialize the node
    rospy.init_node('mpc_controller', anonymous=True)

    # Create publishers for '/robot1/cmd_vel' and '/robot2/cmd_vel'
    pub_follower = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=10)
    pub_leader = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=10)
    
    # Subscribe to the robots' state (e.g., odometry or other relevant topics)
    rospy.Subscriber('/robot1/odom', Odometry, callback_leader)
    rospy.Subscriber('/robot2/odom', Odometry, callback_follower)

    # Set the rate
    rate = rospy.Rate(10)  # 10 Hz, adjust as necessary

    while not rospy.is_shutdown():
        if state_leader is not None and state_follower is not None:
            # Use the leader's state as a reference trajectory for the follower.
            u_optimal = calculate_mpc_control_input(state_leader, state_follower)

            vel_cmd_leader = Twist()
            vel_cmd_leader.linear.x = 0.2
            vel_cmd_leader.angular.z = 0.2
            pub_leader.publish(vel_cmd_leader)

            if u_optimal is not None:
                # Create the Twist message for the follower
                vel_cmd_follower = Twist()
                vel_cmd_follower.linear.x = u_optimal[0]
                vel_cmd_follower.angular.z = u_optimal[1]
                pub_follower.publish(vel_cmd_follower)
                print(u_optimal[0])
                print(u_optimal[1])
            else:
                rospy.logwarn("No optimal control input calculated.")
        else:
            rospy.logwarn("Waiting for robot states.")

        # Sleep to maintain the loop rate
        rate.sleep()

if __name__ == '__main__':
    try:
        mpc_controller()
    except rospy.ROSInterruptException:
        pass
