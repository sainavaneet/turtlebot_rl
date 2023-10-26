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

def cost_function(u, *args):
    state_follower, state_leader, T = args
    N = len(u) // 2
    x = state_follower.copy()
    cost = 0
    
    for k in range(N):
        state_diff = x - state_leader
        cost += np.sum(state_diff**2) + np.sum(u[k*2:k*2+2]**2)

        cos_theta = np.cos(x[2])
        sin_theta = np.sin(x[2])
        u_k = u[k*2:k*2+2]
        x_next = x + np.array([T * cos_theta * u_k[0], T * sin_theta * u_k[0], 0, T * u_k[1]])
        x = x_next.copy()
    
    return cost


def calculate_mpc_control_input(state_leader, state_follower):
    T = 0.1  # Sample time
    N = 10  # Prediction horizon
    v_max = 0.5  # Maximum linear velocity
    omega_max = 0.5  # Maximum angular velocity

    u0 = np.zeros(2 * N)

    # Define bounds for linear velocity allowing both positive and negative values
    bounds_v = [(0, v_max)] * N

    # Define bounds for angular velocity within the range of -π to π
    bounds_omega = [(-np.pi, np.pi)] * N

    # Combine bounds for both linear and angular velocities
    bounds = bounds_v + bounds_omega

    args = (state_follower, state_leader, T)
    result = minimize(cost_function, u0, args=args, bounds=bounds)

    if result.success:
        u_optimal = result.x  # Extract the optimal control inputs
        return u_optimal
    else:
        rospy.logwarn("MPC problem is infeasible. No solution found.")
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
