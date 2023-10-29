import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf.transformations import euler_from_quaternion
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize
from std_srvs.srv import Empty
import os
import atexit

class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AdvancedNN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.main(x)

class TurtlebotController:

    def __init__(self):
        self.obstacle_distance = np.inf
        self.laser_scan_data = None 
        self.state_leader = None
        self.state_follower = None
        rospy.init_node('mpc_follower', anonymous=True)
        self.pub_follower = rospy.Publisher('/robot2/cmd_vel', Twist, queue_size=10)
        self.pub_leader = rospy.Publisher('/robot1/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/robot2/odom', Odometry, self.callback_follower)
        rospy.Subscriber('/robot1/odom', Odometry, self.callback_leader)
        rospy.Subscriber('/robot2/scan', LaserScan, self.scan_callback)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.n_actions = 2  
        self.n_states = 1  
        self.learning_rate = 0.00025
        hidden_size = 128  
        self.policy_net = AdvancedNN(self.n_states, hidden_size, self.n_actions)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)  
        self.criterion = nn.MSELoss()
        self.episode_count = 0  # Counts completed episodes
        self.save_interval = 10  # Save every 10 episodes
        self.rate = rospy.Rate(10)  
        self.model_save_path = "/home/navaneet/rl_ws/src/turtlebot_controller/rl_mpc/saved/model.pt" 

    def scan_callback(self, msg):
        self.obstacle_distance = min(msg.ranges) 

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor([state])
            action_values = self.policy_net(state_tensor)
        return torch.argmax(action_values).item()

    def optimize_policy(self, state, action, reward, next_state):
        state = torch.FloatTensor([state])
        next_state = torch.FloatTensor([next_state])
        reward = torch.FloatTensor([reward])
        action = torch.tensor([action])

        predicted_targets = self.policy_net(state)
        predicted_target = predicted_targets.gather(0, action)

        with torch.no_grad():
            labels_next = self.policy_net(next_state)
            label = reward + 0.99 * torch.max(labels_next)

        loss = self.criterion(predicted_target, label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
        v_max = 0.5
        omega_max = np.pi
        distance_behind = 1.5

        u0 = np.zeros(N * u_dim)

        bounds = [(0, v_max) if i % u_dim == 0 else (-omega_max, omega_max) for i in range(N * u_dim)]

        args = (self.state_follower, self.state_leader, T, distance_behind)

        result = minimize(self.cost_function, u0, args=args, method='SLSQP', bounds=bounds)

        if result.success:
            return result.x[:u_dim]
        else:
            rospy.logwarn("MPC optimization failed.")
            return None

    def convert_rl_action_to_velocity(self, action):
        linear_velocity = 0.0
        angular_velocity = 0.0
        if action == 0:
            linear_velocity = 0.2    
        elif action == 1:
            linear_velocity = 0.1
            angular_velocity = 0.8
        elif action == 2:
            linear_velocity = 0.2
            angular_velocity = 0.8
        elif action == 3:
            linear_velocity = 0.1
            angular_velocity = 0.9
        elif action == 4:
            linear_velocity = 0.1
            angular_velocity = -0.9
        elif action == 5:
            linear_velocity = -0.2
        elif action == 6:
            pass 

        return linear_velocity, angular_velocity
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved at {path} after episode {self.episode_count}")
            
    def load_model(self, path):
    
        if os.path.exists(path):
            self.policy_net.load_state_dict(torch.load(path))
            self.policy_net.eval()  
            print(f"Model loaded from {path}")
        else:
            print("No model found at specified path.")

    
    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
            self.unpause_proxy()
            rospy.sleep(2)  

        except rospy.ServiceException as e:
            rospy.logerr("An error occurred while trying to reset the simulation: %s", e)

            

    def run(self):
        while not rospy.is_shutdown():
            log_info_details = {}

            log_info_details['ObstacleDist'] = f"{self.obstacle_distance:.2f}"

            if self.obstacle_distance <= 0.13:
                log_info_details['Status'] = "RESET"
                self.episode_count += 1  

                if self.episode_count % self.save_interval == 10:
                    self.save_model(self.model_save_path)

                self.reset()
                continue

            elif self.obstacle_distance < 0.3:
                current_state = self.obstacle_distance
                action = self.select_action(current_state)

                linear_velocity, angular_velocity = self.convert_rl_action_to_velocity(action)

                twist = Twist()
                twist.linear.x = linear_velocity
                twist.angular.z = angular_velocity
                self.pub_follower.publish(twist)

                log_info_details['Mode'] = "RL"
                log_info_details['Action'] = f"{action}"
                log_info_details['LinearVel'] = f"{linear_velocity:.2f}"
                log_info_details['AngularVel'] = f"{angular_velocity:.2f}"

            else:
                if self.state_follower is None or self.state_leader is None:
                    rospy.logwarn("State not initialized. Skipping iteration.")
                    continue

                optimal = self.calculate_mpc_control_input()
                if optimal is not None:
                    twist = Twist()
                    twist.linear.x = optimal[0]
                    twist.angular.z = optimal[1]
                    self.pub_follower.publish(twist)

                    # Add MPC details to the log info.
                    log_info_details['Mode'] = "MPC"
                    log_info_details['LinearVel'] = f"{twist.linear.x:.2f}"
                    log_info_details['AngularVel'] = f"{twist.angular.z:.2f}"
                else:
                    rospy.logwarn("MPC calculation failed.")

            rospy.loginfo('; '.join([f"{k}: {v}" for k, v in log_info_details.items()]))

            self.rate.sleep()  # This line should be within the loop.

if __name__ == '__main__':
    try:
        controller = TurtlebotController()
        atexit.register(controller.save_model, controller.model_save_path)
        controller.run()
    except rospy.ROSInterruptException:
        pass
