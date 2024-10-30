# Version: 1.1
# Description: This script performs Bayesian Optimization for three agents controlling a servo motor system. 
# The goal is to find the optimal control gains (Kp and Kd) that maximize a reward function based on tracking performance and minimizing overshoot errors.
# The script uses a combination of remote commands, data retrieval, reward computation, and optimization over multiple iterations to achieve the best parameter settings for the system.

import subprocess
import time
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import os  # For directory operations
import csv  # For writing and reading CSV files
import safeopt
import GPy
from scipy.optimize import minimize
from plot_iteration_3A import plot_iteration

################ PHASE 1 ################

def sent_command(target_uri, modelName, gain_arg, std_args):
    """
    Send command to the target.
    """
    sys_run = f'quarc_run -l -t {target_uri} {modelName}.rt-linux_rt_armv7{gain_arg} {std_args}'
    subprocess.call(sys_run, shell=True)
    

def retrieve_data(target_uri, modelName, gain_arg, std_args, agent, iteration):
    """
    Retrieve data from the target.
    """
    sys_get = f'quarc_run -u -t {target_uri} {modelName}.rt-linux_rt_armv7{gain_arg}{std_args}'
    print(sys_get)
    subprocess.call(sys_get, shell=True)
    # Create 'data_3A' directory if it doesn't exist
    data_dir = 'data_3A'  # Changed to 'data_3A'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Save the .mat files in the 'data_3A' directory
    shutil.copyfile('servoPDF.mat', f'{data_dir}/servoPDF-{agent}_{iteration}.mat')
    
def load_agent_data(filename):
    """
    Load data from the agent and extract relevant data
        
    Returns:
    rt_t: Time
    rt_theta: Theta
    theta_d
    """
    data = loadmat(filename)
    rt_t = data['rt_t'].flatten()
    rt_theta = data['rt_theta'].flatten()
    theta_d = data['rt_theta_d'].flatten()
    return rt_t, rt_theta, theta_d

def compute_reward(theta_d, rt_theta1, rt_theta2, rt_theta3, rt_t1, rt_t2, rt_t3):
    """
    Computes the total reward for the agents
        
    Reward = integral{abs(theta_d - rt_theta)}dt
        
    Returns:
    total_error: Total reward
    os1: Overshoot error for agent 1
    os2: Overshoot error for agent 2
    os3: Overshoot error for agent 3
    """
    # Overshoot error
    os1 = np.abs(theta_d - rt_theta1)
    os2 = np.abs(theta_d - rt_theta2)
    os3 = np.abs(theta_d - rt_theta3)
    
    # Compute error between the agents
    error12 = np.abs(rt_theta1 - rt_theta2)
    error13 = np.abs(rt_theta1 - rt_theta3)
    error23 = np.abs(rt_theta2 - rt_theta3)

    # Compute integral of errors
    integral_os1 = np.trapz(os1, rt_t1)
    integral_os2 = np.trapz(os2, rt_t2)
    integral_os3 = np.trapz(os3, rt_t3)
    total_os = (integral_os1 + integral_os2 + integral_os3) / 3  # Average overshoot error

    integral_error12 = np.trapz(error12, rt_t1)
    integral_error13 = np.trapz(error13, rt_t1)
    integral_error23 = np.trapz(error23, rt_t1)


    total_inter_agent_error = integral_error12 + integral_error13 + integral_error23

    total_error = total_os + 3  * total_inter_agent_error
    total_error = 1 / total_error 
    
    os1 = 1 / integral_os1
    os2 = 1 / integral_os2
    os3 = 1 / integral_os3
    
    return total_error, os1, os2, os3
       
def plot_data(rt_t1, rt_theta1, os1, rt_t2, rt_theta2, os2, rt_t3, rt_theta3, os3):
    """
    Plot the data from the agents
    """
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(rt_t1, rt_theta1, label='Agent-1')
    plt.plot(rt_t2, rt_theta2, label='Agent-2')
    plt.plot(rt_t3, rt_theta3, label='Agent-3')
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.ylabel('theta')
    plt.title("Theta over time")
    
    plt.subplot(1,2,2)
    plt.plot(rt_t1[:4000], os1[:4000], label='Agent-1 OS')
    plt.plot(rt_t2[:4000], os2[:4000], label='Agent-2 OS')
    plt.plot(rt_t3[:4000], os3[:4000], label='Agent-3 OS')
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.ylabel('theta')
    plt.title("Error over time")
    plt.legend()
    plt.show()

# Create directories to save data and plots
if not os.path.exists('plots_3A'):
    os.makedirs('plots_3A')
if not os.path.exists('data_3A'): 
    os.makedirs('data_3A')
if not os.path.exists('agent_data_3A'): 
    os.makedirs('agent_data_3A')

modelName = 'servoPDF'

# Target URIs for the agents
target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'
target_uri_2 = 'tcpip://172.22.11.10:17000?keep_alive=1'
target_uri_3 = 'tcpip://172.22.11.18:17000?keep_alive=1'  

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'

# Download model to target
sys1dl = f'quarc_run -D -t {target_uri_1} {modelName}.rt-linux_rt_armv7{std_args}'
sys2dl = f'quarc_run -D -t {target_uri_2} {modelName}.rt-linux_rt_armv7{std_args}'
sys3dl = f'quarc_run -D -t {target_uri_3} {modelName}.rt-linux_rt_armv7{std_args}' 

# Run the system commands
subprocess.call(sys1dl, shell=True)
subprocess.call(sys2dl, shell=True)
subprocess.call(sys3dl, shell=True)  

# Initial safepoint values.
kp1_0 = 4
kd1_0 = 0.3

kp2_0 = 8
kd2_0 = 0.5

kp3_0 = 6  
kd3_0 = 0.8

x0 = [(kp1_0, kd1_0), (kp2_0, kd2_0), (kp3_0, kd3_0)]


# Delay difference between the agents
td1 = 0.09
td2 = 0.045  
td3 = 0.001 

# Create gain arguments
gain_arg1 = f' -Kp {kp1_0} -Kd {kd1_0}'
gain_arg2 = f' -Kp {kp2_0} -Kd {kd2_0}'
gain_arg3 = f' -Kp {kp3_0} -Kd {kd3_0}'  

print(f'Initial gain arguments for Agent 1: {gain_arg1}')
print(f'Initial gain arguments for Agent 2: {gain_arg2}')
print(f'Initial gain arguments for Agent 3: {gain_arg3}')  

# Create system command for gain arguments
sys1run = f'quarc_run -l -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1} -td {td1:.5f} {std_args}'
sys2run = f'quarc_run -l -t {target_uri_2} {modelName}.rt-linux_rt_armv7{gain_arg2} -td {td2:.5f} {std_args}' 
sys3run = f'quarc_run -l -t {target_uri_3} {modelName}.rt-linux_rt_armv7{gain_arg3} -td {td3:.5f} {std_args}'  

# Run the system commands
subprocess.call(sys1run, shell=True)
subprocess.call(sys2run, shell=True)
subprocess.call(sys3run, shell=True)  

sent_command(target_uri_1, modelName, gain_arg1, std_args)
sent_command(target_uri_2, modelName, gain_arg2, std_args)
sent_command(target_uri_3, modelName, gain_arg3, std_args) 

# Wait for the experiment to finish
time.sleep(7)

# Retrieve data from Agents
retrieve_data(target_uri_1, modelName, gain_arg1, std_args, 1, 0)
retrieve_data(target_uri_2, modelName, gain_arg2, std_args, 2, 0)
retrieve_data(target_uri_3, modelName, gain_arg3, std_args, 3, 0)  

# Load data from Agents
rt_t1, rt_theta1, theta_d = load_agent_data('data_3A/servoPDF-1_0.mat') 
rt_t2, rt_theta2, _ = load_agent_data('data_3A/servoPDF-2_0.mat')       
rt_t3, rt_theta3, _ = load_agent_data('data_3A/servoPDF-3_0.mat')       

# Compute initial safe reward
reward_0, os1_0, os2_0, os3_0 = compute_reward(theta_d, rt_theta1, rt_theta2, rt_theta3, rt_t1, rt_t2, rt_t3)

print(f'Initial reward: {reward_0}')
print(f"Initial error1: {os1_0}")
print(f"Initial error2: {os2_0}")
print(f"Initial error3: {os3_0}") 
wait = input("Press Enter to start Bayesian Optimization...")

# =================== Bayesian Optimization ===================

# Define the Agent class for Bayesian Optimization
class Agent:
    def __init__(self, id, bounds, safe_point, initial_reward):
        self.id = id
        self.bounds = bounds
        self.safe_point = safe_point

        self.x0 = np.asarray([safe_point])
        self.y0 = np.asarray([[initial_reward]]) 

        self.kernel = GPy.kern.RBF(input_dim=len(bounds), ARD=True)
        self.gp = GPy.models.GPRegression(self.x0, self.y0, self.kernel, noise_var=0.05**2)

        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, 0.03, beta=1.0, threshold=0.05)

        self.kp_values = [safe_point]
        self.rewards = [initial_reward]

    def optimize(self):
        x_next = self.opt.optimize()
        return x_next

    def update(self, x_next, y_meas):
        self.opt.add_new_data_point(x_next, y_meas)
        
        self.kp_values.append(x_next)
        self.rewards.append(y_meas)

# Kp bounds
K_bounds = [(0.01, 10), (0.01, 1), (0.01, 10), (0.01, 1), (0.01, 10), (0.01, 1)]

agent1 = Agent(1, K_bounds, x0, reward_0)

wait = input("Press Enter to start Bayesian Optimization for Agent 1...")

# Quarc Experiment
def run_experiment(kp1, kd1, kp2, kd2, kp3, kd3, iteration):
    
    # Set gain arguments
    gain_arg1 = f' -Kp {kp1} -Kd {kd1}'
    gain_arg2 = f' -Kp {kp2} -Kd {kd2}'
    gain_arg3 = f' -Kp {kp3} -Kd {kd3}'

    sent_command(target_uri_1, modelName, gain_arg1, std_args)
    sent_command(target_uri_2, modelName, gain_arg2, std_args)
    sent_command(target_uri_3, modelName, gain_arg3, std_args) 

    # Await experiment completion
    time.sleep(7)

    retrieve_data(target_uri_1, modelName, gain_arg1, std_args, 1, iteration)
    retrieve_data(target_uri_2, modelName, gain_arg2, std_args, 2, iteration)
    retrieve_data(target_uri_3, modelName, gain_arg3, std_args, 3, iteration) 

    rt_t1, rt_theta1, theta_d = load_agent_data(f'data_3A/servoPDF-1_{iteration}.mat') 
    rt_t2, rt_theta2, _ = load_agent_data(f'data_3A/servoPDF-2_{iteration}.mat')       
    rt_t3, rt_theta3, _ = load_agent_data(f'data_3A/servoPDF-3_{iteration}.mat')     
    
    reward, os1, os2, os3 = compute_reward(theta_d, rt_theta1, rt_theta2, rt_theta3, rt_t1, rt_t2, rt_t3)

    return reward, os1, os2, os3

N = 5  # Number of iterations

# Initialize data files
agent_data_dir = 'agent_data_3A'  
if not os.path.exists(agent_data_dir):
    os.makedirs(agent_data_dir)

with open(f'{agent_data_dir}/agent1_data.txt', 'w', newline='') as f1, \
     open(f'{agent_data_dir}/agent2_data.txt', 'w', newline='') as f2, \
     open(f'{agent_data_dir}/agent3_data.txt', 'w', newline='') as f3: 
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)
    writer3 = csv.writer(f3)  
    writer1.writerow(['Iteration', 'Kp', 'Kd', 'Reward'])
    writer2.writerow(['Iteration', 'Kp', 'Kd', 'Reward'])
    writer3.writerow(['Iteration', 'Kp', 'Kd', 'Reward'])  
    writer1.writerow([0, x0[0], x0[1], reward_0])
    writer2.writerow([0, x0[2], x0[3], reward_0])
    writer3.writerow([0, x0[4], x0[5], reward_0]) 

with open(f'{agent_data_dir}/rewards.txt', 'w') as f:
    f.write('Iteration,Reward\n')
    f.write(f"0,{reward_0}\n")

# Bayesian Optimization
for iteration in range(1, N+1):
    
    # Get next Kp values from agents
    K1_next = agent1.optimize()
    
    print(K1_next)
    
    wait = input("Press Enter to run the experiment...")
    

    print(f"Iteration {iteration}, Agent 1:  -Kp {K1_next[0]} -Kd {K1_next[1]}, Agent 2: -Kp {K1_next[2]} -Kd {K1_next[3]}, Agent 3: -Kp {K1_next[4]} -Kd {K1_next[5]}")

    # Run the experiment with kp1_next, kp2_next, kp3_next
    y, os1, os2, os3 = run_experiment(K1_next[0], K1_next[1], K1_next[2], K1_next[3], K1_next[4], K1_next[5], iteration)

    print(f"Reward: {y}")
    
    # Update agents with observations
    agent1.update(K1_next, y)
    
    # Save agent's data to text files
    with open(f'{agent_data_dir}/agent1_data.txt', 'a', newline='') as f1, \
         open(f'{agent_data_dir}/agent2_data.txt', 'a', newline='') as f2, \
         open(f'{agent_data_dir}/agent3_data.txt', 'a', newline='') as f3: 
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer3 = csv.writer(f3) 
        writer1.writerow([iteration, K1_next[0], K1_next[1], y])
        writer2.writerow([iteration, K1_next[2], K1_next[3], y])
        writer3.writerow([iteration, K1_next[4], K1_next[5], y])  

    # Save rewards to a text file
    with open(f'{agent_data_dir}/rewards.txt', 'a') as f:
        f.write(f"{iteration},{y}\n")

    # Plot and save agents' opt plots in one figure with three subplots
    x_max_1, y_max_1 = agent1.opt.get_maximum() 


    plt.tight_layout()
    plt.savefig(f'plots_3A/agents_iteration_{iteration}.png')  
    plt.close()

print("========= BAYESIAN OPTIMIZATION COMPLETED =========")

print("Plotting reward over iterations...")

# Load rewards from text file to find the best iteration
rewards = []
iterations_list = []
with open(f'{agent_data_dir}/rewards.txt', 'r') as f:
    next(f)  # Skip header
    for line in f:
        iteration_str, reward_str = line.strip().split(',')
        iteration = int(iteration_str)
        reward = float(reward_str)
        iterations_list.append(iteration)
        rewards.append(reward)

# Find the best experimental iteration

max_reward = max(rewards)
max_reward_index = rewards.index(max_reward)
best_iteration = iterations_list[max_reward_index]
print(f'Best Experimental Iteration: {best_iteration} | Reward - {max_reward}')
print(f'Agent 1 Kp, Kd: {agent1.kp_values[best_iteration]}')


# Plot Kp values over iterations
iterations = np.arange(0, N+1)

# Load rewards from text file
agent1_iterations = []
agent1_rewards = []
agent2_iterations = []
agent2_rewards = []
agent3_iterations = []  
agent3_rewards = []     

with open(f'{agent_data_dir}/agent1_data.txt', 'r') as f1:
    reader = csv.reader(f1)
    next(reader)  # Skip header
    for row in reader:
        agent1_iterations.append(int(row[0]))
        agent1_rewards.append(float(row[3]))

with open(f'{agent_data_dir}/agent2_data.txt', 'r') as f2:
    reader = csv.reader(f2)
    next(reader)  # Skip header
    for row in reader:
        agent2_iterations.append(int(row[0]))
        agent2_rewards.append(float(row[3]))

with open(f'{agent_data_dir}/agent3_data.txt', 'r') as f3:  
    reader = csv.reader(f3)
    next(reader)  # Skip header
    for row in reader:
        agent3_iterations.append(int(row[0]))
        agent3_rewards.append(float(row[3]))

plt.figure()
plt.plot(agent1_iterations, agent1_rewards, label='Agent 1 Reward')
plt.plot(agent2_iterations, agent2_rewards, label='Agent 2 Reward')
plt.plot(agent3_iterations, agent3_rewards, label='Agent 3 Reward')  
plt.ylabel('Reward') 
plt.legend()
plt.title('Reward over iterations')
plt.grid(True)
plt.savefig('plots_3A/reward_over_iterations.png')  
plt.show()

# Call the function to plot the best iteration
plot_iteration(best_iteration)


