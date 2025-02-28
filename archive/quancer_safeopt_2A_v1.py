"""
# Version: 1.0
# Description: This script performs Bayesian Optimization for two agents controlling a servo motor system. 
# The goal is to find the optimal control gains (Kp and Kd) that maximize a reward function based on tracking performance and minimizing overshoot errors.
# The script uses a combination of remote commands, data retrieval, reward computation, and optimization over multiple iterations to achieve the best parameter settings for the system.


"""

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
from plot_iteration import plot_iteration

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
    # Create 'data' directory if it doesn't exist
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # Save the .mat files in the 'data' directory
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

def compute_reward(theta_d, rt_theta1, rt_theta2, rt_t1, rt_t2):
    """
    Computes the total reward for the agents
        
    Reward = integral{abs(theta_d - rt_theta)}dt
        
    Returns:
    total_error: Total reward
    os1: Overshoot error for agent 1
    os2: Overshoot error for agent 2
        
    """
    # Overshoot error
    os1 = np.abs(theta_d - rt_theta1)
    os2 = np.abs(theta_d - rt_theta2)
    # Compute error between the two agents
    error12 = np.abs(rt_theta1 - rt_theta2)

    # Compute integral of errors
    integral_os1 = np.trapz(os1, rt_t1)
    integral_os2 = np.trapz(os2, rt_t2)
    total_os = 0.5*integral_os1 + 0.5*integral_os2
    
    integral_error12 = np.trapz(error12, rt_t1)
    
    total_error = total_os + 3* integral_error12
    total_error = 1 / total_error 
    
    os1 = 1 / integral_os1
    os2 = 1 / integral_os2
    
    return total_error, os1, os2
       
def plot_data(rt_t1, rt_theta1, os1, rt_t2, rt_theta2, os2):
    """
    Plot the data from the agents
    """
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(rt_t1, rt_theta1, label='Agent-1')
    plt.plot(rt_t2, rt_theta2, label='Agent-2')
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.ylabel('theta')
    plt.title("Theta over time")
    
    plt.subplot(1,2,2)
    plt.plot(rt_t1[:4000], os1[:4000], label='Agent-1 OS')
    plt.plot(rt_t2[:4000], os2[:4000], label='Agent-2 OS')
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.ylabel('theta')
    plt.title("Error over time")
    plt.legend()
    plt.show()

# Create directories to save data and plots
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('agent_data'):
    os.makedirs('agent_data')

modelName = 'servoPDF'

target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'
target_uri_2 = 'tcpip://172.22.11.10:17000?keep_alive=1'

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'

# Download model to target
sys1dl = f'quarc_run -D -t {target_uri_1} {modelName}.rt-linux_rt_armv7{std_args}'
sys2dl = f'quarc_run -D -t {target_uri_2} {modelName}.rt-linux_rt_armv7{std_args}'

# Run the system commands
subprocess.call(sys1dl, shell=True)
subprocess.call(sys2dl, shell=True)

# Initial safepoint values.

kp1_0 = 5
kd1_0 = 0.7

kp2_0 = 4
kd2_0 = 0.5

x0_1 = (kp1_0, kd1_0)
x0_2 = (kp2_0, kd2_0)

# Delay difference between the two agents
td1 = 0.045

# Create gain arguments
gain_arg1 = f' -Kp {kp1_0} -Kd {kd1_0}'
gain_arg2 = f' -Kp {kp2_0} -Kd {kd2_0}'

print(f'Initial gain arguments for Agent 1: {gain_arg1}')
print(f'Initial gain arguments for Agent 2: {gain_arg2}')

# Create system command for gain arguments
sys1run = f'quarc_run -l -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1} -td {td1:.5f} {std_args}'
sys2run = f'quarc_run -l -t {target_uri_2} {modelName}.rt-linux_rt_armv7{gain_arg2}{std_args}'

# Run the system commands

subprocess.call(sys1run, shell=True)
subprocess.call(sys2run, shell=True)

sent_command(target_uri_1, modelName, gain_arg1, std_args)
sent_command(target_uri_2, modelName, gain_arg2, std_args)

# Wait for the experiment to finish
time.sleep(7)

# Retrieve data from Agents
retrieve_data(target_uri_1, modelName, gain_arg1, std_args, 1, 0)
retrieve_data(target_uri_2, modelName, gain_arg2, std_args, 2, 0)

# Load data from Agents
rt_t1, rt_theta1, theta_d = load_agent_data('data/servoPDF-1_0.mat')
rt_t2, rt_theta2, _ = load_agent_data('data/servoPDF-2_0.mat')

# Compute initial safe reward
reward_0, os1_0, os2_0 = compute_reward(theta_d, rt_theta1, rt_theta2, rt_t1, rt_t2)

print(f'Initial reward: {reward_0}')
print(f"Initial error1: {os1_0}")
print(f"Initial error2: {os2_0}")

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
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, 0.1, beta=4, threshold=0.05)

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
K_bounds = [(0.01, 10), (0.01, 1)]

agent1 = Agent(1, K_bounds, x0_1, reward_0)
agent2 = Agent(2, K_bounds, x0_2, reward_0)

# Quarc Experiment
def run_experiment(kp1, kd1, kp2, kd2, iteration):
    
    # Set gain arguments
    gain_arg1 = f' -Kp {kp1} -Kd {kd1}'
    gain_arg2 = f' -Kp {kp2} -Kd {kd2}'

    sent_command(target_uri_1, modelName, gain_arg1, std_args)
    sent_command(target_uri_2, modelName, gain_arg2, std_args)

    # Await experiment completion
    time.sleep(7)

    retrieve_data(target_uri_1, modelName, gain_arg1, std_args, 1, iteration)
    retrieve_data(target_uri_2, modelName, gain_arg2, std_args, 2, iteration)

    rt_t1, rt_theta1, theta_d = load_agent_data(f'data/servoPDF-1_{iteration}.mat')
    rt_t2, rt_theta2, _ = load_agent_data(f'data/servoPDF-2_{iteration}.mat')
    
    reward, os1, os2 = compute_reward(theta_d, rt_theta1, rt_theta2, rt_t1, rt_t2)

    return reward, os1, os2

N = 50  # Number of iterations

# Initialize data files
agent_data_dir = 'agent_data'
if not os.path.exists(agent_data_dir):
    os.makedirs(agent_data_dir)

with open(f'{agent_data_dir}/agent1_data.txt', 'w', newline='') as f1, open(f'{agent_data_dir}/agent2_data.txt', 'w', newline='') as f2:
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)
    writer1.writerow(['Iteration', 'Kp', 'Kd', 'Reward'])
    writer2.writerow(['Iteration', 'Kp', 'Kd', 'Reward'])
    writer1.writerow([0, x0_1[0], x0_1[1], reward_0])
    writer2.writerow([0, x0_2[0], x0_2[1], reward_0])

with open(f'{agent_data_dir}/rewards.txt', 'w') as f:
    f.write('Iteration,Reward\n')
    f.write(f"0,{reward_0}\n")

# Bayesian Optimization
for iteration in range(1, N+1):
    
    # Get next Kp values from agents
    K1_next = agent1.optimize()
    K2_next = agent2.optimize()

    print(f"Iteration {iteration}, Agent 1:  -Kp {K1_next[0]} -Kd {K1_next[1]}, Agent 2: -Kp {K2_next[0]} -Kd {K2_next[1]}")

    # Run the experiment with kp1_next and kp2_next
    y, os1, os2 = run_experiment(K1_next[0], K1_next[1], K2_next[0], K2_next[1], iteration)

    print(f"Reward: {y}")
    
    # Update agents with observations
    agent1.update(K1_next, y)
    agent2.update(K2_next, y)
    
    # Save agent's data to text files
    with open(f'{agent_data_dir}/agent1_data.txt', 'a', newline='') as f1, open(f'{agent_data_dir}/agent2_data.txt', 'a', newline='') as f2:
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer1.writerow([iteration, K1_next[0], K1_next[1], y])
        writer2.writerow([iteration, K2_next[0], K2_next[1], y])

    # Save rewards to a text file
    with open(f'{agent_data_dir}/rewards.txt', 'a') as f:
        f.write(f"{iteration},{y}\n")

    # Plot and save agents' opt plots in one figure with two subplots
    x_max_1, y_max_1 = agent1.opt.get_maximum()
    x_max_2, y_max_2 = agent2.opt.get_maximum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Agent 1 plot
    agent1.opt.plot(100, axes[0])
    axes[0].scatter(x_max_1[0], x_max_1[1], marker="*", color='red', s=100, label='Current Maximum')
    axes[0].set_title(f'Agent 1 - Iteration {iteration}')
    axes[0].set_xlabel('Kp')
    axes[0].set_ylabel('Kd')
    axes[0].legend()

    # Agent 2 plot
    agent2.opt.plot(100, axes[1])
    axes[1].scatter(x_max_2[0], x_max_2[1], marker="*", color='red', s=100, label='Current Maximum')
    axes[1].set_title(f'Agent 2 - Iteration {iteration}')
    axes[1].set_xlabel('Kp')
    axes[1].set_ylabel('Kd')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'plots/agents_iteration_{iteration}.png')
    plt.close()
    # plt.close('all')

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
print(f'Best Experimental Iteration: {best_iteration} | R - {max_reward}, K1 - {agent1.kp_values[best_iteration]}, K2 - {agent2.kp_values[best_iteration]}')

# Find the estimated best Kp values for the agents

# Plot Kp values over iterations
iterations = np.arange(0, N+1)

# Load rewards from text file
agent1_iterations = []
agent1_rewards = []
agent2_iterations = []
agent2_rewards = []

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

plt.figure()
plt.plot(agent1_iterations, agent1_rewards, label='Agent 1 Reward')
plt.plot(agent2_iterations, agent2_rewards, label='Agent 2 Reward')
plt.xlabel('Iteration')
plt.ylabel('Reward') 
plt.legend()
plt.title('Reward over iterations')
plt.grid(True)
plt.savefig('plots/reward_over_iterations.png')
plt.show()


# Call the function to plot the best iteration
plot_iteration(best_iteration)




print(f'========== EXECUTING BEST ESTIMATED PARAMETERS ==========')
# Get the estimated best parameters from the GP models
x_estimated_1, y_estimated_1 = agent1.opt.get_maximum()
x_estimated_2, y_estimated_2 = agent2.opt.get_maximum()


# Increment the iteration number for the final experiment
final_iteration = N + 2
print(f"Running experiment with estimated best parameters at iteration {final_iteration}")

# Run the experiment with the estimated best parameters
y_estimated, os1_estimated, os2_estimated = run_experiment(x_estimated_1[0], x_estimated_1[1], x_estimated_2[0], x_estimated_2[1], final_iteration)


print(f"Reward: {y_estimated}")

# Update agents with observations from the final experiment
agent1.update(x_estimated_1, y_estimated)
agent2.update(x_estimated_2, y_estimated)

with open(f'{agent_data_dir}/agent1_data.txt', 'a', newline='') as f1, open(f'{agent_data_dir}/agent2_data.txt', 'a', newline='') as f2:
    writer1 = csv.writer(f1)
    writer2 = csv.writer(f2)
    writer1.writerow([final_iteration, x_estimated_1[0], x_estimated_1[1], y_estimated])
    writer2.writerow([final_iteration, x_estimated_2[0], x_estimated_2[1], y_estimated])

with open(f'{agent_data_dir}/rewards.txt', 'a') as f:
    f.write(f"{final_iteration},{y_estimated}\n")

plot_iteration(final_iteration)







