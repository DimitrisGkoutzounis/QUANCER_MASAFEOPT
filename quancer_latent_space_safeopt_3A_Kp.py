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
    data_dir = 'data_3A'  
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
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
    total_os = (integral_os1 + integral_os2 + integral_os3) / 3 

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
    
    

# Define the compute_gradient function
def compute_gradient(model, X):
    dmu_dX, _ = model.predictive_gradients(X)
    return dmu_dX


def column_wise(Z_flat, X, D, N, sigma2, f):
    Z = Z_flat.reshape(N, D)

    # Define model_Z with R_z as observations
    model_Z = GPy.models.GPRegression(Z, R.reshape(-1,1), GPy.kern.RBF(D))
    model_all = GPy.models.GPRegression(Z, X,  GPy.kern.RBF(D))
    mu_all, _ = model_all.predict_noiseless(Z)
    # print("mu_all\n", mu_all)

    loss = 0.0
    grad_R_Z_norm_column = []
    grad_R_X_norm_column = []

    # Initialize matrices for U_z and U_x
    U_z = np.zeros((N, D))
    U_x = np.zeros((N, D))
    
    action_term = 0.0

    for d in range(D):
        X_d = np.zeros_like(X)
        X_d[:, d] = X[:, d]
        
        model_d = GPy.models.GPRegression(Z, X_d,GPy.kern.RBF(D))
        mu_d, _ = model_d.predict_noiseless(Z)
        

        diff1 = np.linalg.norm(X_d - mu_d)**2
        diff2 = np.linalg.norm(mu_d - mu_all[:, [d]])**2
        
        action_term += 0.1 * diff1 + 0.2 * diff2

        # Gradient-based alignment term
        grad_R_Z = compute_gradient(model_Z, Z).reshape(N, D)
        grad_R_X = compute_gradient(model_X, X).reshape(N, D)

        grad_R_Z_norm_column.append(np.linalg.norm(grad_R_Z[:, d]))
        grad_R_X_norm_column.append(np.linalg.norm(grad_R_X[:, d]))

        U_z[:, d] = grad_R_Z[:, d] / grad_R_Z_norm_column[d]
        U_x[:, d] = grad_R_X[:, d] / grad_R_X_norm_column[d]

    dot_product_matrix = np.dot(U_z.T, U_x)
    diag_penalty = np.linalg.norm((1 - np.diag(dot_product_matrix))**2)/D
    
    total_loss = action_term + diag_penalty 


    return total_loss


# Define the Agent class for Bayesian Optimization
class Agent:
    def __init__(self, id, bounds, safe_point, initial_reward):
        self.id = id
        self.bounds = bounds
        self.safe_point = safe_point

        self.x0 = np.asarray([safe_point])
        print(self.x0)
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

x0_1 = [(kp1_0)]
x0_2 = [(kp2_0)]
x0_3 = [(kp3_0)] 

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


# Kp bounds
K_bounds = [(0.01, 10)]

agent1 = Agent(1, K_bounds, x0_1, reward_0)
agent2 = Agent(2, K_bounds, x0_2, reward_0)
agent3 = Agent(3, K_bounds, x0_3, reward_0) 

Kd1 = 0.7
Kd2 = 0.7
Kd3 = 0.7 

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
    writer1.writerow([0, x0_1[0], reward_0])
    writer2.writerow([0, x0_2[0], reward_0])
    writer3.writerow([0, x0_3[0], reward_0]) 

with open(f'{agent_data_dir}/rewards.txt', 'w') as f:
    f.write('Iteration,Reward\n')
    f.write(f"0,{reward_0}\n")

# Bayesian Optimization
for iteration in range(1, N+1):
    
    # Get next Kp values from agents
    K1_next = agent1.optimize()
    K2_next = agent2.optimize()
    K3_next = agent3.optimize() 

    print(f"Iteration {iteration}, Agent 1:  -Kp {K1_next[0]} -Kd {Kd1}, Agent 2: -Kp {K2_next[0]} -Kd {Kd2}, Agent 3: -Kp {K3_next[0]} -Kd {Kd3}")

    # Run the experiment with kp1_next, kp2_next, kp3_next
    y, os1, os2, os3 = run_experiment(K1_next[0], Kd1, K2_next[0], Kd2, K3_next[0], Kd3, iteration)

    print(f"Reward: {y}")
    
    # Update agents with observations
    agent1.update(K1_next, y)
    agent2.update(K2_next, y)
    agent3.update(K3_next, y)  
    
    # Save agent's data to text files
    with open(f'{agent_data_dir}/agent1_data.txt', 'a', newline='') as f1, \
         open(f'{agent_data_dir}/agent2_data.txt', 'a', newline='') as f2, \
         open(f'{agent_data_dir}/agent3_data.txt', 'a', newline='') as f3: 
        writer1 = csv.writer(f1)
        writer2 = csv.writer(f2)
        writer3 = csv.writer(f3) 
        writer1.writerow([iteration, K1_next[0], y])
        writer2.writerow([iteration, K2_next[0], y])
        writer3.writerow([iteration, K3_next[0], y])  

    # Save rewards to a text file
    with open(f'{agent_data_dir}/rewards.txt', 'a') as f:
        f.write(f"{iteration},{y}\n")

    # Plot and save agents' opt plots in one figure with three subplots
    x_max_1, y_max_1 = agent1.opt.get_maximum()
    x_max_2, y_max_2 = agent2.opt.get_maximum()
    x_max_3, y_max_3 = agent3.opt.get_maximum()  

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  

    # Agent 1 plot
    agent1.opt.plot(100, axes[0])
    axes[0].scatter(x_max_1[0],y_max_1, marker="*", color='red', s=100, label='Current Maximum')
    axes[0].set_title(f'Agent 1 - Iteration {iteration}')
    axes[0].set_xlabel('Kp')
    axes[0].set_ylabel('Reward')
    axes[0].legend()

    # Agent 2 plot
    agent2.opt.plot(100, axes[1])
    axes[1].scatter(x_max_2[0],y_max_2, marker="*", color='red', s=100, label='Current Maximum')
    axes[1].set_title(f'Agent 2 - Iteration {iteration}')
    axes[1].set_xlabel('Kp')
    axes[1].set_ylabel('Reward')
    axes[1].legend()

    # Agent 3 plot
    agent3.opt.plot(100, axes[2])
    axes[2].scatter(x_max_3[0],y_max_3, marker="*", color='red', s=100, label='Current Maximum')
    axes[2].set_title(f'Agent 3 - Iteration {iteration}')
    axes[2].set_xlabel('Kp')
    axes[2].set_ylabel('Reward')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'plots_3A/agents_iteration_{iteration}.png')  
    plt.close()

print("========= BAYESIAN OPTIMIZATION COMPLETED =========")

# # =================================================
# # After Bayesian Optimization, compute objective function and minimize
# # =================================================


# # Collect data into X and Y
N = len(agent1.kp_values)  # Number of data points (iterations + initial point)
D = 3  # Total number of agents

X1 = np.array(agent1.kp_values).flatten()
X2 = np.array(agent2.kp_values).flatten()
X3 = np.array(agent3.kp_values).flatten()

X = np.vstack((X1, X2,X3)).T

print("X:",X)

Z = np.random.uniform(-1.5, 1.5, (N, D))

print("Z:",Z)


R = np.array(agent1.rewards).flatten()

print("R:",R)

model_X = GPy.models.GPRegression(X, R[:, None], GPy.kern.RBF(input_dim=D))


# print(Z_init.shape)
# pri
result = minimize(column_wise, Z.flatten(), args=(X, D, N, 1e-2, f), method='L-BFGS-B',options={'ftol':1e-2,'gtol':1e-2})
Z_opt = result.x.reshape(N, D)


print("Z_opt:",Z_opt)   


wait = input("go to Dafni")


# Build GP models to map Z to X using the data collected


# For Kp
Z_to_X_0 = GPy.models.GPRegression(Z_opt[:, 0].reshape(-1,1), X[:, 0].reshape(-1,1), kernel=GPy.kern.RBF(1))
Z_to_X_1 = GPy.models.GPRegression(Z_opt[:, 1].reshape(-1,1), X[:, 1].reshape(-1,1), kernel=GPy.kern.RBF(1))
Z_to_X_2 = GPy.models.GPRegression(Z_opt[:, 2].reshape(-1,1), X[:, 2].reshape(-1,1), kernel=GPy.kern.RBF(1))

actions_1 = np.array([])
actions_2 = np.array([])
actions_3 = np.array([])

for iteration in range(0, 5):
    # Get next Z values from agents
    Z1_next = agent1.optimize()
    Z2_next = agent2.optimize()
    Z3_next = agent3.optimize()
    # Z --> X mapping
    Kp1_next, _ = Z_to_X_0.predict(Z1_next[0].reshape(-1,1))
    Kp2_next, _ = Z_to_X_1.predict(Z2_next[0].reshape(-1,1))
    Kp3_next, _ = Z_to_X_2.predict(Z3_next[0].reshape(-1,1))
    
    Kp1_next = np.asarray([Kp1_next]).flatten()
    Kp2_next = np.asarray([Kp2_next]).flatten()
    Kp3_next = np.asarray([Kp3_next]).flatten()
    
    actions_1 = np.append(actions_1, Kp1_next)
    actions_2 = np.append(actions_2, Kp2_next)
    actions_3 = np.append(actions_3, Kp3_next)
    
    wait = input("Next...")

    # Run the experiment with the mapped Kp and Kd values
    y, os1, os2, os3 = run_experiment(Kp1_next[0], Kd1, Kp2_next[0], Kd2, Kp3_next[0], Kd3, iteration)

    print(f"Reward: {y}")

    # Update agents with observations
    agent1.update(Z1_next, y)
    agent2.update(Z2_next, y)
    agent3.update(Z3_next, y)


agent1.opt.plot(100)
agent2.opt.plot(100)
agent3.opt.plot(100)





   

# print("========= SECOND PHASE COMPLETED =========")


