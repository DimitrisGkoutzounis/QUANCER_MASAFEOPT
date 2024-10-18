"""
Bayesian optimization, using SafeOpt, on a single agent platform while tunning Kp and Kd.
"""

import subprocess
import time
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

import safeopt
import GPy

################ PHASE-1 | SAFE EXPERIMENT ################

def sent_command(target_uri, modelName, gain_arg, std_args):
    """
    Send command to the target.
    """
    sys_run = f'quarc_run -l -t {target_uri} {modelName}.rt-linux_rt_armv7{gain_arg} {std_args}'
    subprocess.call(sys_run, shell=True)
    

def retrieve_data(target_uri, modelName, gain_arg, std_args,agent):
    """
    Retrieve data from the target.
    """
    sys_get = f'quarc_run -u -t {target_uri} {modelName}.rt-linux_rt_armv7{gain_arg}{std_args}'
    print(sys_get)
    subprocess.call(sys_get, shell=True)
    shutil.copyfile('servoPDF.mat', f'servoPDF-{agent}.mat')


    
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

    
def compute_reward(theta_d, rt_theta1, rt_t1):
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
    # Compute error between the two agents

    # Compute integral of errors
    integral_os1 = np.trapz(os1, rt_t1)
    total_os = integral_os1
    total_error = total_os
    total_error = 1 / total_error
    
    return total_error,os1
       
def plot_data(rt_t1, rt_theta1, os1):
    """
    Plot the data from the agents
    """
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(rt_t1, rt_theta1, label='Agent-1')
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.ylabel('theta')
    plt.title("Theta over time")
    
    plt.subplot(1,2,2)
    plt.plot(rt_t1[:4000], os1[:4000], label='Agent-1 OS')
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.ylabel('theta')
    plt.title("Error over time")
    plt.legend()
    plt.show()


modelName = 'servoPDF'

target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'

#Download model to target
sys1dl = f'quarc_run -D -t {target_uri_1} {modelName}.rt-linux_rt_armv7{std_args}'

# Run the system commands
subprocess.call(sys1dl, shell=True)

# Initial safepoint values.


kp1_0 = 5
kd1_0 = 0.2

x0 = (kp1_0,kd1_0)

#delay difference between the two agents
td1 = 0.045

# create gain arguments
gain_arg1 = f' -Kp {kp1_0} -Kd {kd1_0}'

print(f'Initial gain arguments for Agent 1: {gain_arg1}')

# create system commnand for gain arguments
sys1run = f'quarc_run -l -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1} -td {td1:.5f} {std_args}'

# Run the system commands

subprocess.call(sys1run, shell=True)
# subprocess.call(sys2run, shell=True)

sent_command(target_uri_1, modelName, gain_arg1, std_args)

# wait for the experiment to finish
# this should be replaced with a more robust method, where the script waits for the experiment to finish
# Possibly by checking the port for incoming data
time.sleep(6.5)


#retrieve data from Agent 1
retrieve_data(target_uri_1, modelName, gain_arg1, std_args,1)

# Load data for agent 1
rt_t1, rt_theta1,theta_d = load_agent_data('servoPDF-1.mat')

#compute initial safe reward
reward_0, os1_0 = compute_reward(theta_d,rt_theta1,rt_t1)

print(f'Initial reward: {reward_0}')

# plot_data(rt_t1, rt_theta1, os1_0, rt_t2, rt_theta2, os2_0)
wait = input("Press Enter to continue...")


# =================== PHASE-2 | Bayesian Optimization ===================

class Agent:
    def __init__(self, id, bounds, safe_point,initial_reward):
        self.id = id
        self.bounds = bounds
        self.safe_point = safe_point

        self.x0 = np.asarray([safe_point])
        self.y0 = np.asarray([[initial_reward]]) 

        self.kernel = GPy.kern.RBF(input_dim=len(bounds),ARD=True)
        self.gp = GPy.models.GPRegression(self.x0, self.y0, self.kernel, noise_var=0.05**2)

        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, 0.3, beta=5,threshold=0.1)

        self.kp_values = []
        self.rewards = []

    def optimize(self):
        x_next = self.opt.optimize()
        print(x_next)
        return x_next

    def update(self, x_next, y_meas):
        self.opt.add_new_data_point(x_next, y_meas)
        self.kp_values.append(x_next)
        self.rewards.append(y_meas)

# Kp bounds
kp_bounds = [(0.01, 10), (0.01, 1)]

agent1 = Agent(1, kp_bounds, x0, reward_0)

wait = input("Press Enter to continue...")

# Initial Values
print("Initial input: ", agent1.opt.x)
print("Intital output:", agent1.opt.y)

# Quancer Experiment
def run_experiment(kp1,kd1):
    
    # set gain arguments
    gain_arg1 = f' -Kp {kp1} -Kd {kd1}'
    
    sent_command(target_uri_1, modelName, gain_arg1, std_args)

    # await experiment completion
    time.sleep(6.5)    
    retrieve_data(target_uri_1, modelName, gain_arg1, std_args,1)

    rt_t1, rt_theta1,theta_d = load_agent_data('servoPDF-1.mat')
        
    reward, os1 = compute_reward(theta_d,rt_theta1,rt_t1)

    # plot_data(rt_t1, rt_theta1, os1)

    return reward,os1


N = 50  # Number of iterations

# Bayesian Optimization
for iteration in range(N):

    # Get next Kp values from agents
    K_next = agent1.optimize()
    
    print("K_next: ", K_next)

    print(f"Iteration {iteration}, Agent 1 Kp: {K_next[0]} Kd: {K_next[1]}")

    y,_ = run_experiment(K_next[0],K_next[1])

    print(f"Reward: {y}")

    
    # wait = input("Press Enter to continue...")

    # Update agents with observations
    
    agent1.update(K_next, y)

    # agent1.opt.plot(100)
    # plt.show()
    
    
print("========= EXPERIMENT COMPLETE =========")

# Plot Kp values over iterations
iterations = np.arange(0,N+1)


plt.figure(2)
y = agent1.opt.y
plt.plot(iterations.flatten(),agent1.opt.y.flatten(), label='Agent 1 Kp')
plt.xlabel('Iteration')
plt.ylabel('Error') 
plt.legend()
plt.title('Error over iterations')
plt.grid(True)

agent1.opt.plot(100)
plt.show()

