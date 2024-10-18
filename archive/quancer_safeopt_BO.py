import subprocess
import time
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

import safeopt
import GPy

################ PHASE 1 ################

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
    error12 = rt_theta1 - rt_theta2

    # Compute integral of errors
    integral_os1 = np.trapz(os1, rt_t1)
    integral_os2 = np.trapz(os2, rt_t2)
    total_os = 1*integral_os1 + 0*integral_os2
    integral_error12 = np.trapz(error12, rt_t1)
    total_error = total_os
    
    return total_error,os1,os2
       
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


modelName = 'servoPDF'

target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'
target_uri_2 = 'tcpip://172.22.11.10:17000?keep_alive=1'

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'

#Download model to target
sys1dl = f'quarc_run -D -t {target_uri_1} {modelName}.rt-linux_rt_armv7{std_args}'
sys2dl = f'quarc_run -D -t {target_uri_2} {modelName}.rt-linux_rt_armv7{std_args}'

# Run the system commands
subprocess.call(sys1dl, shell=True)
subprocess.call(sys2dl, shell=True)

# Initial safepoint values.


kp1_0 = 10
kd1_0 = 0.5

kp2_0 = 1
kd2_0 = 0.2


#delay difference between the two agents
td1 = 0.045

# create gain arguments
gain_arg1 = f' -Kp {kp1_0} -Kd {kd1_0}'
gain_arg2 = f' -Kp {kp2_0} -Kd {kd2_0}'

print(f'Initial gain arguments for Agent 1: {gain_arg1}')
print(f'Initial gain arguments for Agent 2: {gain_arg2}')

# create system commnand for gain arguments
sys1run = f'quarc_run -l -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1} -td {td1:.5f} {std_args}'
sys2run = f'quarc_run -l -t {target_uri_2} {modelName}.rt-linux_rt_armv7{gain_arg2}{std_args}'

# Run the system commands

subprocess.call(sys1run, shell=True)
# subprocess.call(sys2run, shell=True)

sent_command(target_uri_1, modelName, gain_arg1, std_args)
sent_command(target_uri_2, modelName, gain_arg2, std_args)

# wait for the experiment to finish
# this should be replaced with a more robust method, where the script waits for the experiment to finish
# Possibly by checking the port for incoming data
time.sleep(7)


#retrieve data from Agent 1
retrieve_data(target_uri_1, modelName, gain_arg1, std_args,1)

#retrieve data from Agent 2
retrieve_data(target_uri_2, modelName, gain_arg1, std_args,2)

# Load data for agent 1
rt_t1, rt_theta1,theta_d = load_agent_data('servoPDF-1.mat')

#load data for agent 2
rt_t2, rt_theta2, _ = load_agent_data('servoPDF-2.mat')

#compute initial safe reward
reward_0, os1_0 , os2_0 = compute_reward(theta_d,rt_theta1,rt_theta2,rt_t1,rt_t2)

print(f'Initial reward: {reward_0}')

# plot_data(rt_t1, rt_theta1, os1_0, rt_t2, rt_theta2, os2_0)
# exit(0)

# =================== Bayesian Optimization ===================

# Define the Agent class for Bayesian Optimization
class Agent:
    def __init__(self, id, bounds, safe_point,initial_reward):
        self.id = id
        self.bounds = [bounds]
        self.safe_point = safe_point

        self.x0 = np.array([[safe_point]])
        self.y0 = np.array([[initial_reward]]) 

        self.kernel = GPy.kern.RBF(1)
        self.gp = GPy.models.GPRegression(self.x0, self.y0, self.kernel, noise_var=0.05**2)

        self.parameter_set = safeopt.linearly_spaced_combinations(self.bounds, 100)
        self.opt = safeopt.SafeOpt(self.gp, self.parameter_set, -np.inf, beta=10,threshold=0.2)

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
kp_bounds = (0.01, 400)

agent1 = Agent(1, kp_bounds, kp1_0,-reward_0)
agent2 = Agent(2, kp_bounds, kp2_0,-reward_0)

# Quancer Experiment
def run_experiment(kp1, kp2):
    kp1= kp1[0]
    kp2= kp2[0]
    

    # set gain arguments
    gain_arg1 = f' -Kp {kp1} -Kd {kd1_0}'
    gain_arg2 = f' -Kp {kp2} -Kd {kd2_0}'

    
    sent_command(target_uri_1, modelName, gain_arg1, std_args)
    sent_command(target_uri_2, modelName, gain_arg2, std_args)

    # await experiment completion
    time.sleep(6.5)

    
    retrieve_data(target_uri_1, modelName, gain_arg1, std_args,1)
    retrieve_data(target_uri_2, modelName, gain_arg2, std_args,2)

    rt_t1, rt_theta1,theta_d = load_agent_data('servoPDF-1.mat')
    rt_t2, rt_theta2, _ = load_agent_data('servoPDF-2.mat')
    
    
    reward, os1 , os2 = compute_reward(theta_d,rt_theta1,rt_theta2,rt_t1,rt_t2)

    plot_data(rt_t1, rt_theta1, os1, rt_t2, rt_theta2, os2)

    return reward,os1, os2


N = 10  # Number of iterations

# Bayesian Optimization
for iteration in range(N):
    # Get next Kp values from agents
    kp1_next = agent1.optimize()
    kp2_next = agent2.optimize()

    print(f"Iteration {iteration}, Agent 1 Kp: {kp1_next}, Agent 2 Kp: {kp2_next}")

    y,_,_ = run_experiment(kp1_next, kp2_next)

    print(f"Reward: {y}")
    

    # Update agents with observations
    
    agent1.update(kp1_next, y)
    agent2.update(kp2_next, y)
    
    
print("========= EXPERIMENT COMPLETE =========")

# Plot Kp values over iterations
iterations = range(len(agent1.kp_values))

plt.figure(3)

plt.plot(iterations,agent1.opt.y, label='Agent 1 Kp')
plt.plot(iterations,agent2.opt.y, label='Agent 2 Kp')
plt.xlabel('Iteration')
plt.ylabel('Error') 
plt.legend()
plt.title('Error over iterations')
plt.grid(True)
plt.show()

