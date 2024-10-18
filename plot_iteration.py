# plot_iteration.py

import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import csv
import os
import GPy
import safeopt

def load_agent_data(filename):
    """
    Load data from the agent and extract relevant data
    """
    data = loadmat(filename)
    rt_t = data['rt_t'].flatten()
    rt_theta = data['rt_theta'].flatten()
    theta_d = data['rt_theta_d'].flatten()
    return rt_t, rt_theta, theta_d

def plot_iteration(iteration_number):
    """
    Plot the data for a specific iteration.
    The plot includes three subplots:
    - Agent 1 opt plot
    - Agent 2 opt plot
    - State response
    """
    # Check if 'plots' directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    data_dir = 'data'
    agent_data_dir = 'agent_data'

    # Load agent data up to the specified iteration
    agent1_iterations = []
    agent1_kp = []
    agent1_kd = []
    agent1_rewards = []

    agent2_iterations = []
    agent2_kp = []
    agent2_kd = []
    agent2_rewards = []

    with open(f'{agent_data_dir}/agent1_data.txt', 'r') as f1:
        reader = csv.reader(f1)
        next(reader)  # Skip header
        for row in reader:
            iter_num = int(row[0])
            if iter_num <= iteration_number:
                agent1_iterations.append(iter_num)
                agent1_kp.append(float(row[1]))
                agent1_kd.append(float(row[2]))
                agent1_rewards.append(float(row[3]))

    with open(f'{agent_data_dir}/agent2_data.txt', 'r') as f2:
        reader = csv.reader(f2)
        next(reader)  # Skip header
        for row in reader:
            iter_num = int(row[0])
            if iter_num <= iteration_number:
                agent2_iterations.append(iter_num)
                agent2_kp.append(float(row[1]))
                agent2_kd.append(float(row[2]))
                agent2_rewards.append(float(row[3]))

    # Reconstruct GP models
    X1 = np.array(list(zip(agent1_kp, agent1_kd)))
    Y1 = np.array(agent1_rewards).reshape(-1, 1)

    X2 = np.array(list(zip(agent2_kp, agent2_kd)))
    Y2 = np.array(agent2_rewards).reshape(-1, 1)

    # Define bounds
    K_bounds = [(0.01, 10), (0.01, 1)]  # Same as in the main script

    # Reconstruct GP models
    kernel1 = GPy.kern.RBF(input_dim=2, ARD=True)
    gp1 = GPy.models.GPRegression(X1, Y1, kernel1, noise_var=0.05**2)

    kernel2 = GPy.kern.RBF(input_dim=2, ARD=True)
    gp2 = GPy.models.GPRegression(X2, Y2, kernel2, noise_var=0.05**2)

    # Create SafeOpt objects
    parameter_set = safeopt.linearly_spaced_combinations(K_bounds, 100)
    opt1 = safeopt.SafeOpt(gp1, parameter_set, 0.1, beta=4, threshold=0.05)
    opt2 = safeopt.SafeOpt(gp2, parameter_set, 0.1, beta=4, threshold=0.05)

    # Get current maximum
    x_max_1, y_max_1 = opt1.get_maximum()
    x_max_2, y_max_2 = opt2.get_maximum()

    # Load state response data for the specified iteration
    rt_t1, rt_theta1, theta_d = load_agent_data(f'{data_dir}/servoPDF-1_{iteration_number}.mat')
    rt_t2, rt_theta2, _ = load_agent_data(f'{data_dir}/servoPDF-2_{iteration_number}.mat')

    # Plot the figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    # Agent 1 plot
    opt1.plot(100,axes[0])
    axes[0].scatter(x_max_1[0], x_max_1[1], marker="*", color='red', s=100, label='Current Maximum')
    axes[0].set_title(f'Agent 1 - Iteration {iteration_number}')
    axes[0].set_xlabel('Kp')
    axes[0].set_ylabel('Kd')
    axes[0].legend()

    # Agent 2 plot
    opt2.plot(100,axes[1])
    axes[1].scatter(x_max_2[0], x_max_2[1], marker="*", color='red', s=100, label='Current Maximum')
    axes[1].set_title(f'Agent 2 - Iteration {iteration_number}')
    axes[1].set_xlabel('Kp')
    axes[1].set_ylabel('Kd')
    axes[1].legend()

    # State response plot
    axes[2].plot(rt_t1, rt_theta1, label='Agent-1')
    axes[2].plot(rt_t2, rt_theta2, label='Agent-2')
    axes[2].plot(rt_t1, theta_d, label='Desired Theta', linestyle='--')
    axes[2].grid(True)
    axes[2].set_xlabel('t (s)')
    axes[2].set_ylabel('theta')
    axes[2].set_title(f"Theta over time - Iteration {iteration_number}")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f'plots/iteration_{iteration_number}.png')
    plt.show()
    
    opt1.plot(100,axis=None,plot_3d=True)
    plt.title(f'Agent 1 - Iteration {iteration_number}')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.savefig(f'plots/iteration_{iteration_number}_agent1_3d.png')
    
    
    opt2.plot(100,axis=None,plot_3d=True)
    plt.title(f'Agent 2 - Iteration {iteration_number}')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.savefig(f'plots/iteration_{iteration_number}_agent2_3d.png')
    
    plt.show()
    
 


if __name__ == "__main__":
    iteration_to_plot = int(input("Enter iteration number to plot: "))
    plot_iteration(iteration_to_plot)
    print(f"Plots for iteration {iteration_to_plot} have been saved in the 'plots' directory.")
