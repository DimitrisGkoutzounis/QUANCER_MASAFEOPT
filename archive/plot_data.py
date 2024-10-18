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
    # Check if plots directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    data_dir = 'data'
    agent_data_dir = 'agent_data'

    # Load data
    rt_t1, rt_theta1, theta_d = load_agent_data(f'{data_dir}/servoPDF-1_{iteration_number}.mat')
    rt_t2, rt_theta2, _ = load_agent_data(f'{data_dir}/servoPDF-2_{iteration_number}.mat')
    
    # Plot state response
    plt.figure()
    plt.plot(rt_t1, rt_theta1, label='Agent-1')
    plt.plot(rt_t2, rt_theta2, label='Agent-2')
    plt.plot(rt_t1, theta_d, label='Desired Theta', linestyle='--')
    plt.grid(True)
    plt.xlabel('t (s)')
    plt.ylabel('theta')
    plt.title(f"Theta over time - Iteration {iteration_number}")
    plt.legend()
    plt.savefig(f'plots/state_response_iteration_{iteration_number}.png')
    plt.close()
    
    # Load agent data from text files
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
    
    # Plot reward over iterations
    plt.figure()
    plt.plot(agent1_iterations, agent1_rewards, label='Agent 1 Reward')
    plt.plot(agent2_iterations, agent2_rewards, label='Agent 2 Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('Reward over Iterations')
    plt.legend()
    plt.savefig(f'plots/reward_plot_iteration_{iteration_number}.png')
    plt.close()

    # Reconstruct GP models and plot if desired
    reconstruct_and_plot_gp(agent1_kp, agent1_kd, agent1_rewards, 'Agent 1', iteration_number)
    reconstruct_and_plot_gp(agent2_kp, agent2_kd, agent2_rewards, 'Agent 2', iteration_number)

def reconstruct_and_plot_gp(kp_values, kd_values, rewards, agent_name, iteration_number):
    # Prepare data
    X = np.array(list(zip(kp_values, kd_values)))
    Y = np.array(rewards).reshape(-1, 1)

    # Define bounds
    K_bounds = [(0.01, 10), (0.01, 1)]  # Same as in the main script

    # Reconstruct GP model
    kernel = GPy.kern.RBF(input_dim=2, ARD=True)
    gp = GPy.models.GPRegression(X, Y, kernel, noise_var=0.05**2)

    # Create SafeOpt object
    parameter_set = safeopt.linearly_spaced_combinations(K_bounds, 100)
    opt = safeopt.SafeOpt(gp, parameter_set, 0.1, beta=4, threshold=0.05)

    # Get current maximum
    x_max, y_max = opt.get_maximum()

    # Plot GP model
    plt.figure()
    opt.plot(100)
    plt.scatter(x_max[0], x_max[1], marker='*', color='red', s=100, label='Current Maximum')
    plt.title(f'{agent_name} - Iteration {iteration_number}')
    plt.xlabel('Kp')
    plt.ylabel('Kd')
    plt.legend()
    plt.savefig(f'plots/{agent_name.lower().replace(" ", "_")}_plot_iteration_{iteration_number}.png')
    plt.close()

if __name__ == "__main__":
    iteration_to_plot = int(input("Enter iteration number to plot: "))
    plot_iteration(iteration_to_plot)
    print(f"Plots for iteration {iteration_to_plot} have been saved in the 'plots' directory.")
