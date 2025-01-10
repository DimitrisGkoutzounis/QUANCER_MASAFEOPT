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
import os  
import csv  
import safeopt
import GPy
from scipy.optimize import minimize
from datetime import datetime




modelName = 'servoPDF'

# Target URIs for the agents
target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'
target_uri_2 = 'tcpip://172.22.11.10:17000?keep_alive=1'
target_uri_3 = 'tcpip://172.22.11.18:17000?keep_alive=1'  

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'

w1 = 0
w2 = 0
w3 = 0
w4 = 1


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
    subprocess.call(sys_get, shell=True)
    data_dir = 'data_3A_baseline'  
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

def load_z_data_temp(z_data_dir,iteration):
    
            agent1_x1 = []
            agent2_x2 = []
            agent3_x3 = []
            rewards = []

            with open(f'{z_data_dir}/Z_temp_data_{iteration}.txt', 'r') as f1:
                reader = csv.reader(f1)
                next(reader)  # Skip header
                for row in reader:
                    agent1_x1.append(float(row[1]))
                    agent2_x2.append(float(row[2]))
                    agent3_x3.append(float(row[3]))
                    rewards.append(float(row[4]))
                    
            agent1_x1 = np.array(agent1_x1)
            agent2_x2 = np.array(agent2_x2)
            agent3_x3 = np.array(agent3_x3)
            rewards = np.array(rewards).reshape(-1,1)

            return rewards, agent1_x1, agent2_x2, agent3_x3 

def load_initial_data(baseline_dir):
                agent1_x1 = []
                agent2_x2 = []
                agent3_x3 = []
                rewards = []

                with open(f'{baseline_dir}/agent1_data.txt', 'r') as f1:
                    reader = csv.reader(f1)
                    next(reader)  # Skip header
                    for row in reader:
                        agent1_x1.append(float(row[1]))
                        rewards.append(float(row[2]))

                with open(f'{baseline_dir}/agent2_data.txt', 'r') as f2:
                    reader = csv.reader(f2)
                    next(reader)  # Skip header
                    for row in reader:
                        agent2_x2.append(float(row[1]))

                with open(f'{baseline_dir}/agent3_data.txt', 'r') as f3:
                    reader = csv.reader(f3)
                    next(reader)  # Skip header
                    for row in reader:
                        agent3_x3.append(float(row[1]))

                return agent1_x1, agent2_x2, agent3_x3, rewards

def write_z_data(z_data_dir,X1,X2,X3,R_z, iteration):
    
    with open(f'{z_data_dir}/Z_temp_data_{iteration}.txt', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Z1', 'Z2', 'Z3', 'Reward'])
        for i in range(len(R_z)):
            writer.writerow([i, X1[i], X2[i], X3[i],R_z[i]])
            
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

    rt_t1, rt_theta1, theta_d = load_agent_data(f'data_3A_baseline/servoPDF-1_{iteration}.mat') 
    rt_t2, rt_theta2, _ = load_agent_data(f'data_3A_baseline/servoPDF-2_{iteration}.mat')       
    rt_t3, rt_theta3, _ = load_agent_data(f'data_3A_baseline/servoPDF-3_{iteration}.mat')     
    
    reward, os1, os2, os3 = compute_reward(theta_d, rt_theta1, rt_theta2, rt_theta3, rt_t1, rt_t2, rt_t3)

    return reward, os1, os2, os3
    
    

# Define the compute_gradient function
def compute_gradient(model, X):
    dmu_dX, _ = model.predictive_gradients(X)
    return dmu_dX

def column_wise(Z_flat, X, est_R, D, N_sample):
    Z = Z_flat.reshape(N_sample, D)

    # Define model_Z with R_z as observations
    
    model_all = GPy.models.GPRegression(Z, X,  GPy.kern.Matern32(D))
    mu_all, _ = model_all.predict_noiseless(Z)

    loss = 0.0
    action_term = 0.0
    
    
    grad_R_Z_norm_column = []
    grad_R_X_norm_column = []

    # Initialize matrices for U_z and U_x
    U_z = np.zeros((N_sample, D))
    U_x = np.zeros((N_sample, D))
    

    for d in range(D):
        X_d = np.zeros_like(X)
        X_d[:, d] = X[:, d]
        
        model_d = GPy.models.GPRegression(Z, X_d,GPy.kern.Matern32(D))
        mu_d, _ = model_d.predict_noiseless(Z)
        

        diff1 = np.linalg.norm(X_d - mu_d)**2
        diff2 = np.linalg.norm(mu_d - mu_all[:, [d]])**2
        
        action_term += w2 * diff1 + w1 * diff2
        
        # Z_d ---- est_Rd mapping
        model_Zd = GPy.models.GPRegression(Z[:,d].reshape(-1,1), est_R[:,d].reshape(-1,1), GPy.kern.Matern32(1), noise_var=0.01)
        model_Xd = GPy.models.GPRegression(X[:,d].reshape(-1,1), est_R[:,d].reshape(-1,1), GPy.kern.Matern32(1), noise_var=1)

        # Gradient-based alignment term
        grad_R_Zd = compute_gradient(model_Zd, Z[:,d].reshape(-1,1))
        grad_R_Xd = compute_gradient(model_Xd, X[:,d].reshape(-1,1))
        
        
        grad_R_Zd = grad_R_Zd.flatten()
        grad_R_Xd = grad_R_Xd.flatten()
                
        
        grad_R_Z_norm_column.append(np.linalg.norm(grad_R_Zd))
        grad_R_X_norm_column.append(np.linalg.norm(grad_R_Xd))
        

        U_z[:, d] = grad_R_Zd / np.linalg.norm(grad_R_Zd) 
        U_x[:, d] = grad_R_Xd / np.linalg.norm(grad_R_Xd)

    # --- diagonal element ---
    dot_product_matrix = np.dot(U_z.T, U_x)
    # wait = input("Press Enter to continue...")
    print(dot_product_matrix)
    diag_penalty = np.linalg.norm((1 - np.diag(dot_product_matrix))**2)/D
    print(diag_penalty)
    
    # --- off-diagonal elements ---
    sum_upper = np.sum(np.triu(dot_product_matrix, 1)**2) # Upper triangular
    # print(sum_upper)
    # wait = input("Press Enter to continue...")  
    sum_lower = np.sum(np.tril(dot_product_matrix, 1)**2) # Lower triangular
    # print(sum_lower)
    # wait = input("Press Enter to continue...")
    off_diag_penalty = (sum_upper + sum_lower) / (2*D)    
    
    total_loss = action_term + w3 * diag_penalty + w4 * off_diag_penalty


    return total_loss


z_data_dir = 'Z_data'
baseline_dir = 'agent_data_3A_baseline'

if not os.path.exists(z_data_dir):
    os.makedirs(z_data_dir)

if __name__ == '__main__':
    
    today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    
    base_dir = f'latent_experiment_{today}'
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories
    
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    agent_data_dir = os.path.join(base_dir, 'agent_data')
    os.makedirs(agent_data_dir, exist_ok=True)
    
    z_data_dir = os.path.join(base_dir, 'z_data')
    os.makedirs(z_data_dir, exist_ok=True)
    
    # ----- Configuration -----
    
    K_bounds_Z = [(-10,10)]
    beta = 1.0
    safety_threshold = 0.0
    discretization = 1000
    
    K = 4 # Number of experiments
    N = 10  # Number of BO trials
    N_test = 100  # Number of test points
    
    # Delay difference between the agents
    td1 = 0.09
    td2 = 0.045  
    td3 = 0.001 
    
    Kd1 = 0.7
    Kd2 = 0.7
    Kd3 = 0.7
    
    rewards = []

    for j in range(K):
            
        if j == 0:

            agent1_x1 = []
            agent2_x2 = []
            agent3_x3 = []

            agent1_x1, agent2_x2, agent3_x3, rewards = load_initial_data(baseline_dir)

            X1 = np.array(agent1_x1)
            X2 = np.array(agent2_x2)
            X3 = np.array(agent3_x3)
            R = np.array(rewards).reshape(-1,1)
            
            print("Printing initial data..\n")
            print("X1",X1)
            print("X2",X2)
            print("X3",X3)
            
            kernel1 = GPy.kern.Matern32(1)
            kernel2 = GPy.kern.Matern32(1)
            kernel3 = GPy.kern.Matern32(1)

            # Z1 ----> R mapping
            gp1 = GPy.models.GPRegression(X1.reshape(-1,1), R, kernel1, noise_var=0.05**2)
            # Z2 ----> R mapping
            gp2 = GPy.models.GPRegression(X2.reshape(-1,1), R, kernel2, noise_var=0.05**2)
            # Z3 ----> R mapping
            gp3 = GPy.models.GPRegression(X3.reshape(-1,1), R, kernel3, noise_var=0.05**2)

            bounds = [(0.01, 10)]
            latent_parameter_set = safeopt.linearly_spaced_combinations(bounds, discretization)

            # Agent safeopt objects
            opt1 = safeopt.SafeOpt(gp1, latent_parameter_set, safety_threshold, beta, threshold=0.05)
            opt2 = safeopt.SafeOpt(gp2, latent_parameter_set, safety_threshold, beta, threshold=0.05)
            opt3 = safeopt.SafeOpt(gp3, latent_parameter_set, safety_threshold, beta, threshold=0.05)
                
            
            
            
                
            
        else:
            
            # Load agent data
            R, X1, X2, X3 = load_z_data_temp(z_data_dir, j)


        # ------ Simulate Communication ------
        # Combine data to X
        X = np.vstack((X1, X2, X3)).T  
        print(R)

        # 
        N_sample = 30
        N, D = X.shape  # N = No samples, D = No Agents

        # Initialize Z
        Z = np.random.uniform(0, 1, (N_sample, D))

        # X ---> R mapping 
        
        # opt1.plot(100)
        # opt2.plot(100)
        # opt2.plot(100)
        
    
        # plt.show()
        
        
      
        x_test = np.linspace(0.01,10,N_sample).reshape(-1,1)
        mu_sample_log1 = []
        mu_sample_log2 = []
        mu_sample_log3 = []
        
        for i in range(N_sample):
            
            mu_sample1, _ = gp1.predict_noiseless(x_test[i].reshape(-1,1))
            mu_sample2, _ = gp2.predict_noiseless(x_test[i].reshape(-1,1))
            mu_sample3, _ = gp3.predict_noiseless(x_test[i].reshape(-1,1))
            
            mu_sample_log1.append(mu_sample1)
            mu_sample_log2.append(mu_sample2)
            mu_sample_log3.append(mu_sample3)
            
        
        # plt.figure(1)
        # plt.scatter(x_test,mu_sample_log1)
        # # plt.show()
        # plt.figure(2)
        # plt.scatter(x_test,mu_sample_log2)
        # # plt.show()
        # plt.figure(3)
        # plt.scatter(x_test,mu_sample_log3)
        # # plt.show()
        
        X1 = np.array(x_test)
        X2 = np.array(x_test)
        X3 = np.array(x_test)
        X = np.vstack((X1, X2, X3)).reshape(N_sample, D)
        
        est_R1 = np.array(mu_sample_log1).flatten()
        est_R2 = np.array(mu_sample_log2).flatten()
        est_R3 = np.array(mu_sample_log3).flatten() 
        
        est_R = np.vstack((est_R1, est_R2, est_R3)).T   
        
        
        
        

        # ----------- Minimize the loss function ------------

        wait = input("Press Enter to minimize...")
        
        # X ----> R
        model_X1 = GPy.models.GPRegression(X[:,0].reshape(-1,1), est_R[:,0].reshape(-1,1), GPy.kern.Matern32(1))
        model_X2 = GPy.models.GPRegression(X[:,1].reshape(-1,1), est_R[:,1].reshape(-1,1), GPy.kern.Matern32(1))
        model_X3 = GPy.models.GPRegression(X[:,2].reshape(-1,1), est_R[:,2].reshape(-1,1), GPy.kern.Matern32(1))
        
        # plt.figure()
        # model_X1.plot() 
        # plt.show()        
        
        
        result = minimize(column_wise, Z.flatten(), args=(X, est_R, D, N_sample), method='L-BFGS-B',options={'ftol':1e-1,'gtol':1e-1,'maxiter':1000})
        Z_opt = result.x.reshape(N_sample, D)
                
        # Z ---> X mapping
        Z_to_X_0 = GPy.models.GPRegression(Z_opt[:, 0].reshape(-1,1), X[:,0].reshape(-1,1), kernel=GPy.kern.Matern32(1), noise_var=0.01)
        Z_to_X_1 = GPy.models.GPRegression(Z_opt[:, 1].reshape(-1,1), X[:,1].reshape(-1,1), kernel=GPy.kern.Matern32(1), noise_var=0.01)
        Z_to_X_2 = GPy.models.GPRegression(Z_opt[:, 2].reshape(-1,1), X[:,2].reshape(-1,1), kernel=GPy.kern.Matern32(1), noise_var=0.01)

        Z_to_X_0.plot()
        plt.title('Z1 --> X1')
        plt.xlabel('Z1')
        plt.ylabel('X1')
        Z_to_X_1.plot()
        plt.title('Z2 --> X2')
        plt.xlabel('Z2')
        plt.ylabel('X2')
        Z_to_X_2.plot()
        plt.title('Z3 --> X3')
        plt.xlabel('Z3')
        plt.ylabel('X3')
        plt.savefig(os.path.join(plots_dir, f'Z_to_X_mapping_{j+1}.png'))
        plt.show()
        
        m_Z1 = GPy.models.GPRegression(Z_opt[:,0].reshape(-1,1), est_R[:,0].reshape(-1,1), kernel=GPy.kern.Matern32(1), noise_var=0.01)
        m_Z2 = GPy.models.GPRegression(Z_opt[:,1].reshape(-1,1), est_R[:,1].reshape(-1,1), kernel=GPy.kern.Matern32(1), noise_var=0.01)
        m_Z3 = GPy.models.GPRegression(Z_opt[:,2].reshape(-1,1), est_R[:,2].reshape(-1,1), kernel=GPy.kern.Matern32(1), noise_var=0.01)
        
        m_Z1.plot()
        m_Z2.plot() 
        m_Z3.plot()
        plt.show()
        
        # ------ Communication Complete ------

        Z1 = Z_opt[:,0]
        Z2 = Z_opt[:,1]
        Z3 = Z_opt[:,2]
        
        
        
        kernel1 = GPy.kern.Matern32(1)
        kernel2 = GPy.kern.Matern32(1)
        kernel3 = GPy.kern.Matern32(1)

        # Z1 ----> R mapping
        gp1 = GPy.models.GPRegression(Z1.reshape(-1,1), est_R[:,0].reshape(-1,1), kernel1, noise_var=0.05**2)
        # Z2 ----> R mapping
        gp2 = GPy.models.GPRegression(Z2.reshape(-1,1), est_R[:,1].reshape(-1,1), kernel2, noise_var=0.05**2)
        # Z3 ----> R mapping
        gp3 = GPy.models.GPRegression(Z3.reshape(-1,1), est_R[:,2].reshape(-1,1), kernel3, noise_var=0.05**2)


        latent_parameter_set = safeopt.linearly_spaced_combinations(K_bounds_Z, discretization)

        # Agent safeopt objects
        opt1 = safeopt.SafeOpt(gp1, latent_parameter_set, safety_threshold, beta, threshold=0.05)
        opt2 = safeopt.SafeOpt(gp2, latent_parameter_set, safety_threshold, beta, threshold=0.05)
        opt3 = safeopt.SafeOpt(gp3, latent_parameter_set, safety_threshold, beta, threshold=0.05)

        print("Agents initialized...")

        actions_1_log = []
        actions_2_log = []
        actions_3_log = []
        reward_z_log = []

        wait = input("Press enter to start BO")

        # Bayesian Optimization in the latent space
        for iteration in range(N):
            print(iteration)
            
            # Get next Z values from agents
            Z1_next = opt1.optimize()
            Z2_next = opt1.optimize()
            Z3_next = opt1.optimize()
            
            
            # Z --> X mapping
            Kp1_next, _ = Z_to_X_0.predict_noiseless(Z1_next[0].reshape(-1,1))
            Kp2_next, _ = Z_to_X_1.predict_noiseless(Z2_next[0].reshape(-1,1))
            Kp3_next, _ = Z_to_X_2.predict_noiseless(Z3_next[0].reshape(-1,1))
            
            print(f"Kp1_next: {Kp1_next}")
            print(f"Kp2_next: {Kp2_next}")
            print(f"Kp3_next: {Kp3_next}")

            if Kp1_next < 0:
                print("Danger!!!!")
                exit(0)
            if Kp2_next < 0:
                print("Danger!!!!")
                exit(0)
            if Kp3_next < 0:
                print("Danger!!!!")
                exit(0)
            
            Kp1_next = np.asarray([Kp1_next]).flatten()
            Kp2_next = np.asarray([Kp2_next]).flatten()
            Kp3_next = np.asarray([Kp3_next]).flatten()
            
            actions_1_log = np.append(actions_1_log, Kp1_next)
            actions_2_log = np.append(actions_2_log, Kp2_next)
            actions_3_log = np.append(actions_3_log, Kp3_next)
            
            # Run the experiment with the mapped Kp and Kd values
            y, os1, os2, os3 = run_experiment(Kp1_next[0], Kd1, Kp2_next[0], Kd2, Kp3_next[0], Kd3, iteration)

            print(f"Reward: {y}")
            
            reward_z_log.append(y)

            # Update agents with observations
            opt1.add_new_data_point(Z1_next,y)
            opt1.add_new_data_point(Z2_next,y)
            opt1.add_new_data_point(Z3_next,y)
            
            
            
        
        write_z_data(z_data_dir,actions_1_log,actions_2_log,actions_3_log,reward_z_log, j+1)
        
        plt.figure()
        plt.plot(reward_z_log, label='Reward_Z')
        plt.plot(rewards, label='Reward_X')
        plt.savefig(os.path.join(plots_dir, f'reward_comparison_{j+1}.png'))
        plt.show()