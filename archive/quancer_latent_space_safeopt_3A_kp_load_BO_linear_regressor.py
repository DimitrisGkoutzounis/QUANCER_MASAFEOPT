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
from sklearn.linear_model import LinearRegression



modelName = 'servoPDF'

# Target URIs for the agents
target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'
target_uri_2 = 'tcpip://172.22.11.10:17000?keep_alive=1'
target_uri_3 = 'tcpip://172.22.11.18:17000?keep_alive=1'  

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'


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

def column_wise(Z_flat, X, D, N):
    Z = Z_flat.reshape(N, D)

    # Define model_Z with R_z as observations
    model_Z = GPy.models.GPRegression(Z, R.reshape(-1,1), GPy.kern.RBF(D))
    model_all = GPy.models.GPRegression(Z, X,  GPy.kern.RBF(D))
    mu_all, _ = model_all.predict_noiseless(Z)

    loss = 0.0
    action_term = 0.0
    
    
    grad_R_Z_norm_column = []
    grad_R_X_norm_column = []

    # Initialize matrices for U_z and U_x
    U_z = np.zeros((N, D))
    U_x = np.zeros((N, D))
    

    for d in range(D):
        X_d = np.zeros_like(X)
        X_d[:, d] = X[:, d]
        
        model_d = GPy.models.GPRegression(Z, X_d,GPy.kern.RBF(D))
        mu_d, _ = model_d.predict_noiseless(Z)
        

        diff1 = np.linalg.norm(X_d - mu_d)**2
        diff2 = np.linalg.norm(mu_d - mu_all[:, [d]])**2
        
        action_term += 0 * diff1 + 0 * diff2

        # Gradient-based alignment term
        grad_R_Z = compute_gradient(model_Z, Z).reshape(N, D)
        grad_R_X = compute_gradient(model_X, X).reshape(N, D)

        grad_R_Z_norm_column.append(np.linalg.norm(grad_R_Z[:, d]))
        grad_R_X_norm_column.append(np.linalg.norm(grad_R_X[:, d]))

        U_z[:, d] = grad_R_Z[:, d] / grad_R_Z_norm_column[d]
        U_x[:, d] = grad_R_X[:, d] / grad_R_X_norm_column[d]

    dot_product_matrix = np.dot(U_z.T, U_x)
    diag_penalty = np.linalg.norm((1 - np.diag(dot_product_matrix))**2)/D
    
    total_loss = action_term + 1 * diag_penalty
    print(total_loss) 


    return total_loss


z_data_dir = 'Z_data'
agent_data_dir = 'agent_data_3A_baseline'

if not os.path.exists(z_data_dir):
    os.makedirs(z_data_dir)

if __name__ == '__main__':

    K = 4 # Number of experiments
    N = 25  # Number of BO trials
    
    # Delay difference between the agents
    td1 = 0.09
    td2 = 0.045  
    td3 = 0.001 
    
    Kd1 = 0.3
    Kd2 = 0.2
    Kd3 = 0.1
    
    rewards = []

    for j in range(K):
            
        if j == 0:

            agent1_x1 = []
            agent2_x2 = []
            agent3_x3 = []


            with open(f'{agent_data_dir}/agent1_data.txt', 'r') as f1:
                reader = csv.reader(f1)
                next(reader)  # Skip header
                for row in reader:
                    agent1_x1.append(float(row[1]))
                    rewards.append(float(row[2]))
                    
            with open(f'{agent_data_dir}/agent2_data.txt', 'r') as f2:
                reader = csv.reader(f2)
                next(reader)  # Skip header
                for row in reader:
                    agent2_x2.append(float(row[1]))
                    
            with open(f'{agent_data_dir}/agent3_data.txt', 'r') as f3:
                reader = csv.reader(f3)
                next(reader)  # Skip header
                for row in reader:
                    agent3_x3.append(float(row[1]))

            X1 = np.array(agent1_x1)
            X2 = np.array(agent2_x2)
            X3 = np.array(agent3_x3)
            R = np.array(rewards).reshape(-1,1)
            
            print("X1",X1)
            print("X2",X2)
            print("X3",X3)
            
                
            
        else:
            
            # Load agent data
            R, X1, X2, X3 = load_z_data_temp(z_data_dir, j)


        # Combine data to X
        X = np.vstack((X1, X2, X3)).T  
        print(R)

        # 
        N, D = X.shape  # N = No samples, D = No Agents

        # Initialize Z
        Z = np.random.uniform(0, 1, (N, D))

        # X ---> R mapping 
        model_X = GPy.models.GPRegression(X, R, GPy.kern.RBF(input_dim=D))

        # ----------- Minimize the loss function ------------

        wait = input("Press Enter to minimize...")
        result = minimize(column_wise, Z.flatten(), args=(X, D, N), method='L-BFGS-B',options={'ftol':1e-2,'gtol':1e-2,'maxiter':100})
        Z_opt = result.x.reshape(N, D)
                
        # Z ---> X mapping
        Z_to_X_0 = GPy.models.GPRegression(Z_opt[:, 0].reshape(-1,1), X[:,0].reshape(-1,1), kernel=GPy.kern.RBF(1))
        Z_to_X_1 = GPy.models.GPRegression(Z_opt[:, 1].reshape(-1,1), X[:,1].reshape(-1,1), kernel=GPy.kern.RBF(1))
        Z_to_X_2 = GPy.models.GPRegression(Z_opt[:, 2].reshape(-1,1), X[:,2].reshape(-1,1), kernel=GPy.kern.RBF(1))
        
        
        reg1 = LinearRegression().fit(Z_opt[:,0].reshape(-1,1), X[:,0].reshape(-1,1))
        print(reg1.score(Z_opt[:,0].reshape(-1,1), X[:,0].reshape(-1,1)))   
        
        y_pred1 = reg1.predict(Z_opt[:,0].reshape(-1,1))    
        
        plt.figure()
        plt.scatter(Z_opt[:,0], X[:,0])
        plt.plot(Z_opt[:,0], y_pred1, color='red')
        plt.xlabel('Z1')
        plt.ylabel('X1')
        # plt.show()
        
        reg2 = LinearRegression().fit(Z_opt[:,1].reshape(-1,1), X[:,1].reshape(-1,1))
        print(reg2.score(Z_opt[:,1].reshape(-1,1), X[:,1].reshape(-1,1)))   
        
        y_pred2 = reg2.predict(Z_opt[:,1].reshape(-1,1))  
        
        plt.figure()
        plt.scatter(Z_opt[:,1], X[:,1])
        plt.plot(Z_opt[:,1], y_pred2, color='red')
        plt.xlabel('Z2')
        plt.ylabel('X2')
        # plt.show()
        
        
        reg3 = LinearRegression().fit(Z_opt[:,2].reshape(-1,1), X[:,2].reshape(-1,1))
        print(reg3.score(Z_opt[:,2].reshape(-1,1), X[:,2].reshape(-1,1)))   
        
        y_pred3 = reg3.predict(Z_opt[:,2].reshape(-1,1))  
        
        
        plt.figure()
        plt.scatter(Z_opt[:,2], X[:,2])
        plt.plot(Z_opt[:,2], y_pred3, color='red')
        plt.xlabel('Z3')
        plt.ylabel('X3')
        

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
        plt.show()

        Z1 = Z_opt[:,0]
        Z2 = Z_opt[:,1]
        Z3 = Z_opt[:,2]
        
        

        K_bounds_Z = [(-10,10)]
        
        print(K_bounds_Z)
        
        wait = input("Press Enter to continue...")

        kernel1 = GPy.kern.RBF(1)
        kernel2 = GPy.kern.RBF(1)
        kernel3 = GPy.kern.RBF(1)

        # Z1 ----> R mapping
        gp1 = GPy.models.GPRegression(Z1.reshape(-1,1), R, kernel1, noise_var=0.05**2)
        # Z2 ----> R mapping
        gp2 = GPy.models.GPRegression(Z2.reshape(-1,1), R, kernel2, noise_var=0.05**2)
        # Z3 ----> R mapping
        gp3 = GPy.models.GPRegression(Z3.reshape(-1,1), R, kernel3, noise_var=0.05**2)
        latent_parameter_set = safeopt.linearly_spaced_combinations(K_bounds_Z, 1000)

        # Agent safeopt objects
        opt1 = safeopt.SafeOpt(gp1, latent_parameter_set, 0.03, beta=1.0, threshold=0.05)
        opt2 = safeopt.SafeOpt(gp2, latent_parameter_set, 0.03, beta=1.0, threshold=0.05)
        opt3 = safeopt.SafeOpt(gp3, latent_parameter_set, 0.03, beta=1.0, threshold=0.05)

        print("Agents initialized...")

        actions_1 = []
        actions_2 = []
        actions_3 = []

        wait = input("Press enter to start BO")
        reward_z = []

        # Bayesian Optimization in the latent space
        for iteration in range(N):
            print(iteration)
            
            # Get next Z values from agents
            Z1_next = opt1.optimize()
            Z2_next = opt1.optimize()
            Z3_next = opt1.optimize()
            
            Kp1_next = reg1.predict(Z1_next.reshape(-1,1))
            Kp2_next = reg2.predict(Z2_next.reshape(-1,1))
            Kp3_next = reg3.predict(Z3_next.reshape(-1,1))
            
            print("Predicted Kp values")    
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
            
            actions_1 = np.append(actions_1, Kp1_next)
            actions_2 = np.append(actions_2, Kp2_next)
            actions_3 = np.append(actions_3, Kp3_next)
            
            # Run the experiment with the mapped Kp and Kd values
            y, os1, os2, os3 = run_experiment(Kp1_next[0], Kd1, Kp2_next[0], Kd2, Kp3_next[0], Kd3, iteration)

            print(f"Reward: {y}")
            
            reward_z.append(y)

            # Update agents with observations
            opt1.add_new_data_point(Z1_next,y)
            opt1.add_new_data_point(Z2_next,y)
            opt1.add_new_data_point(Z3_next,y)
            
            opt1.plot(100)
            opt2.plot(100)
            opt3.plot(100)
            plt.savefig(f'{z_data_dir}/safeopt_{iteration}.png')


        write_z_data(z_data_dir,actions_1,actions_2,actions_3,reward_z, j+1)
        
        plt.figure()
        plt.plot(reward_z, label='Reward_Z')
        plt.plot(rewards, label='Reward_X')
        plt.savefig(f'{z_data_dir}/reward_{j}.png')
        plt.legend()
        plt.show()

        x_max_1, y_max_1 = opt1.get_maximum()
        x_max_2, y_max_2 = opt2.get_maximum()
        x_max_3, y_max_3 = opt3.get_maximum()
        
        print(f"Maximum for agent 1: {x_max_1}, {y_max_1}")
        print(f"Maximum for agent 2: {x_max_2}, {y_max_2}")
        print(f"Maximum for agent 3: {x_max_3}, {y_max_3}")
        