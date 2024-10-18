import subprocess
import time
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np


modelName = 'servoPDF'

# Define target URIs for each system
target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'
target_uri_2 = 'tcpip://172.22.11.10:17000?keep_alive=1'
target_uri_3 = 'tcpip://172.22.11.18:17000?keep_alive=1'

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'

# Assuming the model is built from MatLab

sys1dl = f'quarc_run -D -t {target_uri_1} {modelName}.rt-linux_rt_armv7{std_args}'
sys2dl = f'quarc_run -D -t {target_uri_2} {modelName}.rt-linux_rt_armv7{std_args}'
sys3dl = f'quarc_run -D -t {target_uri_3} {modelName}.rt-linux_rt_armv7{std_args}'

# Run the system commands

subprocess.call(sys1dl, shell=True)
subprocess.call(sys2dl, shell=True)
subprocess.call(sys3dl, shell=True)

# Define control parameters for each system
kp1, kd1 = 4, 0.05
kp2, kd2 = 4, 0.05
kp3, kd3 = 4, 0.05

td1, td3 = 0.045, 0.045

# Command strings with PD gains for each system
gain_arg1 = f' -Kp {kp1:.4f} -Kd {kd1:.4f}'
gain_arg2 = f' -Kp {kp2:.4f} -Kd {kd2:.4f}'
gain_arg3 = f' -Kp {kp3:.4f} -Kd {kd3:.4f}'

sys1run = f'quarc_run -l -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1} -td {td1:.5f} {std_args}'
sys2run = f'quarc_run -l -t {target_uri_2} {modelName}.rt-linux_rt_armv7{gain_arg2}{std_args}'
sys3run = f'quarc_run -l -t {target_uri_3} {modelName}.rt-linux_rt_armv7{gain_arg3} -td {td3:.5f} {std_args}'

# Run the systems
subprocess.call(sys1run, shell=True)
subprocess.call(sys2run, shell=True)
subprocess.call(sys3run, shell=True)

time.sleep(6)

# Retrieve data from boards
sys1get = f'quarc_run -u -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1}{std_args}'
sys2get = f'quarc_run -u -t {target_uri_2} {modelName}.rt-linux_rt_armv7{gain_arg2}{std_args}'
sys3get = f'quarc_run -u -t {target_uri_3} {modelName}.rt-linux_rt_armv7{gain_arg3}{std_args}'

subprocess.call(sys1get, shell=True)
shutil.copyfile('servoPDF.mat', 'servoPDF-1.mat')

subprocess.call(sys2get, shell=True)
shutil.copyfile('servoPDF.mat', 'servoPDF-2.mat')

subprocess.call(sys3get, shell=True)
shutil.copyfile('servoPDF.mat', 'servoPDF-3.mat')

# Load the data
data1 = loadmat('servoPDF-1.mat')
rt_t1 = data1['rt_t'].flatten()
rt_theta1 = data1['rt_theta'].flatten()

data2 = loadmat('servoPDF-2.mat')
rt_t2 = data2['rt_t'].flatten()
rt_theta2 = data2['rt_theta'].flatten()

data3 = loadmat('servoPDF-3.mat')
rt_t3 = data3['rt_t'].flatten()
rt_theta3 = data3['rt_theta'].flatten()

theta_d = data1['rt_theta_d'].flatten()

# Overshoot error
os1 = theta_d - rt_theta1
os2 = theta_d - rt_theta2
os3 = theta_d - rt_theta3

error12 = rt_theta1 - rt_theta2
error13 = rt_theta1 - rt_theta3

# Compute error
integral_error1 = np.trapz(os1, rt_t1)
integral_error2 = np.trapz(os2, rt_t2)
integral_error3 = np.trapz(os3, rt_t3)
integral_error12 = np.trapz(error12, rt_t1)
integral_error13 = np.trapz(error13, rt_t1)

################### PLOTTING ####################

# Plot agent outputs and errors
plt.figure(1)
plt.plot(rt_t1, rt_theta1, label='Agent-1')
plt.plot(rt_t2, rt_theta2, label='Agent-2')
plt.plot(rt_t3, rt_theta3, label='Agent-3')
plt.plot(rt_t1, error12, 'b--', label='Error 1-2')
plt.plot(rt_t1, error13, 'g--', label='Error 1-3')

plt.grid(True)
plt.xlabel('t (s)')
plt.ylabel('theta (rad)')
plt.legend()

# Plot overshoot error for each agent
plt.figure(2)
plt.plot(rt_t1[:4000], os1[:4000], label='Agent-1 OS')
plt.plot(rt_t2[:4000], os2[:4000], label='Agent-2 OS')
plt.plot(rt_t3[:4000], os3[:4000], label='Agent-3 OS')

plt.grid(True)
plt.xlabel('t (s)')
plt.ylabel('Overshoot error (rad)')
plt.legend()

plt.show()
