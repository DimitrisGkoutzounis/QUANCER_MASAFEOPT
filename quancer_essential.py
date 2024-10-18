"""
Communicates with 2 agents to retrieve data and compute the reward
"""


import subprocess
import time
import shutil
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np


modelName = 'servoPDF'


target_uri_1 = 'tcpip://172.22.11.2:17000?keep_alive=1'
target_uri_2 = 'tcpip://172.22.11.10:17000?keep_alive=1'

std_args = ' -d ./tmp -uri tcpip://linux-dev:17001'


#assuming the model is built from MatLab


sys1dl = f'quarc_run -D -t {target_uri_1} {modelName}.rt-linux_rt_armv7{std_args}'
sys2dl = f'quarc_run -D -t {target_uri_1} {modelName}.rt-linux_rt_armv7{std_args}'
print(sys1dl)

# Run the system commands

subprocess.call(sys1dl, shell=True)
subprocess.call(sys2dl, shell=True)

kp1= 4 
kd1 = 0.05

kp2= 4
kd2 = 0.05

td1 = 0.045
gain_arg1 = f' -Kp {kp1:.4f} -Kd {kd1:.4f}'
gain_arg2 = f' -Kp {kp2:.4f} -Kd {kd2:.4f}'
print(gain_arg1)
sys1run = f'quarc_run -l -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1} -td {td1:.5f} {std_args}'
print(sys1run)
sys2run = f'quarc_run -l -t {target_uri_2} {modelName}.rt-linux_rt_armv7{gain_arg2}{std_args}'


subprocess.call(sys1run, shell=True)
subprocess.call(sys2run, shell=True)

time.sleep(6)

# retrieve data from board

sys1get = f'quarc_run -u -t {target_uri_1} {modelName}.rt-linux_rt_armv7{gain_arg1}{std_args}'
sys2get = f'quarc_run -u -t {target_uri_2} {modelName}.rt-linux_rt_armv7{gain_arg2}{std_args}'

subprocess.call(sys1get, shell=True)

#copy data to new file
shutil.copyfile('servoPDF.mat', 'servoPDF-1.mat')

subprocess.call(sys2get, shell=True)
shutil.copyfile('servoPDF.mat', 'servoPDF-2.mat')

data1 = loadmat('servoPDF-1.mat')
rt_t1 = data1['rt_t'].flatten()
rt_theta1 = data1['rt_theta'].flatten()

data2 = loadmat('servoPDF-2.mat')
# data2 = data2[:4000]
rt_t2 = data2['rt_t'].flatten()
rt_theta2 = data2['rt_theta'].flatten()

theta_d = data1['rt_theta_d'].flatten()

#overshoot error

os1 = theta_d - rt_theta1
os2 = theta_d - rt_theta2

error12 = rt_theta1 - rt_theta2

#compute error

integral_error1 = np.trapz(os1, rt_t1)
integral_error2 = np.trapz(os2, rt_t2)
integral_error12 = np.trapz(error12, rt_t1)






################### PLOTTING ####################

plt.figure(1)
plt.plot(rt_t1, rt_theta1, label='Agent-1')
plt.plot(rt_t2, rt_theta2, label='Agent-2')
plt.plot(rt_t1, error12, 'b--', label='Error 12')

plt.grid(True)
plt.xlabel('t (s)')
plt.ylabel('theta (rad)')
plt.legend()

plt.figure(2)

plt.plot(rt_t1[:4000], os1[:4000], label='Agent-1 OS')
plt.plot(rt_t2[:4000], os2[:4000], label='Agent-2 OS')

plt.grid(True)
plt.xlabel('t (s)')
plt.ylabel('theta (rad)')
plt.legend()
plt.show()


