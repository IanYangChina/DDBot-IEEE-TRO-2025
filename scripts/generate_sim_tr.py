import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'data', 'moveit_trajectories')

n = 0

print(f"==================== Task {n} ====================")
timestamps = np.load(os.path.join(data_path, f'sys_id_{n}_timestamps.npy'))
v = np.load(os.path.join(data_path, f'sys_id_{n}_v.npy'))

# for i in range(len(timestamps)):
#     print(f'index {i}, time: {timestamps[i]}, vx: {v[i, 0]}, vy: {v[i, 1]}, vz: {v[i, 2]}, vrz: {v[i, 5]}')

dt_sim = 0.001
trajectory_1 = np.zeros(shape=(5000, 6))
tr1_dt_1 = timestamps[20]
tr1_n_steps_1 = int(tr1_dt_1 / dt_sim)
tr1_delta_x_1 = 0.09 / tr1_n_steps_1
trajectory_1[:tr1_n_steps_1, 0] = tr1_delta_x_1
tr1_dt_2 = timestamps[31] - timestamps[20]
tr1_n_steps_2 = int(tr1_dt_2 / dt_sim)
tr1_delta_z_2 = -0.05 / tr1_n_steps_2
trajectory_1[tr1_n_steps_1:tr1_n_steps_1 + tr1_n_steps_2, 2] = tr1_delta_z_2
tr1_dt_3 = timestamps[55] - timestamps[31]
tr1_n_steps_3 = int(tr1_dt_3 / dt_sim)
tr1_delta_x_3 = -0.12 / tr1_n_steps_3
trajectory_1[tr1_n_steps_1 + tr1_n_steps_2:tr1_n_steps_1 + tr1_n_steps_2 + tr1_n_steps_3, 0] = tr1_delta_x_3
tr1_dt_4 = timestamps[78] - timestamps[55]
tr1_n_steps_4 = int(tr1_dt_4 / dt_sim)
tr1_delta_z_4 = 0.12 / tr1_n_steps_4
trajectory_1[tr1_n_steps_1 + tr1_n_steps_2 + tr1_n_steps_3:tr1_n_steps_1 + tr1_n_steps_2 + tr1_n_steps_3 + tr1_n_steps_4, 2] = tr1_delta_z_4

trajectory_1 = trajectory_1[:tr1_n_steps_1 + tr1_n_steps_2 + tr1_n_steps_3 + tr1_n_steps_4, :]
print(dt_sim * (tr1_n_steps_1 + tr1_n_steps_2 + tr1_n_steps_3 + tr1_n_steps_4))
np.save(os.path.join(data_path, f'sys_id_sim_0_pos-dt_{dt_sim}.npy'), trajectory_1)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].plot(timestamps)
ax[0].set_title(f'Last timestamp: {timestamps[-1]}')
ax[1].plot(v)
ax[1].legend(['x', 'y', 'z', 'a', 'b', 'c'])
ax[2].plot(trajectory_1)
ax[2].legend(['x', 'y', 'z', 'a', 'b', 'c'])
plt.tight_layout()
plt.show()
plt.close()

n = 1

print(f"==================== Task {n} ====================")
timestamps = np.load(os.path.join(data_path, f'sys_id_{n}_timestamps.npy'))
v = np.load(os.path.join(data_path, f'sys_id_{n}_v.npy'))

# for i in range(len(timestamps)):
#     print(f'index {i}, time: {timestamps[i]}, vx: {v[i, 0]}, vy: {v[i, 1]}, vz: {v[i, 2]}, vrz: {v[i, 5]}')

trajectory_2 = np.zeros(shape=(5000, 6))
tr2_dt_1 = timestamps[47]
tr2_n_steps_1 = int(tr2_dt_1 / dt_sim)
tr2_delta_x_1 = 0.09 / tr2_n_steps_1
tr2_delta_y_1 = 0.09 / tr2_n_steps_1
tr2_delta_rz_1 = (- np.pi / 4) / tr2_n_steps_1
trajectory_2[:tr2_n_steps_1, 0] = tr2_delta_x_1
trajectory_2[:tr2_n_steps_1, 1] = tr2_delta_y_1
trajectory_2[:tr2_n_steps_1, 5] = tr2_delta_rz_1
tr2_dt_2 = timestamps[57] - timestamps[47]
tr2_n_steps_2 = int(tr2_dt_2 / dt_sim)
tr2_delta_z_2 = -0.05 / tr2_n_steps_2
trajectory_2[tr2_n_steps_1:tr2_n_steps_1 + tr2_n_steps_2, 2] = tr2_delta_z_2
tr2_dt_3 = timestamps[91] - timestamps[57]
tr2_n_steps_3 = int(tr2_dt_3 / dt_sim)
tr2_delta_x_3 = -0.12 / tr2_n_steps_3
tr2_delta_y_3 = -0.12 / tr2_n_steps_3
trajectory_2[tr2_n_steps_1 + tr2_n_steps_2:tr2_n_steps_1 + tr2_n_steps_2 + tr2_n_steps_3, 0] = tr2_delta_x_3
trajectory_2[tr2_n_steps_1 + tr2_n_steps_2:tr2_n_steps_1 + tr2_n_steps_2 + tr2_n_steps_3, 1] = tr2_delta_y_3
tr2_dt_4 = timestamps[115] - timestamps[91]
tr2_n_steps_4 = int(tr2_dt_4 / dt_sim)
tr2_delta_z_4 = 0.12 / tr2_n_steps_4
trajectory_2[tr2_n_steps_1 + tr2_n_steps_2 + tr2_n_steps_3:tr2_n_steps_1 + tr2_n_steps_2 + tr2_n_steps_3 + tr2_n_steps_4, 2] = tr2_delta_z_4

trajectory_2 = trajectory_2[:tr2_n_steps_1 + tr2_n_steps_2 + tr2_n_steps_3 + tr2_n_steps_4, :]
print(dt_sim * (tr2_n_steps_1 + tr2_n_steps_2 + tr2_n_steps_3 + tr2_n_steps_4))
np.save(os.path.join(data_path, f'sys_id_sim_1_pos-dt_{dt_sim}.npy'), trajectory_2)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].plot(timestamps)
ax[0].set_title(f'Last timestamp: {timestamps[-1]}')
ax[1].plot(v)
ax[1].legend(['x', 'y', 'z', 'a', 'b', 'c'])
ax[2].plot(trajectory_2)
ax[2].legend(['x', 'y', 'z', 'a', 'b', 'c'])
plt.tight_layout()
plt.show()
plt.close()

