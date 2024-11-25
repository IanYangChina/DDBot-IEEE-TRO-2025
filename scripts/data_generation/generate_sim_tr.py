import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
data_path = os.path.join(script_path, '..', 'data', 'moveit_trajectories')

n = 0
for n in range(2):
    print(f"==================== Task {n} ====================")
    timestamps = np.load(os.path.join(data_path, f'sys_id_{n}_timestamps.npy'))
    v = np.load(os.path.join(data_path, f'sys_id_{n}_v.npy'))
    if n == 1:
        v[:, -1] *= -1

    dt_sim = 0.01
    trajectory_1 = np.zeros(shape=(5000, 6))
    sim_v_1 = np.zeros(shape=(5000, 6))
    cur_step = 0
    for m in range(1, len(timestamps)):
        dt = timestamps[m] - timestamps[m - 1]
        n_steps = int(dt / dt_sim)
        sim_v_1[cur_step:cur_step + n_steps, :] = v[m-1]
        delta_tr = v[m-1] * dt
        delta_tr /= n_steps
        for i in range(n_steps):
            trajectory_1[cur_step] = delta_tr
            cur_step += 1

    trajectory_1 = trajectory_1[:cur_step, :]
    sim_v_1 = sim_v_1[:cur_step, :]
    print(dt_sim * cur_step)
    np.save(os.path.join(data_path, f'sys_id_sim_{n}_pos-dt_{dt_sim}'), trajectory_1)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    ax[0].plot(timestamps)
    ax[0].set_title(f'Last timestamp: {timestamps[-1]}')
    ax[1].plot(v)
    ax[1].legend(['x', 'y', 'z', 'a', 'b', 'c'])
    ax[2].plot(trajectory_1)
    ax[2].legend(['x', 'y', 'z', 'a', 'b', 'c'])
    ax[3].plot(sim_v_1)
    ax[3].legend(['x', 'y', 'z', 'a', 'b', 'c'])
    plt.tight_layout()
    plt.show()
    plt.close()
