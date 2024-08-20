import os
import numpy as np
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'data', 'moveit_trajectories')

for n in [1]:
    print(f"==================== Task {n} ====================")
    timestamps = np.load(os.path.join(data_path, f'sys_id_{n}_timestamps.npy'))
    v = np.load(os.path.join(data_path, f'sys_id_{n}_v.npy'))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(timestamps)
    ax[0].set_title(f'Last timestamp: {timestamps[-1]}')
    ax[1].plot(v)
    ax[1].legend(['x', 'y', 'z', 'a', 'b', 'c'])
    plt.tight_layout()
    plt.show()
    plt.close()

    for i in range(6):
        max_v_i = np.max(v[:, i])
        min_v_i = np.min(v[:, i])
        print(f"======== Dim {i} =========")
        for j in range(len(timestamps)):
            if np.abs(v[j, i]) <= 1e-4:
                print(f'Zero v: {v[j, i]}, time: {timestamps[j]}, index {j}')
            if v[j, i] == min_v_i:
                print(f'Min  v: {v[j, i]}, time: {timestamps[j]}, index {j}')
            if v[j, i] == max_v_i:
                print(f'Max  v: {v[j, i]}, time: {timestamps[j]}, index {j}')
