import os
import numpy as np
from doma.optimiser.adam import Adam
from doma.engine.configs.macros import DTYPE_NP

script_path = os.path.dirname(os.path.realpath(__file__))
subfux = '-hm'
# for d in ["5e6"]:
#     for task in [0]:
#         grad_mean = np.load(os.path.join(script_path, '..', 'log-abs2-adam', f'd{d}-task-{task}{subfux}', 'grads', "grad_mean.npy"))
#         grad_std = np.load(os.path.join(script_path, '..', 'log-abs2-adam', f'd{d}-task-{task}{subfux}', 'grads', "grad_std.npy"))
#         print(f"Found gradient stats for d={d}, task={task}:")
#         print(grad_mean)
#         print(grad_std)
# exit()

for d in ["5e6"]:
    for task in [0]:
        print("d:", d, "task:", task)
        grad_mean = np.load(os.path.join(script_path, '..', 'log-abs2-adam', f'd{d}-task-{task}{subfux}', 'grads', "grad_mean.npy"))
        grad_std = np.load(os.path.join(script_path, '..', 'log-abs2-adam', f'd{d}-task-{task}{subfux}', 'grads', "grad_std.npy"))
        # E_range = (2.5e5, 4e5)
        # nu_range = (0.2, 0.4)
        # rho_range = (1600, 2300)
        # sand_angle_range = (30, 45)
        # mf_range = (0.05, 2.0)
        # sf_range = (0.05, 2.0)
        # n_epoch = 100
        # ranges = [E_range, nu_range, rho_range, sand_angle_range, mf_range, sf_range]
        """
        Found learning rates:
        E: 2e5
        nu: 0.1
        rho: 1e3
        sand_angle: 10
        mf: 1
        sf: 0.5
        """
        lr = [0.02, 0.03, 0.03, 0.01, 0.15]
        n = 5
        for n in range(5):
            bounds = (-1.0, 1.0)
            mean = grad_mean[n]
            std = grad_std[n]
            param = np.asarray(np.random.uniform(bounds[0], bounds[1]), dtype=DTYPE_NP).reshape((1,))
            print("n:", n, param, mean, std)
            optimiser = Adam(parameters_shape=param.shape,
                             cfg={'lr': lr[n], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

            delta = 0.0
            for _ in range(300):
                grad = np.random.uniform(mean-std, mean+std, size=param.shape)
                parm_ = optimiser.step(param.copy(), grad.copy())
                delta += (parm_ - param)
                parm = np.clip(parm_, bounds[0], bounds[1])
                # print(optimiser.cur_lr, grad, parm)
            print("delta:", delta/150)
        print("\n")
