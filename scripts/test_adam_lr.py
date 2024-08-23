import os
import numpy as np
from doma.optimiser.adam import Adam
from doma.engine.configs.macros import DTYPE_NP

script_path = os.path.dirname(os.path.realpath(__file__))

# for d in ["5e6", "1e7", "2e7"]:
#     for ns in [20]:
#         grad_mean = np.load(os.path.join(script_path, '..', 'log-sys_id', f"grads-mean-d{d}-ns{ns}.npy"))
#         grad_std = np.load(os.path.join(script_path, '..', 'log-sys_id', f"grads-std-d{d}-ns{ns}.npy"))
#         print(f"Found gradient stats for d={d}, ns={ns}:")
#         print(grad_mean)
#         print(grad_std)
# exit()

d = "2e7"
ns = 20
grad_mean = np.load(os.path.join(script_path, '..', 'log-sys_id', f"grads-mean-d{d}-ns{ns}.npy"))
grad_std = np.load(os.path.join(script_path, '..', 'log-sys_id', f"grads-std-d{d}-ns{ns}.npy"))
E_range = (2.5e5, 4e5)
nu_range = (0.2, 0.4)
rho_range = (1600, 2300)
sand_angle_range = (30, 45)
mf_range = (0.05, 2.0)
sf_range = (0.05, 2.0)
n_epoch = 100
ranges = [E_range, nu_range, rho_range, sand_angle_range, mf_range, sf_range]
"""
Found learning rates:
E: 2e5
nu: 0.1
rho: 1e3
sand_angle: 10
mf: 1
sf: 0.5
"""
lr = [5e4, 0.05, 1e2, 5, 0.5, 0.1]
n = 5
for n in range(6):
    bounds = ranges[n]
    mean = grad_mean[n]
    std = grad_std[n]
    param = np.asarray(np.random.uniform(bounds[0], bounds[1]), dtype=DTYPE_NP).reshape((1,))
    optimiser = Adam(parameters_shape=param.shape,
                     cfg={'lr': lr[n], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

    print(param, mean, std)
    for _ in range(100):
        grad = np.random.uniform(mean-std, mean+std, size=param.shape)
        parm = optimiser.step(param.copy(), grad.copy())
        parm = np.clip(parm, bounds[0], bounds[1])
        print(optimiser.cur_lr, grad, parm)
