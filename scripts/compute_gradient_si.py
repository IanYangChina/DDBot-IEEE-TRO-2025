import os
import yaml
import logging
import argparse
import open3d as o3d
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from doma.engine.utils.misc import get_gpu_memory
import psutil
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import DTYPE_NP, DTYPE_TI, SAND, NUM_MATERIAL, CLAY
cam_cfg = {
    'pos': (0.2, 1.2, 0.9),
    'lookat': (0.2, 0.2, 0.03),
    'euler': (180+np.rad2deg(np.arctan(1.0/(0.9-0.03))), 0, 180),
    'focal_length': 0.3,
    'fov': 30,
    'lights': [{'pos': (1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
    'particle_radius': 0.001,
    'res': (800, 800),
    'pcd_gen_res': 150
}
SOIL_HEIGHT = 0.095
LINEAR_VELOCITY = 0.1  # m/s
ANGULAR_VELOCITY = 0.5  # rad/s
DT_GLOBAL = 0.01  # sec


def main(args):
    result_path = os.path.join(script_path, '..', 'log-sys_id')
    process = psutil.Process(os.getpid())

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu
    if args['debug']:
        print('[Warning] Debug mode on, printing gradients.')

    task_id = 0
    trajectory = np.load(os.path.join(script_path, '..', 'data',
                                      'moveit_trajectories', f'sys_id_sim_{task_id}_pos.npy'))

    env_cfg = {
        'p_density': arguments['ptcl_density'],
        'material_id': SAND,
        'horizon': trajectory.shape[0],
        'dt_global': 0.01,
        'n_substeps': 10,
        'agent_init_pos': (0.2, 0.2, 0.2),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_height_map_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                   f'pcd_{task_id}_cropped_norm_z_aligned_height_map-res60-vdsize0.001.npy'),
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.006,
    }

    E_range = (2.5e5, 4e5)
    rho_range = (1600, 2300)
    nu_range = (0.2, 0.4)
    sand_angle_range = (30, 45)
    mf_range = (0.01, 2.0)
    n_epoch = 100

    os.makedirs(result_path, exist_ok=True)
    log_file_name = os.path.join(result_path, 'gradient.log')
    if os.path.isfile(log_file_name):
        filemode = "a"
    else:
        filemode = "w"
    logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                        filename=log_file_name,
                        format="%(asctime)s %(levelname)s %(message)s")
    grad_mean_file_name = os.path.join(result_path, f'grads-mean.npy')
    grad_std_file_name = os.path.join(result_path, f'grads-std.npy')

    grads = []
    n_aborted_data = 0
    for n in range(n_epoch):
        # Initialising parameters
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape(
            (1,))  # Poisson's ratio
        rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]), dtype=DTYPE_NP).reshape((1,))  # Density
        sand_angle = np.asarray(np.random.uniform(sand_angle_range[0], sand_angle_range[1]),
                                dtype=DTYPE_NP).reshape((1,))  # Sand friction angle
        manipulator_friction = np.asarray(np.random.uniform(mf_range[0], mf_range[1]), dtype=DTYPE_NP).reshape((1,))  # Manipulator friction
        container_friction = np.asarray(np.random.uniform(mf_range[0], mf_range[1]), dtype=DTYPE_NP).reshape((1,))  # Container friction

        ti.reset()
        ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=1)
        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=args['debug'], logger=logging)
        set_parameters(mpm_env, material_id=SAND, e=E.copy(), nu=nu.copy(), rho=rho.copy(),
                       manipulator_friction=manipulator_friction.copy(),
                       container_friction=container_friction.copy(),
                       sand_friction_angle=sand_angle.copy())
        print(f'===> Created Env with {mpm_env.simulator.n_particles} particles, {mpm_env.loss.n_target_pcd_points} target pcd points.')
        print(f'===> CPU memory occupied after init: {process.memory_percent()} %')
        print(f'===> GPU memory after init: {get_gpu_memory()}')

        """forward pass"""
        mpm_env.set_state(init_state['state'], grad_enabled=True)
        if args['r_human']:
            mpm_env.render(mode='human')
        for i in range(mpm_env.horizon):
            mpm_env.step(trajectory[i])
            if args['r_human']:
                mpm_env.render(mode='human')
        loss_info = mpm_env.get_final_loss()
        print('===> Loss info:', loss_info)
        print(f'===> CPU memory occupied after forward: {process.memory_percent()} %')
        print(f'===> GPU memory after forward: {get_gpu_memory()}')
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].imshow(loss_info['height_map'])
        # ax[0].set_title('Height map')
        # ax[1].imshow(loss_info['height_map_target'])
        # ax[1].set_title('Target height map')
        # plt.show()
        # exit()

        """backward pass"""
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

            # This is a trick that prevents faulty gradient computation
            # It works for unknown reasons
            _ = mpm_env.simulator.particle_param.grad[2].E

        print(f'===> CPU memory occupied after forward-backward: {process.memory_percent()} %')
        print(f'===> GPU memory after forward-backward: {get_gpu_memory()}')

        logging.info(f'===> CPU memory occupied after forward-backward: {process.memory_percent()} %')
        logging.info(f'===> GPU memory after forward-backward: {get_gpu_memory()}')
        logging.info('===> Gradients:')
        logging.info(f"Gradient of E: {mpm_env.simulator.particle_param.grad[SAND].E}")
        logging.info(f"Gradient of nu: {mpm_env.simulator.particle_param.grad[SAND].nu}")
        logging.info(f"Gradient of rho: {mpm_env.simulator.particle_param.grad[SAND].rho}")
        logging.info(f"Gradient of sand angle: {mpm_env.simulator.system_param.grad[None].sand_friction_angle}")
        logging.info(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")

        grad = np.array([mpm_env.simulator.particle_param.grad[SAND].E,
                         mpm_env.simulator.particle_param.grad[SAND].nu,
                         mpm_env.simulator.particle_param.grad[SAND].rho,
                         mpm_env.simulator.system_param.grad[None].sand_friction_angle,
                         mpm_env.simulator.system_param.grad[None].manipulator_friction,
                         mpm_env.simulator.system_param.grad[None].container_friction], dtype=DTYPE_NP)

        """Checking for nan, inf, and strange values"""
        abort = False
        particle_has_naninf = loss_info['particle_has_naninf']
        if particle_has_naninf:
            abort = True

        if not abort:
            for i, v in loss_info.items():
                if i == 'particle_has_naninf':
                    pass
                else:
                    if np.isinf(v) or np.isnan(v):
                        abort = True
                        break

        if not abort:
            if np.any(np.isnan(grad)) or np.any(np.isinf(grad)) or np.any(np.abs(grad) > 1e6):
                abort = True

        if not abort:
            num_zero_grad = 0
            for i in range(grad.shape[0]):
                if grad[i] == 0.0:
                    num_zero_grad += 1
            if num_zero_grad > 4:
                abort = True

        if abort:
            print(f'===> [Warning] Aborting epoch {n}......')
            print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
            print(f'===> [Warning] Strange loss or gradient.')
            print(f'===> [Warning] E: {E}, nu: {nu}, rho: {rho}, sand_angle: {sand_angle}, manipulator_friction: {manipulator_friction}')
            print(f'===> [Warning] Grad: {grad}')
            print(f'===> [Warning] Loss info: {loss_info}')
            logging.error(f'===> [Warning] Aborting epoch: {n}')
            logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
            logging.error(f'===> [Warning] Strange loss or gradient.')
            logging.error(f'===> [Warning] E: {E}, nu: {nu}, rho: {rho}, sand_angle: {sand_angle}, manipulator_friction: {manipulator_friction}')
            logging.error(f'===> [Warning] Grad: {grad}')
            logging.error(f'===> [Warning] Loss info: {loss_info}')
            n_aborted_data += 1
        else:
            grads.append(grad.copy())

        mpm_env.simulator.clear_ckpt()

    grad_mean = np.mean(grads, axis=0)
    grad_std = np.std(grads, axis=0)
    print('===> Avg. Gradients:')
    print(f"Avg. gradient of E: {grad_mean[0]}, std: {grad_std[0]}")
    print(f"Avg. gradient of nu: {grad_mean[1]}, std: {grad_std[1]}")
    print(f"Avg. gradient of rho: {grad_mean[2]}, std: {grad_std[2]}")
    print(f"Avg. gradient of sand angle: {grad_mean[3]}, std: {grad_std[3]}")
    print(f"Avg. gradient of manipulator friction: {grad_mean[4]}, std: {grad_std[4]}")
    print(f"Avg. gradient of container friction: {grad_mean[5]}, std: {grad_std[5]}")

    logging.info('===> Avg. Gradients:')
    logging.info(f"Avg. gradient of E: {grad_mean[0]}, std: {grad_std[0]}")
    logging.info(f"Avg. gradient of nu: {grad_mean[1]}, std: {grad_std[1]}")
    logging.info(f"Avg. gradient of rho: {grad_mean[2]}, std: {grad_std[2]}")
    logging.info(f"Avg. gradient of sand angle: {grad_mean[3]}, std: {grad_std[3]}")
    logging.info(f"Avg. gradient of manipulator friction: {grad_mean[4]}, std: {grad_std[4]}")
    logging.info(f"Avg. gradient of container friction: {grad_mean[5]}, std: {grad_std[5]}")

    if not args['debug']:
        np.save(grad_mean_file_name, grad_mean)
        np.save(grad_std_file_name, grad_std)


if __name__ == '__main__':
    description = 'Compute the means and standard deviations of the gradients for material parameters with randomly sampled parameter values.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--r_human', dest='r_human', default=False, action='store_true', help='Render human view')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=5e6, help='Particle density')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true', help='Debug mode, print gradients for every global step.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    arguments = vars(parser.parse_args())
    main(arguments)
