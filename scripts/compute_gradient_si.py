import os
import yaml
import logging
import argparse
import numpy as np
import taichi as ti
from time import time
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

    trajectory = np.zeros(shape=(160, 6), dtype=DTYPE_NP)
    trajectory[:30, 2] = -0.2
    trajectory[30:130, 0] = -0.3
    trajectory[130:160, 2] = 0.2

    env_cfg = {
        'p_density': arguments['ptcl_density'],
        'material_id': CLAY,
        'horizon': trajectory.shape[0],
        'dt_global': 0.005,
        'n_substeps': 10,
        'agent_init_pos': (0.4, 0.2, 0.2),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_height_map_path': os.path.join(script_path, '..', 'data', 'target_pcds',
                                                   'pcd_2_cropped_norm_z_aligned_height_map.npy'),
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'target_pcds', 'pcd_0_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.01,
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
        manipulator_friction = np.asarray(np.random.uniform(mf_range[0], mf_range[1]), dtype=DTYPE_NP).reshape(
            (1,))  # Manipulator friction

        ti.reset()
        ti.init(arch=backend, device_memory_GB=5, default_fp=ti.f32, fast_math=True, random_seed=1)
        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=args['print_substep_grad'], logger=logging)
        set_parameters(mpm_env, material_id=SAND, E=E.copy(), nu=nu.copy(), rho=rho.copy(),
                       manipulator_friction=manipulator_friction.copy(),
                       sand_friction_angle=sand_angle.copy())
        print(f'===> Created Env with {mpm_env.simulator.n_particles} particles, {mpm_env.loss.n_target_pcd_points} target pcd points.')
        print(f'===> CPU memory occupied after init: {process.memory_percent()} %')
        print(f'===> GPU memory after init: {get_gpu_memory()}')

        def print_grads():
            for j in range(NUM_MATERIAL):
                print(f"Mat {j} Gradient of E: {mpm_env.simulator.particle_param.grad[j].E}")
                print(f"Mat {j} Gradient of nu: {mpm_env.simulator.particle_param.grad[j].nu}")
                print(f"Mat {j} Gradient of rho: {mpm_env.simulator.particle_param.grad[j].rho}")
            print(f"Gradient of sand angle: {mpm_env.simulator.system_param.grad[None].sand_friction_angle}")
            print(f"Gradient of manipulator friction: {mpm_env.simulator.system_param.grad[None].manipulator_friction}")
            print(f"Max epsilon: {mpm_env.simulator.debug_info[None].max_epsilon}")
            print(f"Min epsilon: {mpm_env.simulator.debug_info[None].min_epsilon}")
            print(f"Max delta gamma: {mpm_env.simulator.debug_info[None].max_delta_gamma}")
            print(f"Min delta gamma: {mpm_env.simulator.debug_info[None].min_delta_gamma}")
            print(f"Max stress: {mpm_env.simulator.debug_info[None].max_stress}")
            print(f"Min stress: {mpm_env.simulator.debug_info[None].min_stress}")
            print(f"Max Sigma: {mpm_env.simulator.debug_info[None].max_Sigma}")
            print(f"Min Sigma: {mpm_env.simulator.debug_info[None].min_Sigma}")
            print(f"Max dStress_dF: {mpm_env.simulator.debug_info[None].max_dStress_dF}")
            print(f"Min dStress_dF: {mpm_env.simulator.debug_info[None].min_dStress_dF}")
            print(f"Max dF_dSigma: {mpm_env.simulator.debug_info[None].max_dF_dSigma}")
            print(f"Min dF_dSigma: {mpm_env.simulator.debug_info[None].min_dF_dSigma}")
            print(f"Max dStress_dmu: {mpm_env.simulator.debug_info[None].max_dStress_dmu}")
            print(f"Min dStress_dmu: {mpm_env.simulator.debug_info[None].min_dStress_dmu}")
            print(f"Max dStress_dlam: {mpm_env.simulator.debug_info[None].max_dStress_dlam}")
            print(f"Min dStress_dlam: {mpm_env.simulator.debug_info[None].min_dStress_dlam}")
            print(f"Max dStress_dSigma: {mpm_env.simulator.debug_info[None].max_dStress_dSigma}")
            print(f"Min dStress_dSigma: {mpm_env.simulator.debug_info[None].min_dStress_dSigma}")
            print(f"Max dCentre_dSigma: {mpm_env.simulator.debug_info[None].max_dCentre_dSigma}")
            print(f"Min dCentre_dSigma: {mpm_env.simulator.debug_info[None].min_dCentre_dSigma}")
            print(f"Max dSigma_dmu: {mpm_env.simulator.debug_info[None].max_dSigma_dmu}")
            print(f"Min dSigma_dmu: {mpm_env.simulator.debug_info[None].min_dSigma_dmu}")
            print(f"Max dSigma_dlam: {mpm_env.simulator.debug_info[None].max_dSigma_dlam}")
            print(f"Min dSigma_dlam: {mpm_env.simulator.debug_info[None].min_dSigma_dlam}")
            print(f"Max dSigma_ddeltagamma: {mpm_env.simulator.debug_info[None].max_dSigma_ddeltagamma}")
            print(f"Min dSigma_ddeltagamma: {mpm_env.simulator.debug_info[None].min_dSigma_ddeltagamma}")
            input('Press any key to continue...')

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

        """backward pass"""
        mpm_env.reset_grad()
        if args['debug']:
            print('******Initial grads:')
            print_grads()
        mpm_env.get_final_loss_grad()
        if args['debug']:
            print('******Grads after get_final_loss_grad()')
            print_grads()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)
            if args['debug']:
                print(f'******Grads after step_grad({action})')
                print_grads()

            # This is a trick that prevents faulty gradient computation
            # It works for unknown reasons
            _ = mpm_env.simulator.particle_param.grad[2].E

        print(f'===> CPU memory occupied after forward-backward: {process.memory_percent()} %')
        print(f'===> GPU memory after forward-backward: {get_gpu_memory()}')
        print('===> Gradients:')
        print_grads()

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
                         mpm_env.simulator.system_param.grad[None].manipulator_friction], dtype=DTYPE_NP)

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

    logging.info('===> Avg. Gradients:')
    logging.info(f"Avg. gradient of E: {grad_mean[0]}, std: {grad_std[0]}")
    logging.info(f"Avg. gradient of nu: {grad_mean[1]}, std: {grad_std[1]}")
    logging.info(f"Avg. gradient of rho: {grad_mean[2]}, std: {grad_std[2]}")
    logging.info(f"Avg. gradient of sand angle: {grad_mean[3]}, std: {grad_std[3]}")
    logging.info(f"Avg. gradient of manipulator friction: {grad_mean[4]}, std: {grad_std[4]}")

    if not args['debug']:
        np.save(grad_mean_file_name, grad_mean)
        np.save(grad_std_file_name, grad_std)


if __name__ == '__main__':
    description = 'Compute the means and standard deviations of the gradients for material parameters with randomly sampled parameter values.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--r_human', dest='r_human', default=False, action='store_true', help='Render human view')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=float, default=7e6, help='Particle density')
    parser.add_argument('--debug', dest='debug', default=True, action='store_true', help='Debug mode, print gradients for every global step.')
    parser.add_argument('--debug_substep', dest='print_substep_grad', default=False, action='store_true', help='Debug mode, print gradients for every substep.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    arguments = vars(parser.parse_args())
    main(arguments)
