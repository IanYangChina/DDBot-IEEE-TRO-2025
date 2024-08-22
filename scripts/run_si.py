import os
import yaml
import logging
import argparse
import numpy as np
import taichi as ti
import datetime as dt
from time import time
from torch.utils.tensorboard import SummaryWriter
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.optimiser.adam import Adam
from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import DTYPE_NP, DTYPE_TI, SAND
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


def main(args):
    d_str = args['ptcl_density']
    ns = args['n_substep']
    result_path = os.path.join(script_path, '..', 'log-sys_id', f'd{d_str}_ns{ns}')

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    motion_id = 0
    trajectory = np.load(os.path.join(script_path, '..', 'data',
                                      'moveit_trajectories', f'sys_id_sim_{motion_id}_pos.npy'))
    env_cfg = {
        'p_density': float(arguments['ptcl_density']),
        'horizon': trajectory.shape[0],
        'dt_global': 0.01,
        'n_substeps': args['n_substep'],
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_height_map_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                   f'pcd_{motion_id}_cropped_norm_z_aligned_height_map-res60-vdsize0.001.npy'),
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                        f'pcd_{motion_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.007,
    }

    E_range = (2.5e5, 4e5)
    rho_range = (1600, 2300)
    nu_range = (0.2, 0.4)
    sand_angle_range = (30, 45)
    mf_range = (0.01, 2.0)
    n_epoch = 100
    if arguments['seed'] != -1:
        seeds = [arguments['seed']]
    else:
        seeds = [0, 1, 2, 4, 5]
    training_config = {
        'lr_E': 2e5,
        'lr_nu': 0.1,
        'lr_rho': 1e3,
        'lr_sand_angle': 10,
        'lr_manipulator_friction': 1,
        'lr_container_friction': 0.5,
        'n_epoch': n_epoch,
        'seeds': seeds,
    }

    for seed in seeds:
        log_dir = os.path.join(result_path, f'seed-{seed}')
        os.makedirs(log_dir, exist_ok=True)

        log_file_name = os.path.join(log_dir, 'optimisation.log')
        if os.path.isfile(log_file_name):
            filemode = "a"
        else:
            filemode = "w"
        logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                            filename=log_file_name,
                            format="%(asctime)s %(levelname)s %(message)s")
        logger = SummaryWriter(log_dir=log_dir)
        start_time = dt.datetime.now()
        print(f"===> Start grad computation at:, {start_time.year}-{start_time.month}-{start_time.day} "
              f"{start_time.hour}:{start_time.minute}:{start_time.second}")
        logging.info(f"===> Start grad computation at:, {start_time.year}-{start_time.month}-{start_time.day} "
                     f"{start_time.hour}:{start_time.minute}:{start_time.second}")
        start_time_sec = time()

        # Initialising parameters
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
        rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]), dtype=DTYPE_NP).reshape((1,))  # Density
        sand_angle = np.asarray(np.random.uniform(sand_angle_range[0], sand_angle_range[1]), dtype=DTYPE_NP).reshape((1,))  # Sand friction angle
        manipulator_friction = np.asarray(mf_range[0], mf_range[1], dtype=DTYPE_NP).reshape((1,))  # Manipulator friction
        container_friction = np.asarray(mf_range[0], mf_range[1], dtype=DTYPE_NP).reshape((1,))  # Container friction
        # Optimisers
        optim_E = Adam(parameters_shape=E.shape,
                       cfg={'lr': training_config['lr_E'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_nu = Adam(parameters_shape=nu.shape,
                        cfg={'lr': training_config['lr_nu'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_rho = Adam(parameters_shape=rho.shape,
                         cfg={'lr': training_config['lr_yield_stress'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_sand_angle = Adam(parameters_shape=sand_angle.shape,
                                cfg={'lr': training_config['lr_sand_angle'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_manipulator_friction = Adam(parameters_shape=manipulator_friction.shape,
                                          cfg={'lr': training_config['lr_manipulator_friction'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
        optim_container_friction = Adam(parameters_shape=container_friction.shape,
                                        cfg={'lr': training_config['lr_container_friction'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

        n_aborted_data = 0
        """===========Training==========="""
        loss_names = ['height_map_loss', 'emd_loss', 'total_loss']
        loss_info = {}
        for n in range(n_epoch):
            ti.reset()
            ti.init(arch=backend, device_memory_GB=15, default_fp=ti.f32, fast_math=True, random_seed=arguments['seed'])
            env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)
            set_parameters(mpm_env, material_id=SAND, e=E.copy(), nu=nu.copy(), rho=rho.copy(),
                           manipulator_friction=manipulator_friction.copy(),
                           container_friction=container_friction.copy(),
                           sand_friction_angle=sand_angle.copy())

            """forward pass"""
            mpm_env.set_state(init_state['state'], grad_enabled=True)
            for i in range(mpm_env.horizon):
                mpm_env.step(trajectory[i])
            loss_info = mpm_env.get_final_loss()

            """backward pass"""
            mpm_env.reset_grad()
            mpm_env.get_final_loss_grad()
            for i in range(mpm_env.horizon - 1, -1, -1):
                action = trajectory[i]
                mpm_env.step_grad(action=action)

                # This is a trick that prevents faulty gradient computation
                # It works for unknown reasons
                _ = mpm_env.simulator.particle_param.grad[2].E

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
                if np.any(np.isnan(grad)) or np.any(np.isinf(grad)) or np.any(np.abs(grad) > 1e10):
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
                print(f'===> [Warning] E: {E}, nu: {nu}, rho: {rho}, sand_angle: {sand_angle}, '
                      f'manipulator_friction: {manipulator_friction},'
                      f'container_friction: {container_friction}')
                print(f'===> [Warning] Grad: {grad}')
                print(f'===> [Warning] Loss info: {loss_info}')
                logging.error(f'===> [Warning] Aborting epoch: {n}')
                logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                logging.error(f'===> [Warning] Strange loss or gradient.')
                logging.error(f'===> [Warning] E: {E}, nu: {nu}, rho: {rho}, sand_angle: {sand_angle}, '
                              f'manipulator_friction: {manipulator_friction}'
                              f'container_friction: {container_friction}')
                logging.error(f'===> [Warning] Grad: {grad}')
                logging.error(f'===> [Warning] Loss info: {loss_info}')
                n_aborted_data += 1
            else:
                for loss_name in loss_names:
                    logger.add_scalar(tag=f'Loss/{loss_name}', scalar_value=loss_info[loss_name], global_step=n)
                E = optim_E.step(E.copy(), grad[0])
                E = np.clip(E, E_range[0], E_range[1])
                nu = optim_nu.step(nu.copy(), grad[1])
                nu = np.clip(nu, nu_range[0], nu_range[1])
                rho = optim_rho.step(rho.copy(), grad[2])
                rho = np.clip(rho, rho_range[0], rho_range[1])
                sand_angle = optim_sand_angle.step(sand_angle.copy(), grad[3])
                sand_angle = np.clip(sand_angle, sand_angle_range[0], sand_angle_range[1])
                manipulator_friction = optim_manipulator_friction.step(manipulator_friction.copy(), grad[4])
                manipulator_friction = np.clip(manipulator_friction, mf_range[0], mf_range[1])
                container_friction = optim_container_friction.step(container_friction.copy(), grad[5])
                container_friction = np.clip(container_friction, mf_range[0], mf_range[1])

                print(f'=====> Epoch: {n}')
                print(f'=====> Loss info: {loss_info}')
                print(f'=====> Grad: {grad}')
                print(f"=====> Num. aborted data so far: {n_aborted_data}")
                logging.info(f'=====> Epoch: {n}')
                logging.info(f'=====> Loss info: {loss_info}')
                logging.info(f'=====> Grad: {grad}')
                logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")

                logger.add_scalar(tag='Param/E', scalar_value=E, global_step=n)
                logger.add_scalar(tag='Grad/E', scalar_value=grad[0], global_step=n)
                logger.add_scalar(tag='Param/nu', scalar_value=nu, global_step=n)
                logger.add_scalar(tag='Grad/nu', scalar_value=grad[1], global_step=n)
                logger.add_scalar(tag='Param/rho', scalar_value=rho, global_step=n)
                logger.add_scalar(tag='Grad/rho', scalar_value=grad[2], global_step=n)
                logger.add_scalar(tag='Param/sand_angle', scalar_value=sand_angle, global_step=n)
                logger.add_scalar(tag='Grad/sand_angle', scalar_value=grad[3], global_step=n)
                logger.add_scalar(tag='Param/manipulator_friction', scalar_value=manipulator_friction, global_step=n)
                logger.add_scalar(tag='Grad/manipulator_friction', scalar_value=grad[4], global_step=n)
                logger.add_scalar(tag='Param/container_friction', scalar_value=container_friction, global_step=n)
                logger.add_scalar(tag='Grad/container_friction', scalar_value=grad[5], global_step=n)

                mpm_env.simulator.clear_ckpt()

        logger.close()
        print('====> Finished training.')
        print('====> Final loss: ', loss_info)
        print(f'====> Final params: E {E}, nu {nu}, rho {rho}, sand_angle {sand_angle}, '
              f'manipulator_friction {manipulator_friction},'
              f'container_friction {container_friction}')
        end_time = dt.datetime.now()
        print(f"===> End grad computation at:, {end_time.year}-{end_time.month}-{end_time.day} "
              f"{end_time.hour}:{end_time.minute}:{end_time.second}")
        print(f"===> Total time taken: {time() - start_time_sec} seconds")
        print(f"===> Number of aborted data: {n_aborted_data}")
        logging.info('====> Finished training.')
        logging.info('====> Final loss: ', loss_info)
        logging.info(f'====> Final params: E {E}, nu {nu}, rho {rho}, sand_angle {sand_angle}, '
                     f'manipulator_friction {manipulator_friction},'
                     f'container_friction {container_friction}')
        logging.info(f"===> End grad computation at:, {end_time.year}-{end_time.month}-{end_time.day} "
                     f"{end_time.hour}:{end_time.minute}:{end_time.second}")
        logging.info(f"===> Total time taken: {time() - start_time_sec} seconds")
        logging.info(f"===> Number of aborted data: {n_aborted_data}")

        np.save(os.path.join(log_dir, 'final_params.npy'),
                np.array([E, nu, rho, sand_angle, manipulator_friction, container_friction], dtype=DTYPE_NP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="System identification for soil")
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6,
                        help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str,
                        help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--n_substep', dest='n_substep', default='20', type=int, help='number of simulation substeps')
    arguments = vars(parser.parse_args())
    main(arguments)
