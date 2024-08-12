import os
import yaml
import logging
import argparse
import numpy as np
import taichi as ti
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
SOIL_HEIGHT = 0.095
LINEAR_VELOCITY = 0.1  # m/s
ANGULAR_VELOCITY = 0.5  # rad/s
DT_GLOBAL = 0.01  # sec


def main(args):
    result_path = os.path.join(script_path, '..', 'log-sys_id')

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    trajectory = np.zeros(shape=(100, 6), dtype=DTYPE_NP)
    trajectory[:30, 2] = -0.1
    trajectory[30:60, 0] = -0.1
    trajectory[60:90, 2] = 0.1

    env_cfg = {
        'p_density': arguments['ptcl_density'],
        'horizon': trajectory.shape[0],
        'dt_global': 0.01,
        'n_substeps': 50,
        'agent_init_pos': (0.4, 0.2, 0.2),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_height_map_path': os.path.join(script_path, '..', 'data', 'target_pcds',
                                                   'pcd_2_cropped_norm_z_aligned_height_map.npy'),
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'target_pcds', 'pcd_2_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.005,
    }

    E_range = (1e4, 3e5)
    rho_range = (1000, 2000)
    nu_range = (0.01, 0.48)
    sand_angle_range = (0, 90)
    mf_range = (0.01, 2.0)
    n_epoch = 100
    if arguments['seed'] != -1:
        seeds = [arguments['seed']]
    else:
        seeds = [0, 1, 2]
    training_config = {
        'lr_E': 4e3,
        'lr_nu': 1e-2,
        'lr_rho': 10,
        'lr_sand_angle': 1e-2,
        'lr_manipulator_friction': 1e-2,
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

        # Initialising parameters
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
        rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]), dtype=DTYPE_NP).reshape((1,))  # Density
        sand_angle = np.asarray(np.random.uniform(sand_angle_range[0], sand_angle_range[1]), dtype=DTYPE_NP).reshape((1,))  # Sand friction angle
        manipulator_friction = np.asarray(mf_range[0], mf_range[1], dtype=DTYPE_NP).reshape((1,))  # Manipulator friction
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

        n_aborted_data = 0
        t1 = time()
        """===========Training==========="""
        loss_names = ['height_map_loss', 'emd_loss', 'total_loss']
        loss_info = {}
        for n in range(n_epoch):
            ti.reset()
            ti.init(arch=backend, device_memory_GB=15, default_fp=ti.f32, fast_math=True, random_seed=arguments['seed'])
            env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)
            set_parameters(mpm_env, material_id=SAND, E=E.copy(), nu=nu.copy(), rho=rho.copy(),
                           manipulator_friction=manipulator_friction.copy(), sand_angle=sand_angle.copy())

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
                             mpm_env.simulator.system_param.grad[None].sand_angle,
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
                for i in range(6):
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

                mpm_env.simulator.clear_ckpt()

        logger.close()
        print('====> Finished training.')
        print('====> Final loss: ', loss_info)
        print(f'====> Final params: E {E}, nu {nu}, rho {rho}, sand_angle {sand_angle}, manipulator_friction {manipulator_friction}')
        logging.info('====> Finished training.')
        logging.info('====> Final loss: ', loss_info)
        logging.info(f'====> Final params: E {E}, nu {nu}, rho {rho}, sand_angle {sand_angle}, manipulator_friction {manipulator_friction}')
        np.save(os.path.join(log_dir, 'final_params.npy'),
                np.array([E, nu, rho, sand_angle, manipulator_friction], dtype=DTYPE_NP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arguments = vars(parser.parse_args())
    main(arguments)
