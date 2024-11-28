import os
import json
# import logging
import argparse
import numpy as np
import taichi as ti
import datetime as dt
import open3d as o3d
import matplotlib.pyplot as plt
from copy import deepcopy as dcp
from time import time
from torch.utils.tensorboard import SummaryWriter
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.optimiser.rmsprop import RMSprop
from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters  #, reset_logging
from doma.engine.configs.macros import DTYPE_NP, DTYPE_TI, SAND
cam_cfg = {
    'pos': (0.2, 0.8, 0.7),
    'lookat': (0.2, 0.2, 0.03),
    'euler': (180+np.rad2deg(np.arctan(1.0/(0.9-0.03))), 0, 180),
    'focal_length': 0.3,
    'fov': 30,
    'lights': [{'pos': (1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
    'particle_radius': 0.001,
    'res': (800, 800),
    'pcd_gen_res': 60
}


def main(args):
    d_str = args['ptcl_density']
    case = f'd{d_str}'
    if args['use_height_map_loss']:
        case += '-hm'

    if args['grad_norm']:
        case += '-gnorm'
    elif args['grad_dy_scale']:
        case += '-gdys'
    elif args['grad_clip']:
        case += '-gclip'
    else:
        case += '-gnone'

    if args['line_search']:
        case += '-ls'

    if args['init_guess']:
        case += '-man-init'

    case += f'-res{args["res"]}'
    result_path = os.path.join(script_path, '..', 'log-sys_id', case)

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    dt_global = 0.01
    trajectory = np.load(os.path.join(script_path, '..', 'data',
                                      'moveit_trajectories', f'sys_id_sim_0_pos-dt_{dt_global}.npy'))
    trajectory_valid = np.load(os.path.join(script_path, '..', 'data',
                                            'moveit_trajectories', f'sys_id_sim_1_pos-dt_{dt_global}.npy'))
    env_cfg = {
        'p_density': float(args['ptcl_density']),
        'material_id': SAND,
        'grid_scale': 1,
        'horizon': trajectory.shape[0],
        'dt_global': dt_global,
        'n_substeps': 20,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }

    if args['grad_norm']:
        env_cfg['grad_op'] = 'normalize'
    elif args['grad_dy_scale']:
        env_cfg['grad_op'] = 'dynamic-scale'
    elif args['grad_clip']:
        env_cfg['grad_op'] = 'clip'
    else:
        env_cfg['grad_op'] = 'none'

    loss_cfg = {
        'use_height_map_loss': args['use_height_map_loss'],
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                        f'pcd_0_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'height_grid_res': args["res"],
    }
    loss_cfg_eval = dcp(loss_cfg)
    # loss_cfg_eval['height_grid_res'] = 60
    loss_cfg_valid = dcp(loss_cfg)
    loss_cfg_valid['target_pcd_path'] = os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                        f'pcd_1_cropped_norm_z_aligned.ply')
    # loss_cfg_valid['height_grid_res'] = 60

    E_range = (5e4, 2e5)
    rho_range = (1200, 2200)
    nu_range = (0.1, 0.4)
    sand_angle_range = (10, 40)
    n_epoch = 20
    if args['seed'] != -1:
        seeds = [args['seed']]
    else:
        seeds = [0, 1, 2, 3, 4]
    training_config = {
        'lr_E': 1e4,
        'lr_nu': 0.01,
        'lr_rho': 50,
        'lr_sand_angle': 1,
        'n_epoch': n_epoch,
        'seeds': seeds,
    }

    for seed in seeds:
        log_dir = os.path.join(result_path, f'seed-{seed}')

        if args['eval']:
            print(f"Loading best parameters from {result_path} seed {seed}...")
            with open(os.path.join(result_path, f'seed-{seed}',
                                   'best_heightmap_loss.json')) as f:
                params = json.load(f)["Parameters"]
            E = np.asarray(params['E'])
            nu = np.asarray(params['nu'])
            rho = np.asarray(params['rho'])
            sand_angle = np.asarray(params['sand_angle'])

            print("Loaded parameters: ", E, nu, rho, sand_angle)
        else:
            os.makedirs(log_dir, exist_ok=True)
            logger = SummaryWriter(log_dir=log_dir)
            start_time = dt.datetime.now()
            print(f"===> Start system identification optimisation at: {start_time.year}-{start_time.month}-{start_time.day} "
                  f"{start_time.hour}:{start_time.minute}:{start_time.second}")

            # Initialising parameters
            E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
            nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
            rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]), dtype=DTYPE_NP).reshape((1,))  # Density
            sand_angle = np.asarray(np.random.uniform(sand_angle_range[0], sand_angle_range[1]), dtype=DTYPE_NP).reshape((1,))  # Sand friction angle
            if args['init_guess']:
                E = np.asarray(8e4, dtype=DTYPE_NP).reshape((1,))
                nu = np.asarray(0.15, dtype=DTYPE_NP).reshape((1,))
                rho = np.asarray(1800, dtype=DTYPE_NP).reshape((1,))
                sand_angle = np.asarray(15, dtype=DTYPE_NP).reshape((1,))
            # Optimisers
            optim_E = RMSprop(parameters_shape=E.shape,
                              cfg={'lr': training_config['lr_E'], 'beta': 0.9})
            optim_nu = RMSprop(parameters_shape=nu.shape,
                               cfg={'lr': training_config['lr_nu'], 'beta': 0.9})
            optim_rho = RMSprop(parameters_shape=rho.shape,
                                cfg={'lr': training_config['lr_rho'], 'beta': 0.9})
            optim_sand_angle = RMSprop(parameters_shape=sand_angle.shape,
                                       cfg={'lr': training_config['lr_sand_angle'], 'beta': 0.9})

        start_time_sec = time()
        n_aborted_data = 0
        loss_names = ['height_map_loss', 'emd_loss']
        loss_info_validation = {}
        for n in range(n_epoch):
            """======================================Validation"""
            ti.reset()
            ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=args['seed'])
            env_cfg['horizon'] = trajectory_valid.shape[0]
            env, mpm_env, init_state = make_env(env_cfg, loss_cfg_valid,
                                                cam_cfg=cam_cfg, debug_grad=False)
            set_parameters(mpm_env, material_id=SAND, e=E.copy(), nu=nu.copy(), rho=rho.copy(),
                           sand_friction_angle=sand_angle.copy(),
                           manipulator_friction=0.5, container_friction=0.5)

            """forward pass"""
            mpm_env.set_state(init_state['state'], grad_enabled=False)
            if args['eval']:
                mpm_env.render('human')
            for i in range(mpm_env.horizon):
                mpm_env.step(trajectory_valid[i])
                if args['eval']:
                    mpm_env.render('human')
            loss_info_validation = mpm_env.get_final_loss()

            print(f'===========> Epoch: {n} validation losses')
            print('=====> EMD Loss:', loss_info_validation['emd_loss'])
            print('=====> Height map loss:', loss_info_validation['height_map_loss'])
            for loss_name in loss_names:
                logger.add_scalar(tag=f'Validation Loss/{loss_name}', scalar_value=loss_info_validation[loss_name], global_step=n)

            if args['eval']:
                print(mpm_env.loss.height_map.shape)
                print('=====> EMD Loss:', loss_info_validation['emd_loss'])
                print('=====> Height map loss:', loss_info_validation['height_map_loss'])
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(mpm_env.loss.height_map.to_numpy(),
                             vmin=0.002, vmax=0.09)
                ax[0].set_title('Height map')
                ax[1].imshow(mpm_env.loss.height_map_pcd_target.to_numpy(),
                             vmin=0.002, vmax=0.09)
                ax[1].set_title('Target height map')
                plt.show()
                plt.close()

                cloud_array = mpm_env.render(mode='point_cloud')
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                obj_vec = o3d.utility.Vector3dVector(cloud_array)
                obj_pcd = o3d.geometry.PointCloud(obj_vec)
                cloud_array_2 = mpm_env.loss.target_pcd_original_points_np.copy()
                obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2).voxel_down_sample(0.007)
                bbx = obj_pcd_2.get_axis_aligned_bounding_box()
                obj_pcd = obj_pcd.crop(bbx)
                cloud_array_2[:, 0] += 0.3
                obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2).voxel_down_sample(0.007)
                o3d.visualization.draw_geometries([frame, obj_pcd, obj_pcd_2], width=800, height=600)
                exit()

            mpm_env.simulator.clear_ckpt()

            """======================================Evaluation"""
            ti.reset()
            ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=args['seed'])
            env_cfg['horizon'] = trajectory.shape[0]
            env, mpm_env, init_state = make_env(env_cfg, loss_cfg_eval, cam_cfg=cam_cfg, debug_grad=False)
            set_parameters(mpm_env, material_id=SAND, e=E.copy(), nu=nu.copy(), rho=rho.copy(),
                           sand_friction_angle=sand_angle.copy(),
                           manipulator_friction=0.5, container_friction=0.5)

            """forward pass"""
            mpm_env.set_state(init_state['state'], grad_enabled=True)
            if args['eval']:
                mpm_env.render('human')
            for i in range(mpm_env.horizon):
                mpm_env.step(trajectory[i])
                if args['eval']:
                    mpm_env.render('human')
            loss_info_eval = mpm_env.get_final_loss()

            print(f'===========> Epoch: {n} evaluation losses')
            print('=====> EMD Loss:', loss_info_eval['emd_loss'])
            print('=====> Height map loss:', loss_info_eval['height_map_loss'])

            for loss_name in loss_names:
                logger.add_scalar(tag=f'Evaluation Loss/{loss_name}', scalar_value=loss_info_eval[loss_name], global_step=n)

            mpm_env.simulator.clear_ckpt()

            """======================================Optimisation"""
            ti.reset()
            ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=args['seed'])
            env_cfg['horizon'] = trajectory.shape[0]
            env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False)
            set_parameters(mpm_env, material_id=SAND, e=E.copy(), nu=nu.copy(), rho=rho.copy(),
                           sand_friction_angle=sand_angle.copy(),
                           manipulator_friction=0.5, container_friction=0.5)

            """forward pass"""
            mpm_env.set_state(init_state['state'], grad_enabled=True)
            if args['eval']:
                mpm_env.render('human')
            for i in range(mpm_env.horizon):
                mpm_env.step(trajectory[i])
                if args['eval']:
                    mpm_env.render('human')
            loss_info = mpm_env.get_final_loss()

            """backward pass"""
            mpm_env.reset_grad()
            mpm_env.get_final_loss_grad()
            for i in range(mpm_env.horizon - 1, -1, -1):
                action = trajectory[i]
                mpm_env.step_grad(action=action)

                # This is a trick that prevents faulty gradient computation
                # It works for unknown reasons
                _ = mpm_env.simulator.particle_param.grad[SAND].E

            grad = np.array([mpm_env.simulator.particle_param.grad[SAND].E,
                             mpm_env.simulator.particle_param.grad[SAND].nu,
                             mpm_env.simulator.particle_param.grad[SAND].rho,
                             mpm_env.simulator.system_param.grad[None].sand_friction_angle], dtype=DTYPE_NP)

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
                print(f'===> [Warning] E: {E}, nu: {nu}, rho: {rho}, sand_angle: {sand_angle}')
                print(f'===> [Warning] Grad: {grad}')
                print(f'===> [Warning] Loss info: {loss_info}')
                n_aborted_data += 1
            else:
                print(f'===========> Epoch: {n} optimisation losses')
                print('=====> EMD Loss:', loss_info['emd_loss'])
                print('=====> Height map loss:', loss_info['height_map_loss'])

                for loss_name in loss_names:
                    logger.add_scalar(tag=f'Loss/{loss_name}', scalar_value=loss_info[loss_name], global_step=n)

                if args['line_search']:
                    print(f'===========> Performing line search for epoch {n}......')
                    loss_to_optimise = 'height_map_loss'
                    best_loss_tmp = +np.inf
                    best_alpha = 1.0
                    best_loss_info = loss_info
                    for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
                        optim_E.lr = training_config['lr_E'] * alpha
                        optim_nu.lr = training_config['lr_nu'] * alpha
                        optim_rho.lr = training_config['lr_rho'] * alpha
                        optim_sand_angle.lr = training_config['lr_sand_angle'] * alpha
                        E_ = optim_E.step(E.copy(), grad[0])
                        E_ = np.clip(E_, E_range[0], E_range[1])
                        nu_ = optim_nu.step(nu.copy(), grad[1])
                        nu_ = np.clip(nu_, nu_range[0], nu_range[1])
                        rho_ = optim_rho.step(rho.copy(), grad[2])
                        rho_ = np.clip(rho_, rho_range[0], rho_range[1])
                        sand_angle_ = optim_sand_angle.step(sand_angle.copy(), grad[3])
                        sand_angle_ = np.clip(sand_angle_, sand_angle_range[0], sand_angle_range[1])

                        mpm_env.simulator.clear_ckpt()

                        ti.reset()
                        ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True,
                                random_seed=args['seed'])
                        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False)
                        set_parameters(mpm_env, material_id=SAND, e=E_.copy(), nu=nu_.copy(), rho=rho_.copy(),
                                       sand_friction_angle=sand_angle_.copy(),
                                       manipulator_friction=0.5, container_friction=0.5)

                        """forward pass"""
                        mpm_env.set_state(init_state['state'], grad_enabled=True)
                        for i in range(mpm_env.horizon):
                            mpm_env.step(trajectory[i])
                        loss_info = mpm_env.get_final_loss()

                        print(f'=====> alpha: {alpha}')
                        print('==> EMD Loss:', loss_info['emd_loss'])
                        print('==> Height map loss:', loss_info['height_map_loss'])
                        if loss_info[loss_to_optimise] < best_loss_tmp:
                            best_loss_tmp = loss_info[loss_to_optimise]
                            best_alpha = alpha
                            best_loss_info = loss_info

                        optim_E.reverse_normaliser()
                        optim_nu.reverse_normaliser()
                        optim_rho.reverse_normaliser()
                        optim_sand_angle.reverse_normaliser()

                        mpm_env.simulator.clear_ckpt()

                    print(f'=====> Best alpha: {best_alpha}, Best loss info: {best_loss_info}')
                    optim_E.lr = training_config['lr_E'] * best_alpha
                    optim_nu.lr = training_config['lr_nu'] * best_alpha
                    optim_rho.lr = training_config['lr_rho'] * best_alpha
                    optim_sand_angle.lr = training_config['lr_sand_angle'] * best_alpha
                    logger.add_scalar(tag='Grad/best_alpha', scalar_value=best_alpha, global_step=n)

                E = optim_E.step(E.copy(), grad[0])
                E = np.clip(E, E_range[0], E_range[1])
                nu = optim_nu.step(nu.copy(), grad[1])
                nu = np.clip(nu, nu_range[0], nu_range[1])
                rho = optim_rho.step(rho.copy(), grad[2])
                rho = np.clip(rho, rho_range[0], rho_range[1])
                sand_angle = optim_sand_angle.step(sand_angle.copy(), grad[3])
                sand_angle = np.clip(sand_angle, sand_angle_range[0], sand_angle_range[1])

                print(f'==> Param: E: {E}, nu: {nu}, rho: {rho}, sand_angle: {sand_angle}')
                print(f'==> Grad: {grad}')
                print(f"==> Num. aborted data so far: {n_aborted_data}")

                logger.add_scalar(tag='Param/E', scalar_value=E, global_step=n)
                logger.add_scalar(tag='Grad/E', scalar_value=grad[0], global_step=n)
                logger.add_scalar(tag='Param/nu', scalar_value=nu, global_step=n)
                logger.add_scalar(tag='Grad/nu', scalar_value=grad[1], global_step=n)
                logger.add_scalar(tag='Param/rho', scalar_value=rho, global_step=n)
                logger.add_scalar(tag='Grad/rho', scalar_value=grad[2], global_step=n)
                logger.add_scalar(tag='Param/sand_angle', scalar_value=sand_angle, global_step=n)
                logger.add_scalar(tag='Grad/sand_angle', scalar_value=grad[3], global_step=n)

            mpm_env.simulator.clear_ckpt()

        logger.close()
        print('===========> Finished training.')
        print('====> Final validation loss: ', loss_info_validation)
        print(f'====> Final params: E {E}, nu {nu}, rho {rho}, sand_angle {sand_angle}')
        end_time = dt.datetime.now()
        print(f"===> End system identification optimisation at:, {end_time.year}-{end_time.month}-{end_time.day} "
              f"{end_time.hour}:{end_time.minute}:{end_time.second}")
        print(f"===> Total time taken: {time() - start_time_sec} seconds")
        print(f"===> Number of aborted data: {n_aborted_data}")

        np.save(os.path.join(log_dir, 'final_params.npy'),
                np.array([E, nu, rho, sand_angle], dtype=DTYPE_NP))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="System identification for soil")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default='5e6', help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--grad-clip', dest='grad_clip', action='store_true', default=False, help='Use gradient clipping')
    parser.add_argument('--grad-norm', dest='grad_norm', action='store_true', default=False, help='Use gradient normalisation')
    parser.add_argument('--grad-dy-scale', dest='grad_dy_scale', action='store_true', default=False, help='Use gradient dynamic scaling')
    parser.add_argument('--line-search', dest='line_search', action='store_true', default=False, help='Use line search')
    parser.add_argument('--init-guess', dest='init_guess', action='store_true', default=False, help='Use initial guess')
    parser.add_argument('--res', dest='res', default=60, type=int, help='Height map resolution')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Evaluate the model')
    arguments = vars(parser.parse_args())
    main(arguments)
