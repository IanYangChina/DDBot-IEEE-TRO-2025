import os
import json
import logging
import argparse
import numpy as np
import taichi as ti
from torch.utils.tensorboard import SummaryWriter
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import DTYPE_NP, SAND
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
    if args['soft_contact']:
        case += '-soft'
    if args['toi_contact']:
        case += '-toi'
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
    result_path = os.path.join(script_path, '..', 'log-grad-analysis', case)

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    motion_id = 0
    dt_global = 0.01
    trajectory = np.load(os.path.join(script_path, '..', 'data',
                                      'moveit_trajectories', f'sys_id_sim_{motion_id}_pos-dt_{dt_global}.npy'))
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
    if args['soft_contact']:
        assert not args['toi_contact']
        env_cfg['collide_type'] = 'soft'
    elif args['toi_contact']:
        assert not args['soft_contact']
        env_cfg['collide_type'] = 'toi'
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
        'target_pcd_height_map_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                   f'pcd_{motion_id}_cropped_norm_z_aligned_height_map-res60-vdsize0.001.npy'),
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                        f'pcd_{motion_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.007,
    }

    E_range = (2.5e5, 1e6)
    rho_range = (1200, 2200)
    nu_range = (0.1, 0.4)
    sand_angle_range = (10, 40)
    if args['seed'] != -1:
        seeds = [args['seed']]
    else:
        seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        print(f'Running seed {seed}')
        log_dir = os.path.join(result_path, f'seed-{seed}')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir=log_dir)

        # Initialising parameters
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
        rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]), dtype=DTYPE_NP).reshape((1,))  # Density
        sand_angle = np.asarray(np.random.uniform(sand_angle_range[0], sand_angle_range[1]), dtype=DTYPE_NP).reshape((1,))  # Sand friction angle

        ti.reset()
        ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=args['seed'])
        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)
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

        """backward pass"""
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(mpm_env.horizon - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

            # This is a trick that prevents faulty gradient computation
            # It works for unknown reasons
            _ = mpm_env.simulator.particle_param.grad[SAND].E

            mpm_env.simulator.reset_particle_grid_grad_records()
            mpm_env.simulator.get_min_max_p_grad(mpm_env.simulator.cur_substep_local)
            mpm_env.simulator.get_min_max_grid_grad(mpm_env.simulator.cur_substep_local)

            logger.add_scalar('Grad-p/x_min', mpm_env.simulator.debug_info[None].min_particle_grad_x, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-p/x_max', mpm_env.simulator.debug_info[None].max_particle_grad_x, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-p/v_min', mpm_env.simulator.debug_info[None].min_particle_grad_v, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-p/v_max', mpm_env.simulator.debug_info[None].max_particle_grad_v, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-p/F_min', mpm_env.simulator.debug_info[None].min_particle_grad_F, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-p/F_max', mpm_env.simulator.debug_info[None].max_particle_grad_F, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-grid/mass_min', mpm_env.simulator.debug_info[None].min_grid_grad_mass, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-grid/mass_max', mpm_env.simulator.debug_info[None].max_grid_grad_mass, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-grid/v_in_min', mpm_env.simulator.debug_info[None].min_grid_grad_v_in, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-grid/v_in_max', mpm_env.simulator.debug_info[None].max_grid_grad_v_in, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-grid/v_out_min', mpm_env.simulator.debug_info[None].min_grid_grad_v_out, mpm_env.horizon-1-i)
            logger.add_scalar('Grad-grid/v_out_max', mpm_env.simulator.debug_info[None].max_grid_grad_v_out, mpm_env.horizon-1-i)

            logger.add_scalar('Grad_param/E', mpm_env.simulator.particle_param.grad[SAND].E, mpm_env.horizon-1-i)
            logger.add_scalar('Grad_param/nu', mpm_env.simulator.particle_param.grad[SAND].nu, mpm_env.horizon-1-i)
            logger.add_scalar('Grad_param/rho', mpm_env.simulator.particle_param.grad[SAND].rho, mpm_env.horizon-1-i)
            logger.add_scalar('Grad_param/sand_angle', mpm_env.simulator.system_param.grad[None].sand_friction_angle, mpm_env.horizon-1-i)

        mpm_env.simulator.clear_ckpt()
        print(f'Seed {seed} done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="System identification for soil")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6, help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--grad-clip', dest='grad_clip', action='store_true', default=False, help='Use gradient clipping')
    parser.add_argument('--grad-norm', dest='grad_norm', action='store_true', default=False, help='Use gradient normalisation')
    parser.add_argument('--grad-dy-scale', dest='grad_dy_scale', action='store_true', default=False, help='Use gradient dynamic scaling')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--soft-contact', dest='soft_contact', action='store_true', default=False, help='Use soft contact')
    parser.add_argument('--toi-contact', dest='toi_contact', action='store_true', default=False, help='Use time-of-impact contact')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Evaluate the model')
    arguments = vars(parser.parse_args())
    main(arguments)
