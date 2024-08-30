import os
import argparse
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import taichi as ti
from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import DTYPE_NP, SAND
script_path = os.path.dirname(os.path.realpath(__file__))


def main(args):
    saving_folder = os.path.join(script_path, '..', 'render_test')
    os.makedirs(saving_folder, exist_ok=True)
    sys_id_motion = 1
    dt_sim = 0.01
    trajectory = np.load(os.path.join(script_path, '..', 'data',
                                      'moveit_trajectories', f'sys_id_sim_{sys_id_motion}_pos-dt_{dt_sim}.npy'))

    env_cfg = {
        'p_density': float(args['ptcl_density']),
        'material_id': SAND,
        'horizon': trajectory.shape[0],
        'dt_global': dt_sim,
        'grid_scale': 1.0,
        'n_substeps': args['n_substep'],
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_height_map_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                   f'pcd_{sys_id_motion}_cropped_norm_z_aligned_height_map-res60-vdsize0.001.npy'),
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                        f'pcd_{sys_id_motion}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.007,
    }

    cam_cfg = {
        'pos': (0.2, 1.2, 0.8),
        'lookat': (0.2, 0.2, 0.03),
        'euler': (180 + np.rad2deg(np.arctan(1.0 / (0.9 - 0.03))), 0, 180),
        'focal_length': 0.3,
        'fov': 30,
        'lights': [{'pos': (1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
                   {'pos': (1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
        'particle_radius': 0.001,
        'res': (800, 800),
        'pcd_gen_res': 150
    }

    ti.reset()
    ti.init(arch=ti.cuda, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=1)
    env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg)
    with open(os.path.join(script_path, '..', 'log-sys_id', 'best_params.json')) as f:
        best_params = json.load(f)[args['ptcl_density']]["Parameters"]
    set_parameters(mpm_env, SAND,
                   e=best_params['E'],
                   nu=best_params['nu'],
                   rho=best_params['rho'],
                   sand_friction_angle=best_params['sand_angle'])
    mpm_env.set_state(init_state['state'], grad_enabled=False)
    if args['render']:
        mpm_env.render(mode='human')
    for i in range(mpm_env.horizon):
        mpm_env.step(trajectory[i])
        if args['render']:
            mpm_env.render(mode='human')
    loss_info = mpm_env.get_final_loss()
    print('===> Loss info:', loss_info)

    if args['render']:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(mpm_env.loss.height_map.to_numpy(),
                     vmin=0.002, vmax=0.09)
        ax[0].set_title('Height map')
        ax[1].imshow(mpm_env.loss.height_map_pcd_target.to_numpy(),
                     vmin=0.002, vmax=0.09)
        ax[1].set_title('Target height map')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run system identification simulation')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6, help='Particle density')
    parser.add_argument('--ns', dest='n_substep', type=int, default=20, help='Number of substeps')
    parser.add_argument('--cuda_GB', dest='cuda_GB', type=float, default=5, help='CUDA memory in GB')
    parser.add_argument('--r', dest='render', default=False, action='store_true', help='Render the simulation')
    arguments = vars(parser.parse_args())
    main(arguments)
