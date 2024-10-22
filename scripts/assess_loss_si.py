import os
import open3d as o3d
import logging
import argparse
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import DTYPE_NP, SAND, DTYPE_TI
LINEAR_VELOCITY = 0.2  # m/s
ANGULAR_VELOCITY = np.pi / 4  # rad/s
DT_GLOBAL = 0.01  # sec
SHOVEL_HEIGHT = 0.12
SOIL_HEIGHT = 0.085 + 0.005


def main(args):
    for param in ['-ER', '-NS']:
        d_str = args['ptcl_density']
        case = f'd{d_str}'
        if args['soft_contact']:
            case += '-soft'
        if args['toi_contact']:
            case += '-toi'

        case += f'-res{args["res"]}'
        result_path = os.path.join(script_path, '..', 'log-loss-analysis', case)
        if not args['test']:
            os.makedirs(result_path, exist_ok=True)

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
        cam_cfg = {
            'pos': (0.2, 0.8, 0.7),
            'lookat': (0.2, 0.2, 0.03),
            'euler': (180 + np.rad2deg(np.arctan(1.0 / (0.9 - 0.03))), 0, 180),
            'focal_length': 0.3,
            'fov': 30,
            'lights': [{'pos': (1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
                       {'pos': (1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
                       {'pos': (1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
            'particle_radius': 0.001,
            'res': (800, 800),
            'pcd_gen_res': args['res']
        }

        env_cfg = {
            'p_density': float(args['ptcl_density']),
            'material_id': SAND,
            'grid_scale': 1,
            'horizon': 600,
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

        loss_cfg = {
            'use_height_map_loss': args['use_height_map_loss'],
            'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                            f'pcd_{motion_id}_cropped_norm_z_aligned.ply'),
            'target_pcd_offset': [0.2, 0.2, 0],
            'height_grid_res': args['res'],
        }

        E0 = 2.5e5
        dE = (1e6 - 2.5e5) / 50
        # E_range = (2.5e5, 1e6)
        rho0 = 1200
        drho = (2200 - 1200) / 50
        # rho_range = (1200, 2200)
        nu0 = 0.1
        dnu = (0.4 - 0.1) / 50
        # nu_range = (0.1, 0.4)
        sand_angle0 = 10
        dsand_angle = (40 - 10) / 50
        # sand_angle_range = (10, 40)

        emd_losses = np.zeros((50, 50))
        hm_losses = np.zeros((50, 50))
        for i in range(50):
            for j in range(50):

                ti.reset()
                ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=args['seed'])

                env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg,
                                                    debug_grad=False, logger=logging)

                if param == '-ER':
                    E = np.asarray([E0 + i * dE])
                    rho = np.asarray([rho0 + j * drho])
                    nu = np.asarray([0.3])
                    sand_angle = np.asarray([15])
                else:
                    # param == '-NS'
                    E = np.asarray([6e5])
                    rho = np.asarray([1800])
                    nu = np.asarray([nu0 + i * dnu])
                    sand_angle = np.asarray([sand_angle0 + j * dsand_angle])

                set_parameters(mpm_env, material_id=SAND, e=E.copy(), nu=nu.copy(), rho=rho.copy(),
                               sand_friction_angle=sand_angle.copy(),
                               manipulator_friction=0.5, container_friction=0.5)

                """forward pass"""
                mpm_env.set_state(init_state['state'], grad_enabled=True)
                mpm_env.simulator.trajectory_length = trajectory.shape[0]

                for i in range(trajectory.shape[0]):
                    mpm_env.step(trajectory[i])

                loss_info = mpm_env.get_final_loss()
                print('=====> EMD Loss:', loss_info['emd_loss'])
                print('=====> Height map loss:', loss_info['height_map_loss'])

                # if args['test']:
                #     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                #     ax[0].imshow(mpm_env.loss.height_map.to_numpy(),
                #                  vmin=0.002, vmax=0.09)
                #     ax[0].set_title('Height map')
                #     ax[1].imshow(mpm_env.loss.height_map_pcd_target.to_numpy(),
                #                  vmin=0.002, vmax=0.09)
                #     ax[1].set_title('Target height map')
                #     plt.show()
                #     plt.close()
                #
                #     cloud_array = mpm_env.render(mode='point_cloud')
                #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                #     obj_vec = o3d.utility.Vector3dVector(cloud_array)
                #     obj_pcd = o3d.geometry.PointCloud(obj_vec)
                #     cloud_array_2 = mpm_env.loss.target_pcd_points.to_numpy()
                #     cloud_array_2[:, 0] += 0.3
                #     obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                #     obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2)
                #     cloud_array_3 = mpm_env.loss.target_pcd_original_points_np.copy()
                #     # cloud_array_3[:, 0] -= 0.3
                #     obj_vec_3 = o3d.utility.Vector3dVector(cloud_array_3)
                #     obj_pcd_3 = o3d.geometry.PointCloud(obj_vec_3).paint_uniform_color([0.2, 0.8, 0.2])
                #     print(obj_pcd, obj_pcd_2)
                #     o3d.visualization.draw_geometries([frame, obj_pcd, obj_pcd_2], width=800, height=600)
                #     exit()

                emd_losses[i, j] = loss_info['emd_loss']
                hm_losses[i, j] = loss_info['height_map_loss']

                mpm_env.simulator.clear_ckpt()

        np.save(os.path.join(result_path, f'emd_losses{param}.npy'), emd_losses)
        np.save(os.path.join(result_path, f'hm_losses{param}.npy'), hm_losses)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="System identification for soil")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6, help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--soft-contact', dest='soft_contact', action='store_true', default=False, help='Use soft contact')
    parser.add_argument('--toi-contact', dest='toi_contact', action='store_true', default=False, help='Use time-of-impact contact')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='test')
    parser.add_argument('--res', dest='res', default=60, type=int, help='emd/hm resolution')
    arguments = vars(parser.parse_args())
    main(arguments)
