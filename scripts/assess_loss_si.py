import os
import open3d as o3d
import matplotlib.pyplot as plt
import logging
import argparse
import numpy as np
import taichi as ti

script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import SAND, DTYPE_TI
from scipy.optimize import linear_sum_assignment

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
            'pcd_gen_res': 50
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
            'use_height_map_loss': True,
            'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                            f'pcd_{motion_id}_cropped_norm_z_aligned.ply'),
            'target_pcd_offset': [0.2, 0.2, 0],
            'height_grid_res': 40,
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
        emd_losses = [np.zeros((50, 50)) for _ in range(6)]
        hm_losses = [np.zeros((50, 50)) for _ in range(6)]
        for i in range(50):
            for j in range(50):
                ti.reset()
                ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True)

                env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)

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
                mpm_env.set_state(init_state['state'], grad_enabled=False)
                mpm_env.simulator.trajectory_length = trajectory.shape[0]

                for n in range(trajectory.shape[0]):
                    mpm_env.step(trajectory[n])

                loss_info = mpm_env.get_final_loss()
                if args['test']:
                    print('=====> EMD Loss:', loss_info['emd_loss'])
                    print('=====> Height map loss:', loss_info['height_map_loss'])

                print(f'=====> Iter {i}, {j}')
                particles = mpm_env.simulator.get_x()
                p_radius = mpm_env.loss.particle_radius

                @ti.func
                def from_xy_to_uv(x: DTYPE_TI, y: DTYPE_TI, reso: ti.i32):
                    # this does not need to be differentiable as the loss is connected to the z values of the particles/points
                    u = (x - 0.2) / (0.24 / reso) + reso / 2
                    v = (y - 0.2) / (0.24 / reso) + reso / 2
                    return ti.round(u, ti.i32), ti.round(v, ti.i32)

                ll = 0
                for res in [10, 20, 30, 40, 50, 60]:
                    target_pcd = o3d.io.read_point_cloud(os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                                      f'pcd_{motion_id}_cropped_norm_z_aligned_res{res}.ply'))
                    pcd = np.asarray(target_pcd.points, dtype=np.float32) + (0.2, 0.2, 0.0)
                    target_pcd_heightmap = np.load(os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                                f'pcd_{motion_id}_cropped_norm_z_aligned_height_map-res{res}.npy'))

                    n_particles = int(particles.shape[0])
                    x1 = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_particles, needs_grad=True)
                    x1.from_numpy(particles)
                    height_map_1 = ti.field(dtype=DTYPE_TI, shape=(res, res), needs_grad=True)
                    point_id_1 = ti.field(dtype=ti.i32, shape=res * res, needs_grad=False)
                    x1_surface = ti.Vector.field(3, dtype=DTYPE_TI, shape=(res*res), needs_grad=True)

                    n_pcd_points = pcd.shape[0]
                    x2 = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_pcd_points, needs_grad=False)
                    x2.from_numpy(pcd)

                    emd_loss_ti = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
                    emd_ind_pairs = ti.Vector.field(2, dtype=ti.i32, shape=res * res, needs_grad=False)
                    emd_distance_matrix = ti.field(dtype=DTYPE_TI, shape=(res * res, res * res), needs_grad=False)

                    height_map_1_radius = ti.field(dtype=DTYPE_TI, shape=(res, res), needs_grad=True)
                    height_map_2 = ti.field(dtype=DTYPE_TI, shape=(res, res), needs_grad=False)
                    height_map_2.from_numpy(target_pcd_heightmap)
                    height_map_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

                    @ti.func
                    def compute_euclidean_distance(a, b):
                        return ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

                    @ti.kernel
                    def compute_emd_euclidean_distance():
                        for e in range(res * res):
                            q = point_id_1[emd_ind_pairs[e][0]]
                            w = emd_ind_pairs[e][1]
                            d = compute_euclidean_distance(x1[q], x2[w])
                            emd_loss_ti[None] += d

                    @ti.kernel
                    def calculate_height_map():
                        for p in range(n_particles):
                            u, v = from_xy_to_uv(x1[p][0], x1[p][1], res)
                            z = x1[p][2]
                            ti.atomic_max(height_map_1[u, v], z)

                    @ti.kernel
                    def calculate_height_map_with_radius():
                        for p in range(n_particles):
                            u, v = from_xy_to_uv(x1[p][0], x1[p][1], res)
                            u_0, v_0 = from_xy_to_uv(x1[p][0] - p_radius,
                                                     x1[p][1] - p_radius, res)
                            u_1, v_1 = from_xy_to_uv(x1[p][0] + p_radius,
                                                     x1[p][1] + p_radius, res)
                            ti.atomic_max(height_map_1_radius[u, v], (x1[p][2]))
                            ti.atomic_max(height_map_1_radius[u_0, v_0], (x1[p][2]))
                            ti.atomic_max(height_map_1_radius[u_1, v_1], (x1[p][2]))
                            ti.atomic_max(height_map_1_radius[u_0, v_1], (x1[p][2]))
                            ti.atomic_max(height_map_1_radius[u_1, v_0], (x1[p][2]))

                    @ti.kernel
                    def compute_height_map_masks():
                        for p in range(n_particles):
                            u, v = from_xy_to_uv(x1[p][0], x1[p][1], res)
                            if x1[p][2] >= height_map_1[u, v]:
                                point_id_1[u * res + v] = p

                    @ti.kernel
                    def gather_surface_particles():
                        for q in range(res*res):
                            p = point_id_1[q]
                            x1_surface[q] = x1[p]

                    @ti.kernel
                    def compute_emd_distance_matrix_with_ids():
                        for ind in range(res * res):
                            q = point_id_1[ind]
                            for w in range(n_pcd_points):
                                emd_distance_matrix[ind, w] = compute_euclidean_distance(x1[q], x2[w])

                    def compute_emd_distance_bijection():
                        mat = emd_distance_matrix.to_numpy()
                        attempt = 0
                        done = False
                        while not done:
                            attempt += 1
                            try:
                                ind1, ind2 = linear_sum_assignment(mat)
                                indexes = np.stack((ind1, ind2), axis=-1).astype(np.int32)
                                done = True
                            except:
                                print('Error: linear_sum_assignment failed')
                                print(f'D mat shape: {mat.shape}')
                                print(
                                    f'D mat NaN: {np.isnan(mat).any()}, Inf: {np.isinf(mat).any()}, Max: {np.max(mat)}, Min: {np.min(mat)}')
                            if attempt > 5:
                                print('Linear assignment failed for 5 times, exiting calculation.')
                                done = True
                        if done:
                            emd_ind_pairs.from_numpy(indexes)

                    @ti.kernel
                    def compute_height_map_loss():
                        for q, w in height_map_2:
                            d = ti.sqrt((height_map_2[q, w] - height_map_1_radius[q, w]) ** 2)
                            height_map_loss[None] += d

                    x1_surface.fill(0)
                    height_map_1.fill(0.0001)
                    point_id_1.fill(-1)
                    emd_loss_ti.fill(0)
                    emd_ind_pairs.fill(-1)
                    emd_distance_matrix.fill(10000.0)
                    height_map_1_radius.fill(0)
                    height_map_loss.fill(0)

                    calculate_height_map()
                    compute_height_map_masks()
                    gather_surface_particles()
                    compute_emd_distance_matrix_with_ids()
                    compute_emd_distance_bijection()
                    compute_emd_euclidean_distance()
                    calculate_height_map_with_radius()
                    compute_height_map_loss()

                    if args['test']:
                        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
                        ax[0].imshow(height_map_1.to_numpy(),
                                     vmin=0.002, vmax=0.09)
                        ax[0].set_title('Height map')
                        ax[1].imshow(height_map_1_radius.to_numpy(),
                                     vmin=0.002, vmax=0.09)
                        ax[1].set_title('Height map radius')
                        ax[2].imshow(target_pcd_heightmap,
                                     vmin=0.002, vmax=0.09)
                        ax[2].set_title('Target height map')
                        plt.show()
                        plt.close()

                        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                        cloud_array = x1_surface.to_numpy()
                        obj_vec = o3d.utility.Vector3dVector(cloud_array)
                        obj_pcd = o3d.geometry.PointCloud(obj_vec)
                        cloud_array_2 = pcd.copy()
                        cloud_array_2[:, 0] += 0.3
                        obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                        obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2)
                        cloud_array_3 = particles.copy()
                        cloud_array_3[:, 0] += 0.6
                        obj_vec_3 = o3d.utility.Vector3dVector(cloud_array_3)
                        obj_pcd_3 = o3d.geometry.PointCloud(obj_vec_3)
                        print(obj_pcd, obj_pcd_2)
                        o3d.visualization.draw_geometries([frame, obj_pcd, obj_pcd_2, obj_pcd_3], width=800, height=600)

                    print(f'=====> EMD Loss with res {res}:', emd_loss_ti[None])
                    print(f'=====> Height map loss with res {res}:', height_map_loss[None])

                    emd_losses[ll][i, j] = emd_loss_ti[None]
                    hm_losses[ll][i, j] = height_map_loss[None]

                    ll += 1
                mpm_env.simulator.clear_ckpt()

        ll = 0
        for resolution in [10, 20, 30, 40, 50, 60]:
            np.save(os.path.join(result_path, f'emd_losses-res{resolution}-{param}.npy'), emd_losses[ll])
            np.save(os.path.join(result_path, f'hm_losses-res{resolution}-{param}.npy'), hm_losses[ll])
            ll += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="System identification for soil")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6,
                        help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--soft-contact', dest='soft_contact', action='store_true', default=False,
                        help='Use soft contact')
    parser.add_argument('--toi-contact', dest='toi_contact', action='store_true', default=False,
                        help='Use time-of-impact contact')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str,
                        help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='test')
    arguments = vars(parser.parse_args())
    main(arguments)
