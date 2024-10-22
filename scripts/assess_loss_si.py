import os
import open3d as o3d
import logging
import argparse
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
script_path = os.path.dirname(os.path.realpath(__file__))
from scipy.optimize import linear_sum_assignment

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
            'height_grid_res': 50,
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
                mpm_env.set_state(init_state['state'], grad_enabled=False)
                mpm_env.simulator.trajectory_length = trajectory.shape[0]

                for n in range(trajectory.shape[0]):
                    mpm_env.step(trajectory[n])

                ll = 0
                for res in [10, 20, 30, 40, 50, 60]:
                    target_pcd = o3d.io.read_point_cloud(os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                                      f'pcd_{motion_id}_cropped_norm_z_aligned_res{res}.ply'))
                    target_pcd_points = np.asarray(target_pcd.points)
                    target_pcd_points[:, 0] += 0.2
                    target_pcd_ti = ti.Vector.field(3, dtype=DTYPE_TI, shape=target_pcd_points.shape[0])
                    target_pcd_ti.from_numpy(target_pcd_points)

                    height_map = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=False)
                    height_grid = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=False)
                    p_id = ti.field(dtype=ti.i32, shape=res*res, needs_grad=False)
                    emd_distance_matrix = ti.field(dtype=ti.f32, shape=(res * res, res * res), needs_grad=False)

                    @ti.func
                    def from_xy_to_uv(x: DTYPE_TI, y: DTYPE_TI):
                        # this does not need to be differentiable as the loss is connected to the z values of the particles/points
                        u = (x - 0.2) / 0.24 + res / 2
                        v = (y - 0.2) / 0.24 + res / 2
                        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

                    @ti.kernel
                    def calculate_height_map(f: ti.i32):
                        for p in range(mpm_env.loss.n_particles):
                            if mpm_env.loss.particle_mat[p] == mpm_env.loss.matching_mat:
                                u, v = from_xy_to_uv(mpm_env.loss.particle_x[f, p][0], mpm_env.loss.particle_x[f, p][1])
                                u_0, v_0 = from_xy_to_uv(mpm_env.loss.particle_x[f, p][0] - mpm_env.loss.particle_radius,
                                                         mpm_env.loss.particle_x[f, p][1] - mpm_env.loss.particle_radius)
                                u_1, v_1 = from_xy_to_uv(mpm_env.loss.particle_x[f, p][0] + mpm_env.loss.particle_radius,
                                                         mpm_env.loss.particle_x[f, p][1] + mpm_env.loss.particle_radius)
                                ti.atomic_max(height_map[u, v], (mpm_env.loss.particle_x[f, p][2]))
                                ti.atomic_max(height_map[u_0, v_0], (mpm_env.loss.particle_x[f, p][2]))
                                ti.atomic_max(height_map[u_1, v_1], (mpm_env.loss.particle_x[f, p][2]))
                                ti.atomic_max(height_map[u_0, v_1], (mpm_env.loss.particle_x[f, p][2]))
                                ti.atomic_max(height_map[u_1, v_0], (mpm_env.loss.particle_x[f, p][2]))

                    height_map.fill(0.0)
                    calculate_height_map(mpm_env.simulator.cur_substep_local)

                    @ti.kernel
                    def calculate_height_grid(f: ti.i32):
                        for p in range(mpm_env.loss.n_particles):
                            if mpm_env.loss.particle_mat[p] == mpm_env.loss.matching_mat:
                                u, v = from_xy_to_uv(mpm_env.loss.particle_x[f, p][0], mpm_env.loss.particle_x[f, p][1])
                                ti.atomic_max(height_grid[u, v], (mpm_env.loss.particle_x[f, p][2]))

                    height_grid.fill(0.0)
                    calculate_height_grid(mpm_env.simulator.cur_substep_local)

                    @ti.kernel
                    def compute_height_grid_masks(f: ti.i32):
                        for p in range(mpm_env.loss.n_particles):
                            if mpm_env.loss.particle_mat[p] == mpm_env.loss.matching_mat:
                                u, v = from_xy_to_uv(mpm_env.loss.particle_x[f, p][0], mpm_env.loss.particle_x[f, p][1])
                                if mpm_env.loss.particle_x[f, p][2] >= height_grid[u, v]:
                                    p_id[u * res + v] = p

                    p_id.fill(-1)
                    compute_height_grid_masks(mpm_env.simulator.cur_substep_local)

                    @ti.kernel
                    def compute_emd_distance_matrix(f: ti.i32):
                        for u in range(res * res):
                            for v in range(res * res):
                                p = p_id[u]
                                if mpm_env.loss.particle_mat[u] == mpm_env.loss.matching_mat:
                                    emd_distance_matrix[u, v] = compute_euclidean_distance(
                                        mpm_env.loss.particle_x[f, p],
                                        target_pcd_points[v])

                    emd_distance_matrix.fill(10000.0)
                    p_id_np = p_id.to_numpy()
                    compute_emd_distance_matrix(mpm_env.simulator.cur_substep_local)

                    @ti.func
                    def compute_euclidean_distance(a, b):
                        return ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

                    linear_assignment_failed = False
                    emd_ind_pairs = ti.Vector.field(2, dtype=ti.i32, shape=res * res, needs_grad=False)

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
                                linear_assignment_failed = True
                        if done:
                            emd_ind_pairs.from_numpy(indexes)

                    emd_ind_pairs.fill(-1)
                    compute_emd_distance_bijection()

                    emd_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)

                    @ti.kernel
                    def compute_emd_euclidean_distance(f: ti.i32):
                        for q in range(res * res):
                            w = emd_ind_pairs[q][0]
                            e = emd_ind_pairs[q][1]
                            p = p_id[w]
                            d = compute_euclidean_distance(target_pcd_points[e],
                                                           mpm_env.loss.particle_x[f, p])
                            emd_loss[None] += d

                    emd_loss.fill(0.0)
                    compute_emd_euclidean_distance(mpm_env.simulator.cur_substep_local)

                    target_pcd_heightmap = np.load(os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                                f'pcd_{motion_id}_cropped_norm_z_aligned_height_map-res{res}.npy'))
                    target_pcd_heightmap_ti = ti.field(dtype=ti.f32, shape=(res, res), needs_grad=False)
                    target_pcd_heightmap_ti.from_numpy(target_pcd_heightmap)
                    height_map_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

                    @ti.kernel
                    def compute_height_map_loss():
                        for q, e in target_pcd_heightmap_ti:
                            d = ti.sqrt((target_pcd_heightmap_ti[q, e] - height_map[q, e]) ** 2)
                            height_map_loss[None] += d

                    height_map_loss.fill(0.0)
                    compute_height_map_loss()

                    print(f'=====> EMD Loss with res {res}:', emd_loss[None])
                    print(f'=====> Height map loss with res {res}:', height_map_loss[None])

                    emd_losses[ll][i, j] = emd_loss[None]
                    hm_losses[ll][i, j] = height_map_loss[None]

                    ll += 1
                mpm_env.simulator.clear_ckpt()

        ll = 0
        for res in [10, 20, 30, 40, 50, 60]:
            np.save(os.path.join(result_path, f'emd_losses-res{res}-{param}.npy'), emd_losses[ll])
            np.save(os.path.join(result_path, f'hm_losses-res{res}-{param}.npy'), hm_losses[ll])
            ll += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="System identification for soil")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6, help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--soft-contact', dest='soft_contact', action='store_true', default=False, help='Use soft contact')
    parser.add_argument('--toi-contact', dest='toi_contact', action='store_true', default=False, help='Use time-of-impact contact')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='test')
    arguments = vars(parser.parse_args())
    main(arguments)
