import os
import argparse
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import taichi as ti
from PIL import Image
import imageio
from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import DTYPE_NP, SAND, COLOR
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
LINEAR_VELOCITY = 0.2  # m/s
ANGULAR_VELOCITY = np.pi / 4  # rad/s
DT_GLOBAL = 0.01  # sec


def abstraction_two_skill(skill_params, dt):
    assert np.all(skill_params >= -1.0) and np.all(skill_params <= 1.0), 'RL skill params should be in [-1, 1]'
    trajectory = np.zeros(shape=(1000, 6), dtype=np.float32)
    move_distance = skill_params[0] * 0.12  # map [-1, 1] to [-0.12, 0.12]
    rotate_x = skill_params[1] * (np.pi / 3)  # map [-1, 1] to [-pi/3, pi/3]

    n_step_move = np.abs(int(move_distance / (LINEAR_VELOCITY * dt)))
    n_step_rotate = np.abs((int(rotate_x / (ANGULAR_VELOCITY * dt))))
    n_step = max(n_step_move, n_step_rotate)
    if n_step > 0:
        move_delta_x = move_distance / n_step
        for i in range(n_step):
            trajectory[i][0] = move_delta_x
        rotate_delta_x = rotate_x / n_step
        for i in range(n_step):
            trajectory[i][3] = rotate_delta_x

    insert_angle = rotate_x + np.pi / 2
    insert_distance = (skill_params[2] + 1) / 2 * 0.06  # map [-1, 1] to [0, 0.06]
    n_step_insert = int(insert_distance / (LINEAR_VELOCITY * dt))
    if n_step_insert > 0:
        insert_distance_x = insert_distance * np.cos(insert_angle)
        insert_distance_z = insert_distance * np.sin(insert_angle)
        insert_delta_x = insert_distance_x / n_step_insert
        insert_delta_z = insert_distance_z / n_step_insert
        for i in range(n_step, n_step + n_step_insert):
            trajectory[i][0] = insert_delta_x
            trajectory[i][2] = -insert_delta_z

    push_angle = (skill_params[3] + 3) * np.pi / 3  # map [-1, 1] to [2*pi/3, 4*pi/3]
    push_distance = (skill_params[4] + 1) * 0.1 + 0.04  # map [-1, 1] to [0.04, 0.24]
    n_step_push = int(push_distance / (LINEAR_VELOCITY * dt))
    if n_step_push > 0:
        push_distance_x = push_distance * np.cos(push_angle)
        push_distance_z = push_distance * np.sin(push_angle)
        push_delta_x = push_distance_x / n_step_push
        push_delta_z = push_distance_z / n_step_push
        for i in range(n_step + n_step_insert, n_step + n_step_insert + n_step_push):
            trajectory[i][0] = push_delta_x
            trajectory[i][2] = push_delta_z

    rotate_x_back = -rotate_x
    n_step_rotate_back = n_step_rotate
    move_up_distance = 0.1
    n_step_move_up = int(move_up_distance / (LINEAR_VELOCITY * dt))
    n_step_return = max(n_step_rotate_back, n_step_move_up)
    if n_step_return > 0:
        rotate_delta_x_back = rotate_x_back / n_step_return
        move_up_delta_z = move_up_distance / n_step_return
        for i in range(n_step + n_step_insert + n_step_push, n_step + n_step_insert + n_step_push + n_step_return):
            trajectory[i][3] = rotate_delta_x_back
            trajectory[i][2] = move_up_delta_z

    return trajectory[:n_step + n_step_insert + n_step_push + n_step_return, :]


def main(args):
    if args['sand']:
        mat = '_sand'
    else:
        mat = ''
    saving_folder = os.path.join(script_path, '..', 'render_test')
    os.makedirs(saving_folder, exist_ok=True)
    if args['save_img']:
        os.makedirs(os.path.join(saving_folder, f'imgs{mat}'), exist_ok=True)
    sys_id_motion = 1
    task_id = args['task_id']
    dt_sim = 0.01

    if args['sys_id']:
        trajectory = np.load(os.path.join(script_path, '..', 'data',
                                          'moveit_trajectories',
                                          f'sys_id_sim_{sys_id_motion}_pos-dt_{dt_sim}.npy'))
        target_pcd_path = os.path.join(script_path, '..', 'data', f'sys_id_target_pcds{mat}',
                                       f'pcd_{sys_id_motion}_cropped_norm_z_aligned.ply')
    else:
        assert task_id >= 0, 'Task ID should be provided'
        target_pcd_path = os.path.join(script_path, '..', 'data', f'task_target_pcds{mat}',
                                       f'pcd_{task_id}_cropped_norm_z_aligned.ply')
        seed = 4
        if args['skill']:
            if args['view_demon']:
                skill_params = np.asarray([1.0, 0.2, 0.8, 0.0, -0.5]).astype(DTYPE_NP)
                target_pcd = o3d.io.read_point_cloud(target_pcd_path[:-4] + '_res40.ply')
                target_pcd_points = np.asarray(target_pcd.points) + np.array([0.2, 0.2, 0])
                z_min_idx = np.argmin(target_pcd_points[:, 2])
                x_demo = target_pcd_points[z_min_idx, 0] + 0.02
                skill_params[0] = np.clip((x_demo - 0.2) / 0.12, -1.0, 1.0)
            else:
                with open(os.path.join(script_path, '..', f'log-abs2{mat}',
                                       f'd5e6-task-{task_id}-ls-demo-lr0.03',
                                       'best_loss.json'), 'r') as f:
                    skill_params_json = json.load(f)['Parameters']
                    skill_params = np.array([
                        skill_params_json['skill_params_0'][0],
                        skill_params_json['skill_params_1'][0],
                        skill_params_json['skill_params_2'][0],
                        skill_params_json['skill_params_3'][0],
                        skill_params_json['skill_params_4'][0]
                    ])
                print('Loaded skill parameters: \n\"{p1: ' +
                      f'{skill_params[0]}, ' +
                      f'p2: {skill_params[1]}, ' +
                      f'p2: {skill_params[2]}, ' +
                      f'p2: {skill_params[3]}, ' +
                      f'p2: {skill_params[4]}' + '}\"')

            trajectory = abstraction_two_skill(skill_params, dt_sim)
        else:
            trajectory = np.load(os.path.join(script_path, '..', f'log-abs0{mat}',
                                              f'd5e6-task-{task_id}-ls-demo-lr0.001',
                                              f'seed-{seed}', 'final_trajectory.npy'))

    env_cfg = {
        'p_density': float(args['ptcl_density']),
        'material_id': SAND,
        'horizon': trajectory.shape[0],
        'dt_global': dt_sim,
        'grid_scale': 1.0,
        'n_substeps': 20,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'use_height_map_loss': False,
        'target_pcd_path': target_pcd_path,
        'target_pcd_offset': [0.2, 0.2, 0],
        'height_grid_res': 40,
    }

    cam_cfg = {
        'pos': (0.2, 0.58, 0.51),
        'lookat': (0.2, 0.18, 0.03),
        'euler': (180 + np.rad2deg(np.arctan(1.0 / (0.9 - 0.03))), 0, 180),
        'focal_length': 0.3,
        'fov': 30,
        'lights': [{'pos': (1.2, 0.2, 1.0), 'color': (0.95, 0.95, 0.95)},
                   {'pos': (0.2, 0.5, 1.0), 'color': (0.95, 0.95, 0.95)},
                   {'pos': (0.2, 0.2, 1.0), 'color': (0.95, 0.95, 0.95)}],
        'particle_radius': 0.001,
        'res': (800, 600),
        'pcd_gen_res': 150
    }

    ti.reset()
    ti.init(arch=ti.cuda, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=1)
    env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg)

    with open(os.path.join(script_path, '..', f'log-sys_id{mat}',
                           'd5e6-hm-gclip-ls-man-init-res40',
                           'best_params.json'), 'r') as f:
        best_params = json.load(f)['Parameters']

    set_parameters(mpm_env, SAND,
                   e=best_params['E'],
                   nu=best_params['nu'],
                   rho=best_params['rho'],
                   sand_friction_angle=best_params['sand_angle'],)

    mpm_env.set_state(init_state['state'], grad_enabled=False)
    frames_for_gifs = []
    if args['render']:
        mpm_env.render(mode='human')
    if args['save_video']:
        img = mpm_env.render(mode='rgb_array')
        frames_for_gifs.append(img)
    # input('Press Enter to continue...')
    for i in range(mpm_env.horizon):
        mpm_env.step(trajectory[i])
        if args['render']:
            mpm_env.render(mode='human')
        if args['save_img']:
            img = mpm_env.render(mode='rgb_array')
            if i % 40 == 0:
                Image.fromarray(img).save(os.path.join(saving_folder, f'imgs{mat}',
                                                       f'img_{i}.png'))
        if args['save_video']:
            if i % 5 == 0:
                img = mpm_env.render(mode='rgb_array')
                frames_for_gifs.append(img)
    loss_info = mpm_env.get_final_loss()
    print('===> Validation loss:', loss_info['height_map_loss'] / (40 * 40) + loss_info['emd_loss'] / (40 * 40))

    if args['render_hm'] or args['save_hm']:
        fig, ax = plt.subplots(2, 1, figsize=(3, 6))
        plt.subplots_adjust(wspace=0, hspace=0.2)

        ax[0].imshow(np.rot90(mpm_env.loss.height_map.to_numpy()),
                     vmin=0.002, vmax=0.09)
        ax[0].axis('off')
        # ax[0].set_title('Height map')
        ax[1].imshow(np.rot90(mpm_env.loss.height_map_pcd_target.to_numpy()),
                     vmin=0.002, vmax=0.09)
        ax[1].axis('off')
        # ax[1].set_title('Target height map')
        if args['save_hm']:
            plt.savefig(os.path.join(saving_folder, f'height_map{mat}.png'))
        if args['render_hm']:
            plt.show()

    if args['render_pcd']:
        x = mpm_env.loss.surface_particles.to_numpy()
        x[:, 1] += 0.3
        sim_particles = o3d.utility.Vector3dVector(x)
        sim_particles = o3d.geometry.PointCloud(sim_particles)
        # o3d.visualization.draw_geometries([sim_particles],
        #                                   window_name='Simulated point cloud',
        #                                   width=800, height=600)

        x = mpm_env.loss.target_pcd_points_np
        target_particles = o3d.utility.Vector3dVector(x)
        target_particles = o3d.geometry.PointCloud(target_particles)
        o3d.visualization.draw_geometries([target_particles, sim_particles],
                                            window_name='Target point cloud',
                                            width=800, height=600)

    if args['save_video']:
        with imageio.get_writer(os.path.join(saving_folder, f'video{mat}.gif'), mode='I') as writer:
            for i in range(len(frames_for_gifs)):
                writer.append_data(frames_for_gifs[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run system identification simulation')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6, help='Particle density')
    parser.add_argument('--cuda_GB', dest='cuda_GB', type=float, default=5, help='CUDA memory in GB')
    parser.add_argument('--r', dest='render', default=False, action='store_true', help='Render the simulation')
    parser.add_argument('--rp', dest='render_pcd', default=False, action='store_true', help='Render the point cloud')
    parser.add_argument('--rhm', dest='render_hm', default=False, action='store_true', help='Render the height map')
    parser.add_argument('--shm', dest='save_hm', default=False, action='store_true', help='Render the height map')
    parser.add_argument('--simg', dest='save_img', default=False, action='store_true', help='Save video')
    parser.add_argument('--sv', dest='save_video', default=False, action='store_true', help='Save video')
    parser.add_argument('--sysid', dest='sys_id', default=False, action='store_true', help='Run system identification')
    parser.add_argument('--skill', dest='skill', default=False, action='store_true', help='Run skill')
    parser.add_argument('--task-id', dest='task_id', type=int, default=-1, help='Run task')
    parser.add_argument('--sand', dest='sand', default=False, action='store_true', help='Use sand material')
    parser.add_argument('--view-demon', dest='view_demon', action='store_true', default=False, help='View demonstration')
    arguments = vars(parser.parse_args())
    main(arguments)
