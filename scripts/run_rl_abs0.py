import os
import open3d as o3d
import json
import logging
import argparse
import numpy as np
import taichi as ti
from drl_implementation.agent.continuous_action.sac_pointnet import PointnetSAC
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.engine.configs.macros import DTYPE_NP, SAND
from doma.envs.wrappers import TrajectoryEnv, FakeEnv
cam_cfg = {
    'pos': (0.2, 0.57, 0.6),
    'lookat': (0.2, 0.2, 0.03),
    'euler': (180+np.rad2deg(np.arctan(1.0/(0.9-0.03))), 0, 180),
    'focal_length': 0.3,
    'fov': 60,
    'lights': [{'pos': (1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
    'particle_radius': 0.001,
    'res': (800, 800),
    'pcd_gen_res': 60
}

LINEAR_VELOCITY = 0.2  # m/s
ANGULAR_VELOCITY = np.pi / 4  # rad/s
DT_GLOBAL = 0.01  # sec


def abstraction_two_skill(skill_params, dt):
    assert np.all(skill_params >= -1.0) and np.all(skill_params <= 1.0), 'RL skill params should be in [-1, 1]'
    trajectory = np.zeros(shape=(1000, 6), dtype=np.float32)
    move_distance = skill_params[0] * 0.12  # map [-1, 1] to [-0.12, 0.12]
    rotate_x = skill_params[1] * (np.pi / 2)  # map [-1, 1] to [-pi/2, pi/2]

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

    push_angle = (skill_params[3] + 1) * np.pi / 2  # map [-1, 1] to [0, pi]
    push_distance = (skill_params[4] + 1) * 0.1  # map [-1, 1] to [0, 0.2]
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
            trajectory[i][5] = move_up_delta_z

    return trajectory[:n_step + n_step_insert + n_step_push + n_step_return, :]


def main(arguments):
    seed = arguments['seed']
    task_id = arguments['task_id']
    ptcl_d = arguments['ptcl_density']
    if arguments['use_demo']:
        log_dir = os.path.join(script_path, '..', 'log-abs0-sac', f'd{ptcl_d}-task-{task_id}-demo', f'seed-{seed}')
    else:
        log_dir = os.path.join(script_path, '..', 'log-abs0-sac', f'd{ptcl_d}-task-{task_id}', f'seed-{seed}')
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, 'optimisation.log')
    if os.path.isfile(log_file_name):
        filemode = "a"
    else:
        filemode = "w"
    logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                        filename=log_file_name,
                        format="%(asctime)s %(levelname)s %(message)s")

    if arguments['backend'] == 'opengl':
        backend = ti.opengl
    elif arguments['backend'] == 'cuda':
        backend = ti.cuda
    elif arguments['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    env_cfg = {
        'material_id': SAND,
        'p_density': float(arguments['ptcl_density']),
        'horizon': 600,
        'dt_global': DT_GLOBAL,
        'n_substeps': 20,
        'grid_scale': 1.0,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
        'best_params': None
    }
    loss_cfg = {
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'task_target_pcds',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.007,
    }
    ti_cfg = {
        'arch': backend,
        'device_memory_GB': arguments['cuda_GB'],
        'fast_math': True,
        'random_seed': seed
    }
    ti_env_cfg = {
        'env_cfg': env_cfg,
        'loss_cfg': loss_cfg,
        'ti_cfg': ti_cfg,
        'cam_cfg': cam_cfg
    }

    with open(os.path.join(script_path, '..', 'log-sys_id', 'best_params.json')) as f:
        best_params = json.load(f)[arguments['ptcl_density']]["Parameters"]
    env_cfg['best_params'] = best_params
    gym_env_config = {
        'pcd_file_path': os.path.join(script_path, '..', 'data', 'task_target_pcds',
                                      f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'render_skill': arguments['env_test'],
        'horizon': 300,
        'obs_mode': 'point_cloud',
        'reward_scale': -1.0,
        'action_dim': 6,
        'action_max': 1.0,
        'action_min': -1.0,
        'skill_generation_func': abstraction_two_skill
    }
    gym_env = TrajectoryEnv(ti_env_cfg, gym_env_config, seed=arguments['seed'], logger=logging)
    # gym_env = FakeEnv(gym_env_config, onestep=True, seed=seed, logger=logging)
    if arguments['env_test']:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        for n in range(5):
            gym_env.reset()
            gym_env.render(mode='human')
            done = False
            while not done:
                a = gym_env.action_space.sample().astype(DTYPE_NP)
                obs, reward, done, info = gym_env.step(a)

                print(reward)
                cloud_array = obs['observation']
                obj_vec = o3d.utility.Vector3dVector(cloud_array)
                obj_pcd = o3d.geometry.PointCloud(obj_vec)
                cloud_array_2 = obs['desired_goal']
                cloud_array_2[:, 0] += 0.3
                obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2)
                print(obj_pcd, obj_pcd_2)
                o3d.visualization.draw_geometries([frame, obj_pcd, obj_pcd_2], width=800, height=600)

                gym_env.render(mode='human')
    else:
        with open(os.path.join(script_path, '..', 'data', 'rl_agent_config.json'), 'rb') as f_ac:
            rl_agent_config = json.load(f_ac)
        rl_agent_config['cuda_device_id'] = arguments['torch_cuda_device_id']
        rl_agent_config['batch_size'] = 24
        rl_agent_config['optimization_steps'] = 1
        rl_agent_config['hindsight'] = True
        rl_agent_config['sampling_strategy'] = 'future'
        rl_agent_config['use_demonstrations'] = arguments['use_demo']
        rl_agent_config['demonstrate_percentage'] = 0.50
        demon_tr = np.zeros(shape=(300, 6), dtype=np.float32)
        demon_tr_ = abstraction_two_skill([1.0, 0.3, 0.8, 1.0, 0.3], DT_GLOBAL)
        demon_tr[:len(demon_tr_), :] = demon_tr_
        rl_agent_config['demonstration_action'] = demon_tr
        with open(os.path.join(log_dir, 'rl_agent_config.json'), 'w') as f_ac:
            json.dump(rl_agent_config, f_ac)

        agent = PointnetSAC(rl_agent_config, gym_env, logging=logging, path=log_dir, seed=seed)
        agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='seed')
    parser.add_argument('--backend', dest='backend', type=str, default='cuda', help='backend')
    parser.add_argument('--task-id', dest='task_id', type=int, default=0, help='task id')
    parser.add_argument('--demo', dest='use_demo', action='store_true', default=False, help='use demonstrations')
    parser.add_argument('--e-test', dest='env_test', action='store_true', default=False, help='testing gym env')
    parser.add_argument('--ptcl-d', dest='ptcl_density', type=str, default="5e6", help='particle density')
    parser.add_argument('--t-cuda-id', dest='torch_cuda_device_id', type=int, default=0, help='cuda device id')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=4, type=int, help='preallocated GPU memory in GB')

    args = vars(parser.parse_args())
    main(args)
