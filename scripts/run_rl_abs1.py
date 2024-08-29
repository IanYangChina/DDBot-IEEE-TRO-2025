import os
import open3d as o3d
import json
import logging
import argparse
import numpy as np
import taichi as ti
from drl_implementation.agent.continuous_action.sac_parameterised_action_goal_conditioned import GPASAC
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.planting_env import make_env
from doma.engine.configs.macros import DTYPE_NP, SAND
from doma.engine.utils.misc import set_parameters
from doma.envs.wrappers import SingleSkillEnv, HybridActionEnv, FakeEnv
cam_cfg = {  # same camera pose as the real-world setup
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


def abstraction_one_skills(skill_index, skill_params, dt):
    assert np.all(skill_params >= -1.0) and np.all(skill_params <= 1.0), 'RL skill params should be in [-1, 1]'
    # set the initial pose of the end effector
    trajectory = np.zeros(shape=(1000, 6), dtype=np.float32)
    if skill_index == 0:
        # planar movement
        move_angle = skill_params[0]  # angle between the x-axis and the movement direction
        move_angle = np.pi * (move_angle + 1)  # map [-1, 1] to [0, 2pi]
        move_distance = skill_params[1]
        move_distance = 0.1 * (move_distance + 1)  # map [-1, 1] to [0, 0.2]
        n_step = int(move_distance / (LINEAR_VELOCITY * dt))
        move_distance_x = move_distance * ti.cos(move_angle)
        delta_x = move_distance_x / n_step
        move_distance_y = move_distance * ti.sin(move_angle)
        delta_y = move_distance_y / n_step
        for i in range(n_step):
            trajectory[i][0] = delta_x
            trajectory[i][1] = delta_y
        return trajectory[:n_step]
    elif skill_index == 1:
        # insertion
        insert_angle = skill_params[0]  # angle between the x-axis and the insertion direction
        insert_angle = np.pi * (insert_angle + 1) / 2  # map [-1, 1] to [0, pi]
        insert_distance = skill_params[1]
        insert_distance = 0.05 * (insert_distance + 1) / 2  # map [-1, 1] to [0, 0.05]
        n_step = int(insert_distance / (LINEAR_VELOCITY * dt))
        insert_distance_x = insert_distance * ti.cos(insert_angle)
        delta_x = insert_distance_x / n_step
        insert_distance_z = insert_distance * ti.sin(insert_angle)
        delta_z = insert_distance_z / n_step
        for i in range(n_step):
            trajectory[i][0] = delta_x
            trajectory[i][2] = delta_z * -1
        return trajectory[:n_step]
    elif skill_index == 2:
        # pullout
        pullout_angle = skill_params[0]
        pullout_angle = np.pi * (pullout_angle + 1) / 2  # map [-1, 1] to [0, pi]
        pullout_distance = skill_params[1]
        pullout_distance = 0.1 * (pullout_distance + 1) / 2  # map [-1, 1] to [0, 0.1]
        n_step = int(pullout_distance / (LINEAR_VELOCITY * dt))
        pullout_distance_x = pullout_distance * ti.cos(pullout_angle)
        delta_x = pullout_distance_x / n_step
        pullout_distance_z = pullout_distance * ti.sin(pullout_angle)
        delta_z = pullout_distance_z / n_step
        for i in range(n_step):
            trajectory[i][0] = delta_x
            trajectory[i][2] = delta_z
        return trajectory[:n_step]
    elif skill_index == 3:
        # rotate about y and z axes
        rotate_angle_x = skill_params[0]
        rotate_angle_x = (np.pi / 2) * rotate_angle_x  # map [-1, 1] to [-pi/2, pi/2]
        rotate_angle_z = skill_params[1]
        rotate_angle_z = (np.pi / 2) * rotate_angle_z  # map [-1, 1] to [-pi/2, pi/2]
        n_step_x = np.abs(int(rotate_angle_x / (ANGULAR_VELOCITY * dt)))
        n_step_z = np.abs(int(rotate_angle_z / (ANGULAR_VELOCITY * dt)))
        n_step = max(n_step_x, n_step_z)
        delta_x = rotate_angle_x / n_step
        delta_z = rotate_angle_z / n_step
        for i in range(n_step):
            trajectory[i][3] = delta_x
            trajectory[i][5] = delta_z
        return trajectory[:n_step]
    else:
        raise ValueError('Invalid skill index')


def main(arguments):
    seed = arguments['seed']
    task_id = arguments['task_id']
    n_skill = arguments['n_skills']
    if arguments['demonstrate_skills']:
        log_dir = os.path.join(script_path, '..', 'log-abs1-gpasac-ds', f'task{task_id}-{n_skill}skills', f'seed-{seed}')
    elif arguments['planned_skills']:
        log_dir = os.path.join(script_path, '..', 'log-abs1-gpasac-ps', f'task{task_id}-{n_skill}skills', f'seed-{seed}')
    else:
        log_dir = os.path.join(script_path, '..', 'log-abs1-gpasac', f'task{task_id}-{n_skill}skills', f'seed-{seed}')
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
        'p_density': arguments['ptcl_density'],
        'horizon': 10000,
        'dt_global': 0.01,
        'n_substeps': 20,
        'grid_scale': 1.0,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'task_target_pcds',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.007,
    }

    ti.reset()
    ti.init(arch=backend, device_memory_GB=arguments['cuda_GB'],
            default_fp=ti.f32, fast_math=True, random_seed=seed)
    env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)
    set_parameters(mpm_env, SAND, e=4e5, nu=0.2, rho=1800., sand_friction_angle=45.,
                   manipulator_friction=0.05, container_friction=0.5)
    gym_env_config = {
        # 'mpm_env_init_state': init_state['state'],
        'pcd_file_path': os.path.join(script_path, '..', 'data', 'task_target_pcds',
                                      f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'render_skill': arguments['env_test'],
        'horizon': n_skill,
        'obs_mode': 'point_cloud',
        'reward_scale': -1.0,
        'n_discrete_action': 3,
        'dim_continuous_action': 6,
        'continuous_action_max': 1.0,
        'continuous_action_min': -1.0,
        'skill_generation_func': abstraction_one_skills
    }
    gym_env = HybridActionEnv(mpm_env, gym_env_config, seed=arguments['seed'], logger=logging)
    # gym_env = FakeEnv(gym_env_config, seed=seed, logger=logging)

    if arguments['env_test']:
        skill_plan = [0, 1, 0, 2]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        for n in range(3):
            gym_env.reset()
            gym_env.render(mode='human')
            done = False
            while not done:
                input('Press Enter to continue...')
                if arguments['abstraction_level'] == 1:
                    a0 = skill_plan[gym_env.step_count]
                    a1 = gym_env.continuous_action_space.sample()
                    obs, reward, done, info = gym_env.step([a0, a1[0], a1[1]])
                else:
                    a1 = float(input('Enter the skill param 1: '))
                    a2 = float(input('Enter the skill param 2: '))
                    a3 = float(input('Enter the skill param 3: '))
                    a4 = float(input('Enter the skill param 4: '))
                    a5 = float(input('Enter the skill param 5: '))
                    # a = gym_env.continuous_action_space.sample()
                    a = np.asarray([a1, a2, a3, a4, a5], dtype=DTYPE_NP)
                    obs, reward, done, info = gym_env.step(a)

                print(reward)
                cloud_array = obs['observation']
                obj_vec = o3d.utility.Vector3dVector(cloud_array)
                obj_pcd = o3d.geometry.PointCloud(obj_vec)
                cloud_array_2 = obs['desired_goal']
                cloud_array_2[:, 0] += 0.3
                obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2)
                o3d.visualization.draw_geometries([frame, obj_pcd, obj_pcd_2], width=800, height=600)

                gym_env.render(mode='human')
    else:
        with open(os.path.join(script_path, '..', 'data', 'rl_agent_config.json'), 'rb') as f_ac:
            rl_agent_config = json.load(f_ac)
        rl_agent_config['cuda_device_id'] = arguments['torch_cuda_device_id']
        rl_agent_config['batch_size'] = 24
        rl_agent_config['optimization_steps'] = n_skill
        rl_agent_config['demonstrate_skills'] = arguments['demonstrate_skills']
        rl_agent_config['demonstrate_percentage'] = arguments['demonstrate_percentage']
        rl_agent_config['planned_skills'] = arguments['planned_skills']
        rl_agent_config['skill_plan'] = [0, 1, 0, 2]  # move, insert, move, pullout
        rl_agent_config['hindsight'] = True
        rl_agent_config['her_sampling_strategy'] = 'future'
        rl_agent_config['num_sampled_goal'] = 2
        assert len(rl_agent_config['skill_plan']) == n_skill, 'The length of skill plan should be equal to n_skills'
        with open(os.path.join(log_dir, 'rl_agent_config.json'), 'w') as f_ac:
            json.dump(rl_agent_config, f_ac)

        gpasac_agent = GPASAC(rl_agent_config, gym_env, path=log_dir, seed=seed)
        gpasac_agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='seed')
    parser.add_argument('--backend', dest='backend', type=str, default='cuda', help='backend')
    parser.add_argument('--task-id', dest='task_id', type=int, default=0, help='task id')
    parser.add_argument('--e-test', dest='env_test', action='store_true', default=False, help='testing gym env')
    parser.add_argument('--ptcl-d', dest='ptcl_density', type=float, default=1e7, help='particle density')
    parser.add_argument('--n-skills', dest='n_skills', type=int, default=4, help='number of skills')
    parser.add_argument('--demo-skills', dest='demonstrate_skills', action='store_true', default=False, help='demonstrate the order of skills')
    parser.add_argument('--demo-frequency', dest='demonstrate_percentage', type=float, default=0.5, help='percentage of demonstration episodes')
    parser.add_argument('--planned-skills', dest='planned_skills', action='store_true', default=False, help='always ues planed order of skills')
    parser.add_argument('--t-cuda-id', dest='torch_cuda_device_id', type=int, default=0, help='cuda device id')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=2, type=int, help='preallocated GPU memory in GB')
    args = vars(parser.parse_args())
    main(args)
