import os
import open3d as o3d
import gym
import json
import logging
import argparse
import numpy as np
import taichi as ti
from drl_implementation.agent.continuous_action.sac_parameterised_action_goal_conditioned import GPASAC
from gym.spaces import Box, Discrete
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.planting_env import make_env
from doma.engine.configs.macros import DTYPE_NP
from doma.engine.loss_function.emd_loss_external import compute_emd_loss_external
cam_cfg = {  # same camera pose as the real-world setup
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
LINEAR_VELOCITY = 0.1  # m/s
ANGULAR_VELOCITY = 0.5  # rad/s


def skill_generation_func(skill_index, skill_params, dt):
    assert np.all(skill_params >= -1.0) and np.all(skill_params <= 1.0), 'RL skill params should be in [-1, 1]'
    # set the initial pose of the end effector
    trajectory = np.zeros(shape=(1000, 6), dtype=np.float32)
    finish_generation = False
    if skill_index == 0:
        # planar movement
        move_angle = skill_params[0]  # angle between the x-axis and the movement direction
        move_angle = np.pi * (move_angle + 1)  # map [-1, 1] to [0, 2pi]
        move_distance = skill_params[1]
        move_distance *= 0.05  # map [-1, 1] to [-0.05, 0.05]
        n_step = int(move_distance / (LINEAR_VELOCITY * dt))
        move_distance_x = move_distance * ti.cos(move_angle)
        delta_x = move_distance_x / n_step
        move_distance_y = move_distance * ti.sin(move_angle)
        delta_y = move_distance_y / n_step
        if 0 <= move_angle < np.pi / 2:
            x_sign = 1
            y_sign = 1
        elif np.pi / 2 <= move_angle < np.pi:
            x_sign = -1
            y_sign = 1
        elif np.pi <= move_angle < 3 * np.pi / 2:
            x_sign = -1
            y_sign = -1
        else:
            x_sign = 1
            y_sign = -1
        for i in range(n_step):
            trajectory[i][0] = delta_x * x_sign
            trajectory[i][1] = delta_y * y_sign
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
        if insert_angle < np.pi / 2:
            x_sign = 1
        else:
            x_sign = -1
        for i in range(n_step):
            trajectory[i][0] = delta_x * x_sign
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
        if pullout_angle < np.pi / 2:
            x_sign = 1
        else:
            x_sign = -1
        for i in range(n_step):
            trajectory[i][0] = delta_x * x_sign
            trajectory[i][2] = delta_z
        return trajectory[:n_step]
    elif skill_index == 3:
        # rotate about y and z axes
        rotate_angle_y = skill_params[0]
        rotate_angle_y = (np.pi / 2) * rotate_angle_y  # map [-1, 1] to [-pi/2, pi/2]
        rotate_angle_z = skill_params[1]
        rotate_angle_z = (np.pi / 2) * rotate_angle_z  # map [-1, 1] to [-pi/2, pi/2]
        n_step_y = int(rotate_angle_y / (ANGULAR_VELOCITY * dt))
        n_step_z = int(rotate_angle_z / (ANGULAR_VELOCITY * dt))
        n_step = max(n_step_y, n_step_z)
        delta_y = rotate_angle_y / n_step
        delta_z = rotate_angle_z / n_step
        for i in range(n_step):
            trajectory[i][4] = delta_y
            trajectory[i][5] = delta_z
    else:
        raise ValueError('Invalid skill index')


class HybridActionEnv(gym.Env):
    def __init__(self, mem_env, gym_env_config, seed=None, logger=None):
        if seed is not None:
            self.seed(seed)
        self.mpm_env = mem_env
        self.mpm_env_init_state = gym_env_config['mpm_env_init_state']

        self.step_count = 0
        self.render_skill = gym_env_config['render_skill']
        self.horizon = gym_env_config['horizon']
        self.obs_mode = gym_env_config['obs_mode']
        self.reward_scale = gym_env_config['reward_scale']
        self.discrete_action_space = Discrete(n=gym_env_config['n_discrete_action'])
        self.continuous_action_space = Box(low=gym_env_config['continuous_action_min'],
                                           high=gym_env_config['continuous_action_max'],
                                           shape=(gym_env_config['dim_continuous_action'],),
                                           dtype=DTYPE_NP)
        self.skill_generation_func = gym_env_config['skill_generation_func']

        self.logger = logger
        self.goal_conditioned_reward_function = self.mpm_env.loss.compute_emd_loss_external_func
        self.distance_threshold = 0.01

    def seed(self, seed=None):
        super(HybridActionEnv, self).seed(seed)

    def step(self, action):
        discrete_action = int(action[0])
        continuous_action = action[1:]
        skill_trajectory = self.skill_generation_func(discrete_action, continuous_action,
                                                      dt=self.mpm_env.simulator.dt_global)
        for i in range(len(skill_trajectory)):
            self.mpm_env.step(skill_trajectory[i])
            if self.render_skill:
                self.render(mode='human')

        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        loss_info = self.mpm_env.get_final_loss()
        reward = loss_info['emd_loss'].to_numpy() * self.reward_scale
        self.step_count += 1
        done = self.step_count >= self.horizon
        return {
            'observation': obs.copy(),
            'agent_state': agent_state,
            'desired_goal': self.mpm_env.loss.target_pcd_points_np,
            'achieved_goal': obs.copy()
        }, reward, done, loss_info

    def reset(self):
        self.mpm_env.set_state(self.mpm_env_init_state, grad_enabled=False)
        self.step_count = 0
        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        return {
            'observation': obs.copy(),
            'agent_state': agent_state,
            'desired_goal': self.mpm_env.loss.target_pcd_points_np,
            'achieved_goal': obs.copy()
        }

    def render(self, mode='human'):
        return self.mpm_env.render(mode=mode)


class FakeEnv(gym.Env):
    def __init__(self, gym_env_config, seed=None, logger=None):
        if seed is not None:
            self.seed(seed)
        self.pcd_file_path = gym_env_config['pcd_file_path']
        self.pcd_np = np.asarray(o3d.io.read_point_cloud(self.pcd_file_path).voxel_down_sample(voxel_size=0.005).points,
                                 dtype=DTYPE_NP)

        self.step_count = 0
        self.horizon = gym_env_config['horizon']
        self.obs_mode = gym_env_config['obs_mode']
        self.reward_scale = gym_env_config['reward_scale']
        self.discrete_action_space = Discrete(n=gym_env_config['n_discrete_action'])
        self.continuous_action_space = Box(low=gym_env_config['continuous_action_min'],
                                           high=gym_env_config['continuous_action_max'],
                                           shape=(gym_env_config['dim_continuous_action'],),
                                           dtype=DTYPE_NP)
        self.skill_generation_func = gym_env_config['skill_generation_func']

        self.logger = logger
        self.goal_conditioned_reward_function = compute_emd_loss_external
        self.distance_threshold = 0.01

    def seed(self, seed=None):
        super(FakeEnv, self).seed(seed)

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.horizon
        reward = self.goal_conditioned_reward_function(self.pcd_np, self.pcd_np)
        return {
            'observation': self.pcd_np.copy(),
            'agent_state': np.ones(shape=(6,), dtype=np.float32),
            'desired_goal': self.pcd_np.copy(),
            'achieved_goal': self.pcd_np.copy()
        }, reward, done, {}

    def reset(self):
        self.step_count = 0
        return {
            'observation': self.pcd_np.copy(),
            'agent_state': np.ones(shape=(6,), dtype=np.float32),
            'desired_goal': self.pcd_np.copy(),
            'achieved_goal': self.pcd_np.copy()
        }

    def render(self, mode='human'):
        pass


def main(arguments):
    seed = arguments['seed']
    if arguments['demonstrate_skills']:
        log_dir = os.path.join(script_path, '..', 'log-gpasac-ds', f'seed-{seed}')
    elif arguments['planned_skills']:
        log_dir = os.path.join(script_path, '..', 'log-gpasac-ps', f'seed-{seed}')
    else:
        log_dir = os.path.join(script_path, '..', 'log-gpasac', f'seed-{seed}')
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
        'p_density': arguments['ptcl_density'],
        'horizon': 500,
        'dt_global': 0.01,
        'n_substeps': 50,
        'agent_init_pos': (0.4, 0.2, 0.2),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'target_pcds', 'pcd_2_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0, 0, 0],
        'down_sample_voxel_size': 0.005,
    }

    ti.reset()
    ti.init(arch=backend,
            # device_memory_GB=4,
            default_fp=ti.f32, fast_math=True, random_seed=seed)
    # env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)
    gym_env_config = {
        # 'mpm_env_init_state': init_state['state'],
        'pcd_file_path': os.path.join(script_path, '..', 'data', 'target_pcds', 'pcd_2_cropped_norm_z_aligned.ply'),
        'render_skill': False,
        'horizon': arguments['n_skills'],
        'obs_mode': 'point_cloud',
        'reward_scale': -1.0,
        'n_discrete_action': 3,
        'dim_continuous_action': 6,
        'continuous_action_max': 1.0,
        'continuous_action_min': -1.0,
        'skill_generation_func': skill_generation_func,
    }
    # gym_env = HybridActionEnv(mpm_env, gym_env_config, seed=arguments['seed'], logger=logging)
    gym_env = FakeEnv(gym_env_config, seed=seed, logger=logging)

    with open(os.path.join(script_path, '..', 'data', 'rl_agent_config.json'), 'rb') as f_ac:
        rl_agent_config = json.load(f_ac)
    rl_agent_config['cuda_device_id'] = arguments['torch_cuda_device_id']
    rl_agent_config['batch_size'] = 8
    rl_agent_config['optimization_steps'] = arguments['n_skills']
    rl_agent_config['demonstrate_skills'] = arguments['demonstrate_skills']
    rl_agent_config['demonstrate_percentage'] = arguments['demonstrate_percentage']
    rl_agent_config['planned_skills'] = arguments['planned_skills']
    rl_agent_config['skill_plan'] = [0, 1, 0, 2]  # move, insert, move, pullout
    assert len(rl_agent_config['skill_plan']) == arguments['n_skills'], 'The length of skill plan should be equal to n_skills'
    with open(os.path.join(log_dir, 'rl_agent_config.json'), 'w') as f_ac:
        json.dump(rl_agent_config, f_ac)

    gpasac_agent = GPASAC(rl_agent_config, gym_env, path=log_dir, seed=seed)
    gpasac_agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='seed')
    parser.add_argument('--backend', dest='backend', type=str, default='cuda', help='backend')
    parser.add_argument('--ptcl-d', dest='ptcl_density', type=float, default=3e6, help='particle density')
    parser.add_argument('--n-skills', dest='n_skills', type=int, default=4, help='number of skills')
    parser.add_argument('--demo-skills', dest='demonstrate_skills', action='store_true', default=False, help='demonstrate the order of skills')
    parser.add_argument('--demo-frequency', dest='demonstrate_percentage', type=float, default=0.5, help='percentage of demonstration episodes')
    parser.add_argument('--planned-skills', dest='planned_skills', action='store_true', default=True, help='always ues planed order of skills')
    parser.add_argument('--t-cuda-id', dest='torch_cuda_device_id', type=int, default=0, help='cuda device id')
    args = vars(parser.parse_args())
    main(args)
