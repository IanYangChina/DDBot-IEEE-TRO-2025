import gym
import os
from gym.spaces import Box
from doma.engine.base_envs.doma_env import DomaEnv
from yacs.config import CfgNode
from doma.engine.configs.macros import *
from doma.engine.loss_function.pcd_emd_hm_loss import PointCloudEMDHMLosses
from doma.engine.configs import agent_cfg_dir
from doma.engine.utils.misc import set_parameters
from doma.engine.loss_function.emd_loss_external import compute_emd_loss_external


class PlantingEnvV1(DomaEnv):
    def __init__(self, seed=None, horizon=500, horizon_action=500, dt_global=0.001, n_substeps=10, grid_scale=1,
                 n_obs_ptcls_per_body=200, material_id=SAND, collide_type='rigid', grad_op='clip',
                 has_loss=True, loss_cfg=None,
                 agent_cfg_file='rectangle_eef.yaml', agent_init_pos=None, agent_init_euler=None,
                 render_agent=True, camera_cfg=None,
                 action_min=-1.0, action_max=1.0, problem_dim=3, ptcl_density=1e6,
                 debug_grad=False, logger=None):
        self.material_id = material_id
        self.agent_cfg_file_path = os.path.join(agent_cfg_dir, agent_cfg_file)
        if has_loss:
            self.loss_cfg = loss_cfg
            assert self.loss_cfg is not None
        self.agent_init_pos = agent_init_pos
        self.agent_init_euler = agent_init_euler
        self.render_agent = render_agent
        self.camera_cfg = camera_cfg
        self.logger = logger
        super(PlantingEnvV1, self).__init__(horizon=horizon, horizon_action=horizon_action,
                                            dt_global=dt_global, n_substeps=n_substeps, grid_scale=grid_scale,
                                            n_obs_ptcls_per_body=n_obs_ptcls_per_body,
                                            collide_type=collide_type, grad_op=grad_op,
                                            has_loss=has_loss, seed=seed,
                                            action_min=action_min, action_max=action_max,
                                            problem_dim=problem_dim, ptcl_density=ptcl_density,
                                            debug_grad=debug_grad, logger=self.logger)

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(self.agent_cfg_file_path)
        # manipulator init pos
        if self.agent_init_pos is not None:
            agent_cfg.effectors[0]['params']['init_pos'] = self.agent_init_pos
        if self.agent_init_euler is not None:
            agent_cfg.effectors[0]['params']['init_euler'] = self.agent_init_euler
        self.mpm_env.setup_agent(agent_cfg)
        self.agent = self.mpm_env.agent
        if self.logger is not None:
            self.agent.logger = self.logger

    def setup_statics(self):
        self.mpm_env.add_static(
            file='SoilBox.obj',
            pos=(0.2, 0.2, 0.015),
            euler=(0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            material=CONTAINER,
            has_dynamics=False,
            sdf_res=150,
        )

    def setup_bodies(self):
        self.mpm_env.add_body(
            type='cube',
            lower=(0.06, 0.06, 0.015),
            upper=(0.34, 0.34, 0.085),  # we have 0.07m for the soil
            material=self.material_id,
        )

    def setup_boundary(self):
        self.mpm_env.setup_boundary(
            type='cube',
            lower=(0.06, 0.06, 0.015),
            upper=(0.34, 0.34, 0.15)
        )

    def setup_renderer(self):
        gl_render = False
        # gl_render = True
        if gl_render:
            self.mpm_env.setup_renderer(
                type='GL',
                # render_particle=True,
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(0.5, 5.0, 0.55),
                light_lookat=(0.5, 0.5, 0.49),
            )
        else:
            if self.camera_cfg is not None:
                cam_pos = self.camera_cfg['pos']
                cam_lookat = self.camera_cfg['lookat']
                fov = self.camera_cfg['fov']
                cam_euler = self.camera_cfg['euler']
                cam_focal_length = self.camera_cfg['focal_length']
                lights = self.camera_cfg['lights']
                particle_radius = self.camera_cfg['particle_radius']
                res = self.camera_cfg['res']
                pcd_gen_res = self.camera_cfg['pcd_gen_res']
            else:
                cam_pos = (0.3, -0.1, 0.1)
                cam_lookat = (0.25, 0.25, 0.05)
                fov = 30
                lights = [{'pos': (0.5, -1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                          {'pos': (0.5, -1.5, 1.5), 'color': (0.5, 0.5, 0.5)}]
                particle_radius = 0.003
                res = (640, 480)
                pcd_gen_res = 100
                cam_euler = (135, 0, 180)
                cam_focal_length = 0.01
            self.mpm_env.setup_renderer(
                type='GGUI',
                res=res,
                pcd_gen_res=pcd_gen_res,
                # render_particle=True,
                camera_pos=cam_pos,
                camera_lookat=cam_lookat,
                camera_fov=fov,
                camera_euler=cam_euler,
                camera_focal_length=cam_focal_length,
                lights=lights,
                render_agent=self.render_agent,
                render_world_frame=False,
                particle_radius=particle_radius
            )

    def setup_loss(self):
        self.mpm_env.setup_loss(
            loss_cls=PointCloudEMDHMLosses,
            use_height_map_loss=self.loss_cfg['use_height_map_loss'],
            matching_mat=self.material_id,
            target_pcd_path=self.loss_cfg['target_pcd_path'],
            target_pcd_offset=self.loss_cfg['target_pcd_offset'],
            height_grid_res=self.loss_cfg['height_grid_res'],
            height_grid_size=0.24,
            logger=self.logger
        )


def make_env(env_cfg, loss_config, cam_cfg=None, debug_grad=False, logger=None):
    try:
        collide_type = env_cfg['collide_type']
    except KeyError:
        collide_type = 'rigid'
    try:
        grad_op = env_cfg['grad_op']
    except KeyError:
        grad_op = 'clip'

    # Environment config
    env = PlantingEnvV1(ptcl_density=env_cfg['p_density'], material_id=env_cfg['material_id'],
                        collide_type=collide_type, grad_op=grad_op,
                        agent_cfg_file='shovel_eef.yaml',
                        horizon=env_cfg['horizon'], dt_global=env_cfg['dt_global'], n_substeps=env_cfg['n_substeps'],
                        grid_scale=env_cfg['grid_scale'],
                        agent_init_euler=env_cfg['agent_init_euler'], agent_init_pos=env_cfg['agent_init_pos'],
                        has_loss=True, loss_cfg=loss_config,
                        render_agent=True, camera_cfg=cam_cfg,
                        debug_grad=debug_grad, logger=logger)

    env.reset()
    mpm_env = env.mpm_env
    init_state = mpm_env.get_state()

    return env, mpm_env, init_state


class SingleSkillEnv(gym.Env):
    def __init__(self, ti_env_cfg, gym_env_config, seed=None, logger=None):
        self.logger = logger
        self.mpm_env_cfg = ti_env_cfg['env_cfg']
        self.loss_cfg = ti_env_cfg['loss_cfg']
        self.cam_cfg = ti_env_cfg['cam_cfg']
        self.ti_cfg = ti_env_cfg['ti_cfg']
        self.mpm_env, self.mpm_env_init_state = None, None
        self.mpm_env, self.mpm_env_init_state = self.recreate_mpm_env()

        if seed is not None:
            self.seed(seed)

        self.agent_init_state = self.mpm_env_init_state['agent']

        self.step_count = 0
        self.render_skill = gym_env_config['render_skill']
        self.obs_mode = gym_env_config['obs_mode']
        self.reward_scale = gym_env_config['reward_scale']
        self.horizon = gym_env_config['horizon']
        self.action_space = Box(low=gym_env_config['action_min'],
                                high=gym_env_config['action_max'],
                                shape=(gym_env_config['action_dim'],),
                                dtype=DTYPE_NP)
        self.skill_generation_func = gym_env_config['skill_generation_func']

        self.distance_threshold = 0.01
        self.goal_conditioned_reward_function = compute_emd_loss_external

    def recreate_mpm_env(self):
        if self.mpm_env is not None:
            self.mpm_env.simulator.clear_ckpt()

        ti.reset()
        ti.init(arch=self.ti_cfg['arch'], device_memory_GB=self.ti_cfg['device_memory_GB'],
                default_fp=DTYPE_TI, fast_math=self.ti_cfg['fast_math'],
                random_seed=self.ti_cfg['random_seed'])
        env, mpm_env, init_state = make_env(self.mpm_env_cfg,
                                            self.loss_cfg,
                                            cam_cfg=self.cam_cfg,
                                            debug_grad=False,
                                            logger=self.logger)
        set_parameters(mpm_env, SAND,
                       e=self.mpm_env_cfg['best_params']['E'],
                       nu=self.mpm_env_cfg['best_params']['nu'],
                       rho=self.mpm_env_cfg['best_params']['rho'],
                       sand_friction_angle=self.mpm_env_cfg['best_params']['sand_angle'])
        return mpm_env, init_state['state']

    def step(self, action):
        skill_trajectory = self.skill_generation_func(action,
                                                      dt=self.mpm_env.simulator.dt_global)
        for i in range(len(skill_trajectory)):
            self.mpm_env.step(skill_trajectory[i])
            if self.render_skill:
                self.render(mode='human')
        self.mpm_env.simulator.agent.set_state(self.mpm_env.simulator.cur_substep_local,
                                               self.agent_init_state)
        if self.render_skill:
            self.render(mode='human')
        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        loss_info = self.mpm_env.get_final_loss()
        reward = loss_info['total_loss'] * self.reward_scale
        self.step_count += 1
        done = self.step_count >= self.horizon
        return {
            'observation': obs.copy(),
            'agent_state': agent_state.copy(),
            'desired_goal': self.up_sample_pcd(self.mpm_env.loss.target_pcd_points_np.copy()),
            'achieved_goal': obs.copy()
        }, reward, done, loss_info

    def reset(self):
        self.mpm_env, self.mpm_env_init_state = self.recreate_mpm_env()

        self.mpm_env.set_state(self.mpm_env_init_state, grad_enabled=False)
        self.step_count = 0
        obs = self.render(mode=self.obs_mode)
        agent_state = self.mpm_env.simulator.agent.get_state(self.mpm_env.simulator.cur_substep_local)
        return {
            'observation': obs.copy(),
            'agent_state': agent_state.copy(),
            'desired_goal': self.up_sample_pcd(self.mpm_env.loss.target_pcd_points_np.copy()),
            'achieved_goal': obs.copy()
        }

    def render(self, mode='human'):
        return self.mpm_env.render(mode=mode)

    def up_sample_pcd(self, pcd_np):
        target_size = self.mpm_env.loss.height_grid_res ** 2
        current_size = pcd_np.shape[0]
        if current_size >= target_size:
            return pcd_np
        random_indices = np.random.choice(current_size, size=(target_size - current_size), replace=False)
        new_points = []
        for i in range(len(random_indices)):
            new_points.append(pcd_np[random_indices[i]] + np.random.normal(0, 0.0005, size=(3,)))
        new_points = np.asarray(new_points, dtype=DTYPE_NP)
        return np.concatenate((pcd_np, new_points), axis=0)