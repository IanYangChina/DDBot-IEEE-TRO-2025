import os
import open3d as o3d
import json
import logging
import argparse
import numpy as np
import taichi as ti
from torch.utils.tensorboard import SummaryWriter
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.engine.configs.macros import DTYPE_NP, SAND
from doma.envs.gym_wrappers import SingleSkillEnv
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
    'pcd_gen_res': 40
}

LINEAR_VELOCITY = 0.2  # m/s
ANGULAR_VELOCITY = np.pi / 4  # rad/s
DT_GLOBAL = 0.01  # sec

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

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

def main(arguments):
    seed = arguments['seed']
    np_rng = np.random.default_rng(seed=seed)
    task_id = arguments['task_id']
    ptcl_d = arguments['ptcl_density']
    suffix = ''
    if arguments['use_demo']:
        suffix += '-demo'
    if arguments['sand']:
        mat = '_sand'
    else:
        mat = ''

    suffix += f'-em{arguments["n_emitters"]}'
    suffix += f'-bs{arguments["batch_size"]}'

    log_dir = os.path.join(script_path, '..', f'log-abs2-cmamae{mat}', f'd{ptcl_d}-task-{task_id}{suffix}', f'seed-{seed}')
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, 'optimisation.log')
    if os.path.isfile(log_file_name):
        filemode = "a"
    else:
        filemode = "w"
    logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                        filename=log_file_name,
                        format="%(asctime)s %(levelname)s %(message)s")
    logger = SummaryWriter(log_dir=log_dir)
    backend = ti.cuda
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
        'use_height_map_loss': arguments['use_height_map_loss'],
        'target_pcd_path': os.path.join(script_path, '..', 'data', f'task_target_pcds{mat}',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'height_grid_res': 40,
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
    with open(os.path.join(script_path, '..', f'log-sys_id{mat}',
                           f'd{ptcl_d}-hm-gclip-ls-man-init-res40', 'best_params.json')) as f:
        best_params = json.load(f)["Parameters"]
    env_cfg['best_params'] = best_params

    gym_env_config = {
        'pcd_file_path': os.path.join(script_path, '..', 'data', 'task_target_pcds',
                                      f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'render_skill': False,
        'horizon': 1,
        'obs_mode': 'point_cloud',
        'reward_scale': -1.0,
        'action_dim': 5,
        'action_max': 1.0,
        'action_min': -1.0,
        'skill_generation_func': abstraction_two_skill
    }
    gym_env = SingleSkillEnv(ti_env_cfg, gym_env_config, seed=arguments['seed'], logger=logging)

    with open(os.path.join(script_path, '..', 'data', 'cma_mae_config.json'), 'rb') as f_ac:
        cma_mae_config = json.load(f_ac)
    cma_mae_config['use_demonstrations'] = arguments['use_demo']
    cma_mae_config['total_iterations'] = arguments['total_iterations']
    cma_mae_config['n_emitters'] = arguments['n_emitters']
    cma_mae_config['batch_size'] = arguments['batch_size']

    if arguments['use_demo']:
        target_pcd = o3d.io.read_point_cloud(os.path.join(script_path, '..', 'data', f'task_target_pcds{mat}',
                                                          f'pcd_{task_id}_cropped_norm_z_aligned.ply'))
        target_pcd_points = np.asarray(target_pcd.points) + np.asarray([0.2, 0.2, 0])
        z_min_idx = np.argmin(target_pcd_points[:, 2])
        x_demo = target_pcd_points[z_min_idx, 0] + 0.02
        init_solution = np.clip((x_demo - 0.2) / 0.12, -1.0, 1.0)
    else:
        init_solution = np_rng.uniform(-1, 1, size=gym_env_config['action_dim']).astype(DTYPE_NP)

    archive = GridArchive(solution_dim=gym_env_config['action_dim'],
                          dims=[200, 200],
                          ranges=[(-2.5, 2.5), (-2.5, 2.5)],  # 5 / 2 * 1.0
                          learning_rate=cma_mae_config['learning_rate'],
                          threshold_min=0.0)
    result_archive = GridArchive(solution_dim=gym_env_config['action_dim'],
                                 dims=[200, 200],
                                 ranges=[(-2.5, 2.5), (-2.5, 2.5)])  # 5 / 2 * 1.0

    batch_size = int(cma_mae_config['batch_size'] / arguments['n_emitters'])
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=init_solution,
            sigma0=cma_mae_config['sigma0'],
            bounds=[(-1.0, 1.0)] * gym_env_config['action_dim'],
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=batch_size,
            seed=seed,
        ) for _ in range(arguments['n_emitters'])  # n_emitters
    ]

    scheduler = Scheduler(archive, emitters, result_archive=result_archive)

    print("Starting optimization...")
    logging.info("Starting optimization...")
    print("Configs:")
    logging.info("Configs:")
    for k, v in cma_mae_config.items():
        print(f"{k}: {v}")
        logging.info(f"{k}: {v}")

    for itr in range(arguments['total_iterations']):
        solution_batch = scheduler.ask()
        print(f"Num of solutions: {solution_batch.shape[0]}")
        objective_batch = np.zeros((solution_batch.shape[0],), dtype=DTYPE_NP)
        for i in range(solution_batch.shape[0]):
            skill = solution_batch[i]
            gym_env.reset()
            _, _, _, loss_info = gym_env.step(skill)
            objective_batch[i] = loss_info['emd_loss']

            if i < 5:
                logger.add_scalar(tag=f'loss_{i}/EMD', scalar_value=loss_info['emd_loss'], global_step=itr)
                logger.add_scalar(tag=f'loss_{i}/Heightmap', scalar_value=loss_info['height_map_loss'], global_step=itr)
                logger.add_scalar(tag=f'param_{i}/0-move_distance', scalar_value=skill[0], global_step=itr)
                logger.add_scalar(tag=f'param_{i}/1-rotate_x', scalar_value=skill[1], global_step=itr)
                logger.add_scalar(tag=f'param_{i}/2-insert_distance', scalar_value=skill[2], global_step=itr)
                logger.add_scalar(tag=f'param_{i}/3-push_angle', scalar_value=skill[3], global_step=itr)
                logger.add_scalar(tag=f'param_{i}/4-push_distance', scalar_value=skill[4], global_step=itr)

        # calculate measures
        clipped = solution_batch.copy()
        clip_mask = (clipped < -1.0) | (clipped > 1.0)
        clipped[clip_mask] = np.clip(clipped[clip_mask], -1.0, 1.0)
        measure_batch = np.concatenate((
            np.sum(clipped[:, :batch_size // 2], axis=1, keepdims=True),
            np.sum(clipped[:, batch_size // 2:], axis=1, keepdims=True),
        ), axis=1,)
        scheduler.tell(objective_batch, measure_batch)

        print(f"Iteration {itr} | "
              f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}% "
              f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")
        logging.info(f"Iteration {itr} | "
                     f"Archive Coverage: {result_archive.stats.coverage * 100:6.3f}% "
                     f"Normalized QD Score: {result_archive.stats.norm_qd_score:6.3f}")

    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='seed')
    parser.add_argument('--task-id', dest='task_id', type=int, default=0, help='task id')
    parser.add_argument('--demo', dest='use_demo', action='store_true', default=False, help='use demonstrations')
    parser.add_argument('--ptcl-d', dest='ptcl_density', type=str, default="5e6", help='particle density')
    parser.add_argument('--t-cuda-id', dest='torch_cuda_device_id', type=int, default=0, help='cuda device id')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--sand', dest='sand', action='store_true', default=False, help='Use sand')
    parser.add_argument('--total_iterations', dest='total_iterations', type=int, default=200, help='number of iterations')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=10, help='batch size')
    parser.add_argument('--n_emitters', dest='n_emitters', type=int, default=1, help='number of emitters')
    args = vars(parser.parse_args())
    main(args)
