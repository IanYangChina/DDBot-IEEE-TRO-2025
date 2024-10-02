import os
import yaml
import json
import logging
import argparse
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.optimiser.adam import Adam
from doma.envs.planting_env import make_env
from doma.engine.configs.macros import DTYPE_NP, DTYPE_TI, SAND
from doma.engine.utils.misc import set_parameters
cam_cfg = {
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

LINEAR_VELOCITY = 0.2  # m/s
ANGULAR_VELOCITY = np.pi / 4  # rad/s
DT_GLOBAL = 0.01  # sec


def main(args):
    seed = args['seed']
    # np_rng = np.random.default_rng(seed=seed)
    task_id = args['task_id']
    ptcl_d = arguments['ptcl_density']
    subfix = ''
    if args['use_height_map_loss']:
        subfix += '-hm'
    if args['demon']:
        subfix += '-demo'
    if args['mini_batch']:
        subfix += '-batch'

    log_dir = os.path.join(script_path, '..', 'log-abs0-adam', f'd{ptcl_d}-task-{task_id}{subfix}', f'seed-{seed}')
    if args['compute_grad']:
        log_dir = log_dir[:-6] + 'grads'
    grad_mean_file_name = os.path.join(log_dir, 'grad_mean.npy')
    grad_std_file_name = os.path.join(log_dir, 'grad_std.npy')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'ckpts'), exist_ok=True)

    if not args['eval']:
        log_file_name = os.path.join(log_dir, 'optimisation.log')
        if os.path.isfile(log_file_name):
            filemode = "a"
        else:
            filemode = "w"
        logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                            filename=log_file_name,
                            format="%(asctime)s %(levelname)s %(message)s")
        logger = SummaryWriter(log_dir=log_dir)

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    horizon = 300
    with open(os.path.join(script_path, '..', 'log-sys_id', 'best_params.json')) as f:
        best_params = json.load(f)[arguments['ptcl_density']]["Parameters"]

    env_cfg = {
        'p_density': float(args['ptcl_density']),
        'material_id': SAND,
        'grid_scale': 1,
        'horizon': horizon,
        'dt_global': DT_GLOBAL,
        'n_substeps': 20,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'use_height_map_loss': args['use_height_map_loss'],
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'task_target_pcds',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.007,
    }

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

    trajectory_np = np.zeros((horizon, 6), dtype=DTYPE_NP)
    if args['eval']:
        with open(os.path.join(script_path, '..', 'log-abs0-adam',
                               f'best_tr-task-{task_id}{subfix}.json')) as f:
            trajectory_np = json.load(f)[ptcl_d]["Trajectory"]
        if args['view_demon']:
            trajectory_demon = abstraction_two_skill(np.asarray([1.0, 0.3, 0.8, 1.0, 0.3], dtype=DTYPE_NP), DT_GLOBAL)
            trajectory_length = trajectory_demon.shape[0]
            trajectory_np[:trajectory_length] = trajectory_demon[:trajectory_length]
        print("===> Loaded trajectory.")
    else:
        if args['demon']:
            trajectory_demon = abstraction_two_skill(np.asarray([1.0, 0.3, 0.8, 1.0, 0.3], dtype=DTYPE_NP), DT_GLOBAL)
            trajectory_length = trajectory_demon.shape[0]
            trajectory_np[:trajectory_length] = trajectory_demon[:trajectory_length]

    n_epoch = 150
    n_aborted_data = 0
    emd_loss = 0.0
    height_map_loss = 0.0
    grads_to_save = []
    if args['use_height_map_loss']:
        lrs = 0.0001
        if args['demon']:
            lrs = 0.001
    else:
        lrs = 0.001
    tr_optim = Adam(parameters_shape=trajectory_np.shape,
                    cfg={'lr': lrs, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

    for n in range(n_epoch):
        grads = []
        emd_losses = []
        height_map_losses = []
        if args['mini_batch']:
            n_inner_epoch = 3
        else:
            n_inner_epoch = 1
        for k in range(n_inner_epoch):
            ti.reset()
            ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=DTYPE_TI,
                    fast_math=True, random_seed=args['seed'])

            env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)

            set_parameters(mpm_env, SAND,
                           e=best_params['E'],
                           nu=best_params['nu'],
                           rho=best_params['rho'],
                           sand_friction_angle=best_params['sand_angle'])

            """forward pass"""
            mpm_env.set_state(init_state['state'], grad_enabled=True)
            if args['eval']:
                mpm_env.render(mode='human')

            for i in range(horizon):
                mpm_env.step(trajectory_np[i])
                if args['eval']:
                    mpm_env.render(mode='human')
            loss_info = mpm_env.get_final_loss()

            if args['eval']:
                print('===> Loss info:', loss_info)
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(mpm_env.loss.height_map.to_numpy(),
                             vmin=0.002, vmax=0.09)
                ax[0].set_title('Height map')
                ax[1].imshow(mpm_env.loss.height_map_pcd_target.to_numpy(),
                             vmin=0.002, vmax=0.09)
                ax[1].set_title('Target height map')
                plt.show()
                plt.close()

                cloud_array = mpm_env.render(mode='point_cloud')
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                obj_vec = o3d.utility.Vector3dVector(cloud_array)
                obj_pcd = o3d.geometry.PointCloud(obj_vec)
                cloud_array_2 = mpm_env.loss.target_pcd_original_points_np.copy()
                cloud_array_2[:, 0] += 0.3
                obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2)
                print(obj_pcd, obj_pcd_2)
                o3d.visualization.draw_geometries([frame, obj_pcd, obj_pcd_2], width=800, height=600)

                exit()

            """backward pass"""
            mpm_env.reset_grad()
            mpm_env.get_final_loss_grad()
            for i in range(horizon - 1, -1, -1):
                action = trajectory_np[i]
                mpm_env.step_grad(action=action)

                # This is a trick that prevents faulty gradient computation
                # It works for unknown reasons
                _ = mpm_env.simulator.particle_param.grad[2].E

            trajectory_grads_np = mpm_env.agent.get_grad(horizon)

            """Checking for nan, inf, and strange values"""
            abort = False
            particle_has_naninf = loss_info['particle_has_naninf']
            if particle_has_naninf:
                abort = True

            if not abort:
                for i, v in loss_info.items():
                    if i == 'particle_has_naninf':
                        pass
                    else:
                        if np.isinf(v) or np.isnan(v):
                            abort = True
                            break

            if not abort:
                if np.any(np.isnan(trajectory_grads_np)) or np.any(np.isinf(trajectory_grads_np)) or np.any(np.abs(trajectory_grads_np) > 1e10):
                    abort = True

            if abort:
                print(f'===> [Warning] Aborting epoch: {n}')
                print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                print(f'===> [Warning] Strange loss or gradient.')
                print(f'===> [Warning] Trajectory grad mean: {trajectory_grads_np.mean()}')
                print(f'===> [Warning] Loss info: {loss_info}')
                logging.error(f'===> [Warning] Aborting epoch: {n}')
                logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                logging.error(f'===> [Warning] Strange loss or gradient.')
                logging.error(f'===> [Warning] Trajectory grad mean: {trajectory_grads_np.mean()}')
                logging.error(f'===> [Warning] Loss info: {loss_info}')
                n_aborted_data += 1
                grads.append(np.ones_like(trajectory_grads_np)*1e-6)
            else:
                grads.append(trajectory_grads_np)
                emd_losses.append(loss_info['emd_loss'])
                height_map_losses.append(loss_info['height_map_loss'])

            mpm_env.simulator.clear_ckpt()

        grad = np.mean(grads, axis=0) * -1.0
        emd_loss = np.mean(emd_losses)
        height_map_loss = np.mean(height_map_losses)
        if args['compute_grad']:
            grads_to_save.append(grad)
            trajectory_np = np.ones((horizon, 6), dtype=DTYPE_NP)
            trajectory_np[:, :3] = np.random.uniform(-0.004, 0.004, size=(horizon, 3))
            trajectory_np[:, 3:] = np.random.uniform(-0.0157, 0.0157, size=(horizon, 3))
            if args['demon']:
                skill_params_np = np.asarray([1.0, 0.3, 0.8, 1.0, 0.3]).astype(DTYPE_NP) + np.random.uniform(-1, 1, size=5).astype(DTYPE_NP) * 0.5
                skill_params_np = np.clip(skill_params_np, -1, 1)
                trajectory_demon = abstraction_two_skill(skill_params_np, DT_GLOBAL)
                trajectory_length = trajectory_demon.shape[0]
                if trajectory_length > horizon:
                    trajectory_length = horizon
                trajectory_np[:trajectory_length] = trajectory_demon[:trajectory_length]

            print(f'=====> Epoch: {n}')
            print(f'=====> EMD Loss: {emd_loss}')
            print(f'=====> Height map loss: {height_map_loss}')
            print(f'=====> Trajectory grad mean: {grad.mean()}')
            print(f"=====> Num. aborted data so far: {n_aborted_data}")
            logging.info(f'=====> Epoch: {n}')
            logging.info(f'=====> EMD Loss: {emd_loss}')
            logging.info(f'=====> Height map loss: {height_map_loss}')
            logging.info(f'=====> Trajectory grad mean: {grad.mean()}')
            logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")
        else:
            trajectory_np = tr_optim.step(trajectory_np, grad[:trajectory_np.shape[0]])
            trajectory_np[:, :3] = np.clip(trajectory_np[:, :3], -0.004, 0.004)
            trajectory_np[:, 3:] = np.clip(trajectory_np[:, 3:], -0.0157, 0.0157)

            np.save(os.path.join(log_dir, 'ckpts', f'grad_{n}.npy'), grad)
            np.save(os.path.join(log_dir, 'ckpts', f'trajectory_{n}.npy'), trajectory_np)
            print(f'=====> Epoch: {n}')
            print(f'=====> EMD Loss: {emd_loss}')
            print(f'=====> Height map loss: {height_map_loss}')
            print(f'=====> Trajectory grad mean: {grad.mean()}')
            print(f"=====> Num. aborted data so far: {n_aborted_data}")
            logging.info(f'=====> Epoch: {n}')
            logging.info(f'=====> EMD Loss: {emd_loss}')
            logging.info(f'=====> Height map loss: {height_map_loss}')
            logging.info(f'=====> Trajectory grad mean: {grad.mean()}')
            logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")

            logger.add_scalar(tag='loss/EMD', scalar_value=emd_loss, global_step=n)
            logger.add_scalar(tag='loss/Heightmap', scalar_value=height_map_loss, global_step=n)

    logger.close()
    if args['compute_grad']:
        grad_mean = np.mean(grads_to_save, axis=0)
        grad_std = np.std(grads_to_save, axis=0)
        np.save(grad_mean_file_name, grad_mean)
        np.save(grad_std_file_name, grad_std)
        print('====> Mean grad: ', grad_mean)
        print('====> Std grad: ', grad_std)
        logging.info(f'====> Mean grad: {grad_mean}')
        logging.info(f'====> Std grad: {grad_std}')
    else:
        np.save(os.path.join(log_dir, 'final_trajectory.npy'), trajectory_np)
        print('====> Finished training.')
        print('====> Final EMD loss: ', emd_loss)
        print('====> Final height map loss: ', height_map_loss)
        logging.info('====> Finished training.')
        logging.info(f'====> Final EMD loss: {emd_loss}')
        logging.info(f'====> Final height map loss: {height_map_loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient-based trajectory optimisation")
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='Random seed')
    parser.add_argument('--com_grad', dest='compute_grad', action='store_true', default=False, help='Compute gradient')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default="5e6",
                        help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str,
                        help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--task-id', dest='task_id', type=int, default=0, help='task id')
    parser.add_argument('--demon', dest='demon', action='store_true', default=False, help='Use demonstration')
    parser.add_argument('--mini-batch', dest='mini_batch', action='store_true', default=False, help='Use mini-batch')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('--view-demon', dest='view_demon', action='store_true', default=False, help='View demonstration')
    arguments = vars(parser.parse_args())
    main(arguments)
