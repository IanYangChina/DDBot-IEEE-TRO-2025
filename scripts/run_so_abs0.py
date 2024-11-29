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

from doma.optimiser.rmsprop import RMSprop
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
    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]
    task_id = args['task_id']
    ptcl_d = arguments['ptcl_density']
    subfix = ''
    if args['use_height_map_loss']:
        subfix += '-hm'
    if args['demon']:
        subfix += '-demo'
    learning_rate = args['lr']
    subfix += f'-lr{learning_rate}'

    case = f'd{ptcl_d}-task-{task_id}{subfix}'
    result_path = os.path.join(script_path, '..', 'log-abs0')

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    horizon = 300
    with open(os.path.join(script_path, '..', 'log-sys_id', f'd{ptcl_d}-hm-gclip-ls-res40', 'best_params.json')) as f:
        best_params = json.load(f)["Parameters"]

    env_cfg = {
        'p_density': float(args['ptcl_density']),
        'material_id': SAND,
        'grid_scale': 1,
        'horizon': horizon,
        'dt_global': DT_GLOBAL,
        'n_substeps': 20,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
        'grad_op': 'clip'
    }
    loss_cfg = {
        'use_height_map_loss': args['use_height_map_loss'],
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'task_target_pcds',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'height_grid_res': 40,
    }

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
                trajectory[i][5] = move_up_delta_z

        return trajectory[:n_step + n_step_insert + n_step_push + n_step_return, :]

    for seed in seeds:
        log_dir = os.path.join(result_path, case, f'seed-{seed}')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'ckpts'), exist_ok=True)

        if not args['eval'] and not args['eval_specific']:
            log_file_name = os.path.join(log_dir, 'optimisation.log')
            if os.path.isfile(log_file_name):
                filemode = "a"
            else:
                filemode = "w"
            logging.basicConfig(level=logging.NOTSET, filemode=filemode,
                                filename=log_file_name,
                                format="%(asctime)s %(levelname)s %(message)s")
            logger = SummaryWriter(log_dir=log_dir)
        trajectory_np = np.zeros((horizon, 6), dtype=DTYPE_NP)
        if args['eval']:
            with open(os.path.join(result_path, f'best_tr-task-{task_id}{subfix}.json')) as f:
                trajectory_np = json.load(f)[ptcl_d]["Trajectory"]
            if args['view_demon']:
                trajectory_demon = abstraction_two_skill(np.asarray([1.0, 0.45, 0.8, 0.0, -0.1], dtype=DTYPE_NP), DT_GLOBAL)
                trajectory_length = trajectory_demon.shape[0]
                trajectory_np[:trajectory_length] = trajectory_demon[:trajectory_length]
            print("===> Loaded trajectory.")
        elif args['eval_specific']:
            seed = 0
            epoch = 46
            trajectory_np = np.load(os.path.join(result_path, case, f'seed-{seed}', 'ckpts', f'trajectory_{epoch}.npy'))
        else:
            if args['demon']:
                trajectory_demon = abstraction_two_skill(np.asarray([1.0, 0.45, 0.8, 0.0, -0.1], dtype=DTYPE_NP), DT_GLOBAL)
                trajectory_length = trajectory_demon.shape[0]
                trajectory_np[:trajectory_length] = trajectory_demon[:trajectory_length]

        n_epoch = 50
        n_aborted_data = 0
        tr_optim = RMSprop(parameters_shape=trajectory_np.shape,
                           cfg={'lr': learning_rate, 'beta': 0.9})

        for n in range(n_epoch):
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
            if args['eval'] or args['eval_specific']:
                mpm_env.render(mode='human')

            for i in range(horizon):
                mpm_env.step(trajectory_np[i])
                if args['eval'] or args['eval_specific']:
                    mpm_env.render(mode='human')
            loss_info = mpm_env.get_final_loss()
            logger.add_scalar(tag='loss/EMD', scalar_value=loss_info["emd_loss"], global_step=n)
            logger.add_scalar(tag='loss/Heightmap', scalar_value=loss_info['height_map_loss'], global_step=n)

            if args['eval'] or args['eval_specific']:
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

                # cloud_array = mpm_env.render(mode='point_cloud')
                # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                # obj_vec = o3d.utility.Vector3dVector(cloud_array)
                # obj_pcd = o3d.geometry.PointCloud(obj_vec)
                # cloud_array_2 = mpm_env.loss.target_pcd_original_points_np.copy()
                # cloud_array_2[:, 0] += 0.3
                # obj_vec_2 = o3d.utility.Vector3dVector(cloud_array_2)
                # obj_pcd_2 = o3d.geometry.PointCloud(obj_vec_2)
                # print(obj_pcd, obj_pcd_2)
                # o3d.visualization.draw_geometries([frame, obj_pcd, obj_pcd_2], width=800, height=600)

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
            else:
                print(f'=====> Epoch: {n}')
                print(f'=====> EMD Loss: %f' % loss_info["emd_loss"])
                print(f'=====> Height map loss: %f' % loss_info['height_map_loss'])
                print(f"=====> Num. aborted data so far: {n_aborted_data}")
                logging.info(f'=====> Epoch: {n}')
                logging.info(f'=====> EMD Loss: %f' % loss_info["emd_loss"])
                logging.info(f'=====> Height map loss:  %f' % loss_info['height_map_loss'])
                logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")

                mpm_env.simulator.clear_ckpt()

                if args['line_search']:
                    print(f'===========> Performing line search for epoch {n}......')
                    loss_to_optimise = 'height_map_loss'
                    best_loss_tmp = +np.inf
                    best_alpha = 1.0
                    best_loss_info = loss_info
                    for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
                        tr_optim.lr = learning_rate * alpha
                        trajectory_tmp = tr_optim.step(trajectory_np, trajectory_grads_np[:trajectory_np.shape[0]])

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
                        for i in range(horizon):
                            mpm_env.step(trajectory_tmp[i])
                        loss_info = mpm_env.get_final_loss()

                        print(f'=====> alpha: {alpha}')
                        print(f'==> EMD Loss: %f' % loss_info["emd_loss"])
                        print(f'==> Height map loss: %f' % loss_info['height_map_loss'])
                        logging.info(f'==> EMD Loss: %f' % loss_info["emd_loss"])
                        logging.info(f'==> Height map loss:  %f' % loss_info['height_map_loss'])
                        if loss_info[loss_to_optimise] < best_loss_tmp:
                            best_loss_tmp = loss_info[loss_to_optimise]
                            best_alpha = alpha
                            best_loss_info = loss_info

                        tr_optim.reverse_normaliser()

                    print(f'=====> Best alpha: {best_alpha}, Best loss info: {best_loss_info}')
                    logging.info(f'=====> Best alpha: {best_alpha}, Best loss info: {best_loss_info}')
                    tr_optim.lr = learning_rate * best_alpha
                    logger.add_scalar(tag='Grad/best_alpha', scalar_value=best_alpha, global_step=n)

                trajectory_np = tr_optim.step(trajectory_np, trajectory_grads_np[:trajectory_np.shape[0]])
                trajectory_np = np.clip(trajectory_np, -0.004, 0.004)

                np.save(os.path.join(log_dir, 'ckpts', f'grad_{n}.npy'), grad)
                np.save(os.path.join(log_dir, 'ckpts', f'trajectory_{n}.npy'), trajectory_np)

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
        for i in range(horizon):
            mpm_env.step(trajectory_np[i])
        loss_info = mpm_env.get_final_loss()

        logger.add_scalar(tag='loss/EMD', scalar_value=loss_info["emd_loss"], global_step=n_epoch)
        logger.add_scalar(tag='loss/Heightmap', scalar_value=loss_info['height_map_loss'], global_step=n_epoch)
        print('===========> Finished training.')
        print('====> Final EMD loss: %f' % loss_info["emd_loss"])
        print('====> Final height map loss: %f' % loss_info['height_map_loss'])
        logging.info('===========> Finished training.')
        logging.info(f'====> Final EMD loss: %f' % loss_info["emd_loss"])
        logging.info(f'====> Final height map loss: %f' % loss_info['height_map_loss'])
        logger.close()
        np.save(os.path.join(log_dir, 'final_trajectory.npy'), trajectory_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient-based trajectory optimisation")
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default="5e6", help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--task-id', dest='task_id', type=int, default=0, help='task id')
    parser.add_argument('--demon', dest='demon', action='store_true', default=False, help='Use demonstration')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('--eval-specific', dest='eval_specific', action='store_true', default=False, help='Evaluate the model with specific epoch')
    parser.add_argument('--view-demon', dest='view_demon', action='store_true', default=False, help='View demonstration')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='Learning rate')
    arguments = vars(parser.parse_args())
    main(arguments)
