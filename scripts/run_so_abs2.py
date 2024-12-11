import os
import yaml
import json
import argparse
import logging
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
import open3d as o3d
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy as dcp

script_path = os.path.dirname(os.path.realpath(__file__))

from doma.optimiser.adam import Adam, GD
from doma.optimiser.rmsprop import RMSprop
from doma.envs.planting_env import make_env
from doma.engine.configs.macros import DTYPE_NP, DTYPE_TI, SAND
from doma.engine.utils.misc import set_parameters, reset_logging

cam_cfg = {
    'pos': (0.2, 1.2, 0.9),
    'lookat': (0.2, 0.2, 0.03),
    'euler': (180 + np.rad2deg(np.arctan(1.0 / (0.9 - 0.03))), 0, 180),
    'focal_length': 0.3,
    'fov': 30,
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


def main(args):
    seed = args['seed']
    if seed == -1:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [seed]

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    task_id = args['task_id']
    ptcl_d = arguments['ptcl_density']
    subfix = ''
    if args['use_height_map_loss']:
        subfix += '-hm'
    if args['line_search']:
        subfix += '-ls'
    if args['demon']:
        subfix += '-demo'

    learning_rate = args['lr']
    subfix += f'-lr{learning_rate}'

    if args['sand']:
        mat = '_sand'
    else:
        mat = ''

    case = f'd{ptcl_d}-task-{task_id}{subfix}'
    if args['multi_skill']:
        result_path = os.path.join(script_path, '..', f'log-abs2-ms{mat}')
    else:
        result_path = os.path.join(script_path, '..', f'log-abs2{mat}')

    with open(os.path.join(script_path, '..', f'log-sys_id{mat}',
                           f'd{ptcl_d}-hm-gclip-ls-man-init-res40', 'best_params.json')) as f:
        best_params = json.load(f)["Parameters"]

    horizon = 600
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
        'target_pcd_path': os.path.join(script_path, '..', 'data', f'task_target_pcds{mat}',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'height_grid_res': 40,
    }
    if args['multi_skill']:
        loss_cfg['target_pcd_path'] = os.path.join(script_path, '..', 'data', f'task_target_pcds{mat}',
                                                   f'pcd_0_multi_skill_cropped_norm_z_aligned.ply')

    for seed in seeds:
        reset_logging(logging)
        np_rng = np.random.default_rng(seed=seed)
        log_dir = os.path.join(result_path, case, f'seed-{seed}')
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

        if args['demon']:
            skill_params_np = np.asarray([1.0, 0.2, 0.8, 0.0, -0.5]).astype(DTYPE_NP)
            target_pcd = o3d.io.read_point_cloud(os.path.join(script_path, '..', 'data', f'task_target_pcds{mat}',
                                                              f'pcd_{task_id}_cropped_norm_z_aligned.ply'))
            target_pcd_points = np.asarray(target_pcd.points) + np.asarray([0.2, 0.2, 0])
            z_min_idx = np.argmin(target_pcd_points[:, 2])
            x_demo = target_pcd_points[z_min_idx, 0] + 0.02
            skill_params_np[0] = np.clip((x_demo - 0.2) / 0.12, -1.0, 1.0)
        else:
            skill_params_np = np_rng.uniform(-1, 1, size=5).astype(DTYPE_NP)

        n_epoch = 20
        n_aborted_data = 0
        optimiser = RMSprop(parameters_shape=(5,),
                            cfg={'lr': learning_rate, 'beta': 0.9})

        for n in range(n_epoch):
            ti.reset()
            ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=DTYPE_TI,
                    fast_math=True, random_seed=args['seed'])

            skill_params_ti = ti.field(dtype=DTYPE_TI, shape=5, needs_grad=True)
            n_step_total = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

            move_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            rotate_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            n_step_move = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            insert_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            insert_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            n_step_insert = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            push_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            push_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            n_step_push = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            rotate_delta_x_back = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            move_up_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
            n_step_return = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

            trajectory = ti.Vector.field(n=6, dtype=DTYPE_TI, shape=horizon, needs_grad=True)

            def reset_vars():
                n_step_total.fill(0)
                move_delta_x.fill(0)
                rotate_delta_x.fill(0)
                n_step_move.fill(0)
                insert_delta_x.fill(0)
                insert_delta_z.fill(0)
                n_step_insert.fill(0)
                push_delta_x.fill(0)
                push_delta_z.fill(0)
                n_step_push.fill(0)
                rotate_delta_x_back.fill(0)
                move_up_delta_z.fill(0)
                n_step_return.fill(0)
                trajectory.fill(0)

            def reset_grads():
                skill_params_ti.grad.fill(0)
                trajectory.grad.fill(0)
                move_delta_x.grad.fill(0)
                rotate_delta_x.grad.fill(0)
                n_step_move.grad.fill(0)
                insert_delta_x.grad.fill(0)
                insert_delta_z.grad.fill(0)
                n_step_insert.grad.fill(0)
                push_delta_x.grad.fill(0)
                push_delta_z.grad.fill(0)
                n_step_push.grad.fill(0)
                rotate_delta_x_back.grad.fill(0)
                move_up_delta_z.grad.fill(0)
                n_step_return.grad.fill(0)

            @ti.kernel
            def abstraction_two_skill():
                move_distance = skill_params_ti[0] * 0.12
                rotate_x = skill_params_ti[1] * (np.pi / 3)  # map [-1, 1] to [-pi/3, pi/3]
                n_step_move[None] = ti.abs(move_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                n_step_rotate = ti.abs(rotate_x / (ANGULAR_VELOCITY * DT_GLOBAL))
                ti.atomic_max(n_step_move[None], n_step_rotate)
                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                if n_step_move_int > 0:
                    move_delta_x[None] = move_distance / n_step_move[None]
                    rotate_delta_x[None] = rotate_x / n_step_move[None]
                n_step_total[None] += n_step_move_int

                insert_angle = rotate_x + np.pi / 2
                insert_distance = (skill_params_ti[2] + 1) / 2 * 0.06  # map [-1, 1] to [0, 0.06]
                n_step_insert[None] = ti.abs(insert_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                if n_step_insert_int > 0:
                    insert_distance_x = insert_distance * ti.cos(insert_angle)
                    insert_distance_z = insert_distance * ti.sin(insert_angle)
                    insert_delta_x[None] = insert_distance_x / n_step_insert[None]
                    insert_delta_z[None] = insert_distance_z / n_step_insert[None]
                n_step_total[None] += n_step_insert_int

                push_angle = (skill_params_ti[3] + 3) * np.pi / 3  # map [-1, 1] to [2*pi/3, 4*pi/3]
                push_distance = (skill_params_ti[4] + 1) * 0.1 + 0.04  # map [-1, 1] to [0.04, 0.24]
                n_step_push[None] = ti.abs(push_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                n_step_push_int = ti.floor(n_step_push[None], ti.i32)
                # print('n_step_push', n_step_push[None])
                if n_step_push_int > 0:
                    push_distance_x = push_distance * ti.cos(push_angle)
                    # print('push_distance_x:', push_distance_x)
                    push_distance_z = push_distance * ti.sin(push_angle)
                    # print('push_distance_z:', push_distance_z)
                    push_delta_x[None] = push_distance_x / n_step_push[None]
                    push_delta_z[None] = push_distance_z / n_step_push[None]
                n_step_total[None] += n_step_push_int

                rotate_x_back = -rotate_x
                n_step_rotate_back = ti.abs(rotate_x / (ANGULAR_VELOCITY * DT_GLOBAL))
                move_up_distance = 0.1
                n_step_move_up = ti.abs(move_up_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                n_step_return[None] = n_step_rotate_back
                ti.atomic_max(n_step_return[None], n_step_move_up)
                n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                if n_step_return_int > 0:
                    rotate_delta_x_back[None] = rotate_x_back / n_step_return[None]
                    move_up_delta_z[None] = move_up_distance / n_step_return[None]
                n_step_total[None] += n_step_return_int

            @ti.kernel
            def fill_trajectory_10():
                for k in range(1):
                    n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                    half_n_step_move_int = n_step_move_int // 2
                    for j in range(half_n_step_move_int):
                        trajectory[j][0] = move_delta_x[None]
                        trajectory[j][3] = rotate_delta_x[None]

            @ti.kernel
            def fill_trajectory_11():
                for k in range(1):
                    n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                    half_n_step_move_int = n_step_move_int // 2
                    for j in range(n_step_move_int - half_n_step_move_int):
                        index = j + half_n_step_move_int
                        trajectory[index][0] = move_delta_x[None]
                        trajectory[index][3] = rotate_delta_x[None]

            @ti.kernel
            def fill_trajectory_2():
                for k in range(1):
                    n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                    n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                    for j in range(n_step_insert_int):
                        index = j + n_step_move_int
                        trajectory[index][0] = insert_delta_x[None]
                        trajectory[index][2] = -insert_delta_z[None]

            @ti.kernel
            def fill_trajectory_3():
                for k in range(1):
                    n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                    n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                    n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                    for j in range(n_step_push_int):
                        index = j + n_step_move_int
                        index = index + n_step_insert_int
                        trajectory[index][0] = push_delta_x[None]
                        trajectory[index][2] = push_delta_z[None]

            @ti.kernel
            def fill_trajectory_40():
                for k in range(1):
                    n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                    n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                    n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                    n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                    half_n_step_return_int = n_step_return_int // 2
                    for j in range(half_n_step_return_int):
                        index = j + n_step_move_int + n_step_insert_int + n_step_push_int
                        trajectory[index][3] = rotate_delta_x_back[None]
                        trajectory[index][2] = move_up_delta_z[None]

            @ti.kernel
            def fill_trajectory_41():
                for k in range(1):
                    n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                    n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                    n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                    n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                    half_n_step_return_int = n_step_return_int // 2
                    for j in range(half_n_step_return_int - half_n_step_return_int):
                        index = j + n_step_move_int + n_step_insert_int + n_step_push_int + half_n_step_return_int
                        trajectory[index][3] = rotate_delta_x_back[None]
                        trajectory[index][2] = move_up_delta_z[None]

            def update_trajectory_grad(tr_grads, length):
                for k in range(length):
                    trajectory.grad[k][0] = tr_grads[k][0]
                    trajectory.grad[k][1] = tr_grads[k][1]
                    trajectory.grad[k][2] = tr_grads[k][2]
                    trajectory.grad[k][3] = tr_grads[k][3]
                    trajectory.grad[k][4] = tr_grads[k][4]
                    trajectory.grad[k][5] = tr_grads[k][5]

            env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False)

            set_parameters(mpm_env, SAND,
                           e=best_params['E'],
                           nu=best_params['nu'],
                           rho=best_params['rho'],
                           sand_friction_angle=best_params['sand_angle'])

            """forward pass"""
            mpm_env.set_state(init_state['state'], grad_enabled=True)
            # prepare trajectory
            skill_params_ti.from_numpy(skill_params_np.copy())
            reset_vars()
            abstraction_two_skill()
            fill_trajectory_10()
            fill_trajectory_11()
            fill_trajectory_2()
            fill_trajectory_3()
            fill_trajectory_40()
            fill_trajectory_41()
            trajectory_np = trajectory.to_numpy()
            trajectory_length = int(n_step_total[None])
            for i in range(trajectory_length):
                mpm_env.step(trajectory_np[i])
            loss_info = mpm_env.get_final_loss()

            """backward pass"""
            reset_grads()
            mpm_env.reset_grad()
            mpm_env.get_final_loss_grad()
            for i in range(trajectory_length - 1, -1, -1):
                action = trajectory_np[i]
                mpm_env.step_grad(action=action)

                # This is a trick that prevents faulty gradient computation
                # It works for unknown reasons
                _ = mpm_env.simulator.particle_param.grad[2].E

            trajectory_grads = mpm_env.agent.get_grad(trajectory_length)
            update_trajectory_grad(trajectory_grads, trajectory_length)
            fill_trajectory_41.grad()
            fill_trajectory_40.grad()
            fill_trajectory_3.grad()
            fill_trajectory_2.grad()
            fill_trajectory_11.grad()
            fill_trajectory_10.grad()
            abstraction_two_skill.grad()

            skill_params_grad_np = skill_params_ti.grad.to_numpy()

            print(f'=====> Skill params: {skill_params_np}')
            print(f'=====> Grad: {skill_params_grad_np}')
            print(f"=====> Num. aborted data so far: {n_aborted_data}")
            logging.info(f'=====> Skill params: {skill_params_np}')
            logging.info(f'=====> Grad: {skill_params_grad_np}')
            logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")

            logger.add_scalar(tag='loss/EMD', scalar_value=loss_info['emd_loss'], global_step=n)
            logger.add_scalar(tag='loss/Heightmap', scalar_value=loss_info['height_map_loss'], global_step=n)

            logger.add_scalar(tag='param/0-move_distance', scalar_value=skill_params_np[0], global_step=n)
            logger.add_scalar(tag='param/1-rotate_x', scalar_value=skill_params_np[1], global_step=n)
            logger.add_scalar(tag='param/2-insert_distance', scalar_value=skill_params_np[2], global_step=n)
            logger.add_scalar(tag='param/3-push_angle', scalar_value=skill_params_np[3], global_step=n)
            logger.add_scalar(tag='param/4-push_distance', scalar_value=skill_params_np[4], global_step=n)

            logger.add_scalar(tag='grad/0-move_distance', scalar_value=skill_params_grad_np[0], global_step=n)
            logger.add_scalar(tag='grad/1-rotate_x', scalar_value=skill_params_grad_np[1], global_step=n)
            logger.add_scalar(tag='grad/2-insert_distance', scalar_value=skill_params_grad_np[2], global_step=n)
            logger.add_scalar(tag='grad/3-push_angle', scalar_value=skill_params_grad_np[3], global_step=n)
            logger.add_scalar(tag='grad/4-push_distance', scalar_value=skill_params_grad_np[4], global_step=n)

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
                if np.any(np.isnan(skill_params_grad_np)) or np.any(np.isinf(skill_params_grad_np)) or np.any(
                        np.abs(skill_params_grad_np) > 1e10):
                    abort = True

            #if not abort:
            #    num_zero_grad = 0
            #    for h in range(5):
            #        if skill_params_grad_np[h] == 0.0:
            #            num_zero_grad += 1
            #    if num_zero_grad > 4:
            #        abort = True

            if abort:
                print(f'===> [Warning] Aborting epoch: {n}')
                print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                print(f'===> [Warning] Strange loss or gradient.')
                print(f'===> [Warning] Skill params: {skill_params_np}')
                print(f'===> [Warning] Skill params grad: {skill_params_grad_np}')
                print('===> [Warning] EMD Loss: %f' % loss_info['emd_loss'])
                print('===> [Warning] Height map loss: %f' % loss_info['height_map_loss'])
                logging.error(f'===> [Warning] Aborting epoch: {n}')
                logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                logging.error(f'===> [Warning] Strange loss or gradient.')
                logging.error(f'===> [Warning] Skill params: {skill_params_np}')
                logging.error(f'===> [Warning] Skill params grad: {skill_params_grad_np}')
                logging.error('===> [Warning] EMD Loss: %f' % loss_info['emd_loss'])
                logging.error('===> [Warning] Height map loss: %f' % loss_info['height_map_loss'])
                n_aborted_data += 1
            else:
                print(f'===========> Epoch: {n} optimisation losses')
                print('=====> EMD Loss: %f' % loss_info['emd_loss'])
                print('=====> Height map loss: %f' % loss_info['height_map_loss'])
                logging.info(f'===========> Epoch: {n} optimisation losses')
                logging.info('=====> EMD Loss: %f' % loss_info['emd_loss'])
                logging.info('=====> Height map loss: %f' % loss_info['height_map_loss'])

                mpm_env.simulator.clear_ckpt()

                if args['line_search']:
                    print(f'===========> Performing line search for epoch {n}......')
                    best_loss_tmp = +np.inf
                    best_alpha = 1.0
                    best_loss_info = loss_info
                    for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
                        optimiser.lr = learning_rate * alpha
                        skill_params_np_ = optimiser.step(skill_params_np, skill_params_grad_np)
                        skill_params_np_ = np.clip(skill_params_np_, -1, 1)

                        ti.reset()
                        ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=DTYPE_TI,
                                fast_math=True, random_seed=args['seed'])

                        skill_params_ti = ti.field(dtype=DTYPE_TI, shape=5, needs_grad=True)
                        n_step_total = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

                        move_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        rotate_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        n_step_move = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        insert_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        insert_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        n_step_insert = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        push_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        push_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        n_step_push = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        rotate_delta_x_back = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        move_up_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                        n_step_return = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

                        trajectory = ti.Vector.field(n=6, dtype=DTYPE_TI, shape=horizon, needs_grad=True)

                        def reset_vars():
                            n_step_total.fill(0)
                            move_delta_x.fill(0)
                            rotate_delta_x.fill(0)
                            n_step_move.fill(0)
                            insert_delta_x.fill(0)
                            insert_delta_z.fill(0)
                            n_step_insert.fill(0)
                            push_delta_x.fill(0)
                            push_delta_z.fill(0)
                            n_step_push.fill(0)
                            rotate_delta_x_back.fill(0)
                            move_up_delta_z.fill(0)
                            n_step_return.fill(0)
                            trajectory.fill(0)

                        @ti.kernel
                        def abstraction_two_skill():
                            move_distance = skill_params_ti[0] * 0.12
                            rotate_x = skill_params_ti[1] * (np.pi / 3)  # map [-1, 1] to [-pi/3, pi/3]
                            n_step_move[None] = ti.abs(move_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                            n_step_rotate = ti.abs(rotate_x / (ANGULAR_VELOCITY * DT_GLOBAL))
                            ti.atomic_max(n_step_move[None], n_step_rotate)
                            n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                            if n_step_move_int > 0:
                                move_delta_x[None] = move_distance / n_step_move[None]
                                rotate_delta_x[None] = rotate_x / n_step_move[None]
                            n_step_total[None] += n_step_move_int

                            insert_angle = rotate_x + np.pi / 2
                            insert_distance = (skill_params_ti[2] + 1) / 2 * 0.06  # map [-1, 1] to [0, 0.06]
                            n_step_insert[None] = ti.abs(insert_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                            n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                            if n_step_insert_int > 0:
                                insert_distance_x = insert_distance * ti.cos(insert_angle)
                                insert_distance_z = insert_distance * ti.sin(insert_angle)
                                insert_delta_x[None] = insert_distance_x / n_step_insert[None]
                                insert_delta_z[None] = insert_distance_z / n_step_insert[None]
                            n_step_total[None] += n_step_insert_int

                            push_angle = (skill_params_ti[3] + 3) * np.pi / 3  # map [-1, 1] to [2*pi/3, 4*pi/3]
                            push_distance = (skill_params_ti[4] + 1) * 0.1 + 0.04  # map [-1, 1] to [0.04, 0.24]
                            n_step_push[None] = ti.abs(push_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                            n_step_push_int = ti.floor(n_step_push[None], ti.i32)
                            # print('n_step_push', n_step_push[None])
                            if n_step_push_int > 0:
                                push_distance_x = push_distance * ti.cos(push_angle)
                                # print('push_distance_x:', push_distance_x)
                                push_distance_z = push_distance * ti.sin(push_angle)
                                # print('push_distance_z:', push_distance_z)
                                push_delta_x[None] = push_distance_x / n_step_push[None]
                                push_delta_z[None] = push_distance_z / n_step_push[None]
                            n_step_total[None] += n_step_push_int

                            rotate_x_back = -rotate_x
                            n_step_rotate_back = ti.abs(rotate_x / (ANGULAR_VELOCITY * DT_GLOBAL))
                            move_up_distance = 0.1
                            n_step_move_up = ti.abs(move_up_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                            n_step_return[None] = n_step_rotate_back
                            ti.atomic_max(n_step_return[None], n_step_move_up)
                            n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                            if n_step_return_int > 0:
                                rotate_delta_x_back[None] = rotate_x_back / n_step_return[None]
                                move_up_delta_z[None] = move_up_distance / n_step_return[None]
                            n_step_total[None] += n_step_return_int

                        @ti.kernel
                        def fill_trajectory_10():
                            for k in range(1):
                                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                                half_n_step_move_int = n_step_move_int // 2
                                for j in range(half_n_step_move_int):
                                    trajectory[j][0] = move_delta_x[None]
                                    trajectory[j][3] = rotate_delta_x[None]

                        @ti.kernel
                        def fill_trajectory_11():
                            for k in range(1):
                                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                                half_n_step_move_int = n_step_move_int // 2
                                for j in range(n_step_move_int - half_n_step_move_int):
                                    index = j + half_n_step_move_int
                                    trajectory[index][0] = move_delta_x[None]
                                    trajectory[index][3] = rotate_delta_x[None]

                        @ti.kernel
                        def fill_trajectory_2():
                            for k in range(1):
                                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                                for j in range(n_step_insert_int):
                                    index = j + n_step_move_int
                                    trajectory[index][0] = insert_delta_x[None]
                                    trajectory[index][2] = -insert_delta_z[None]

                        @ti.kernel
                        def fill_trajectory_3():
                            for k in range(1):
                                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                                n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                                for j in range(n_step_push_int):
                                    index = j + n_step_move_int
                                    index = index + n_step_insert_int
                                    trajectory[index][0] = push_delta_x[None]
                                    trajectory[index][2] = push_delta_z[None]

                        @ti.kernel
                        def fill_trajectory_40():
                            for k in range(1):
                                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                                n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                                n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                                half_n_step_return_int = n_step_return_int // 2
                                for j in range(half_n_step_return_int):
                                    index = j + n_step_move_int + n_step_insert_int + n_step_push_int
                                    trajectory[index][3] = rotate_delta_x_back[None]
                                    trajectory[index][2] = move_up_delta_z[None]

                        @ti.kernel
                        def fill_trajectory_41():
                            for k in range(1):
                                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                                n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                                n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                                half_n_step_return_int = n_step_return_int // 2
                                for j in range(half_n_step_return_int - half_n_step_return_int):
                                    index = j + n_step_move_int + n_step_insert_int + n_step_push_int + half_n_step_return_int
                                    trajectory[index][3] = rotate_delta_x_back[None]
                                    trajectory[index][2] = move_up_delta_z[None]

                        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False)

                        set_parameters(mpm_env, SAND,
                                       e=best_params['E'],
                                       nu=best_params['nu'],
                                       rho=best_params['rho'],
                                       sand_friction_angle=best_params['sand_angle'])

                        """forward pass"""
                        mpm_env.set_state(init_state['state'], grad_enabled=True)
                        # prepare trajectory
                        skill_params_ti.from_numpy(skill_params_np_.copy())
                        reset_vars()
                        abstraction_two_skill()
                        fill_trajectory_10()
                        fill_trajectory_11()
                        fill_trajectory_2()
                        fill_trajectory_3()
                        fill_trajectory_40()
                        fill_trajectory_41()
                        trajectory_np = trajectory.to_numpy()
                        trajectory_length = int(n_step_total[None])
                        for i in range(trajectory_length):
                            mpm_env.step(trajectory_np[i])
                        loss_info = mpm_env.get_final_loss()

                        print(f'=====> alpha: {alpha}')
                        print(f'==> EMD Loss: %f' % loss_info["emd_loss"])
                        print(f'==> Height map loss: %f' % loss_info['height_map_loss'])
                        logging.info(f'==> EMD Loss: %f' % loss_info["emd_loss"])
                        logging.info(f'==> Height map loss:  %f' % loss_info['height_map_loss'])

                        total_loss = loss_info['height_map_loss']
                        if total_loss < best_loss_tmp:
                            best_loss_tmp = total_loss
                            best_alpha = alpha
                            best_loss_info = loss_info

                        optimiser.reverse_normaliser()
                        mpm_env.simulator.clear_ckpt()

                    print(f'=====> Best alpha: {best_alpha}, Best loss info: {best_loss_info}')
                    optimiser.lr = learning_rate * best_alpha
                    logger.add_scalar(tag='grad/best_alpha', scalar_value=best_alpha, global_step=n)

                skill_params_np = optimiser.step(skill_params_np, skill_params_grad_np)
                skill_params_np = np.clip(skill_params_np, -1, 1)

        ti.reset()
        ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=DTYPE_TI,
                fast_math=True, random_seed=args['seed'])

        skill_params_ti = ti.field(dtype=DTYPE_TI, shape=5, needs_grad=True)
        n_step_total = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

        move_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        rotate_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        n_step_move = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        insert_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        insert_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        n_step_insert = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        push_delta_x = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        push_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        n_step_push = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        rotate_delta_x_back = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        move_up_delta_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        n_step_return = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

        trajectory = ti.Vector.field(n=6, dtype=DTYPE_TI, shape=horizon, needs_grad=True)

        def reset_vars():
            n_step_total.fill(0)
            move_delta_x.fill(0)
            rotate_delta_x.fill(0)
            n_step_move.fill(0)
            insert_delta_x.fill(0)
            insert_delta_z.fill(0)
            n_step_insert.fill(0)
            push_delta_x.fill(0)
            push_delta_z.fill(0)
            n_step_push.fill(0)
            rotate_delta_x_back.fill(0)
            move_up_delta_z.fill(0)
            n_step_return.fill(0)
            trajectory.fill(0)

        @ti.kernel
        def abstraction_two_skill():
            move_distance = skill_params_ti[0] * 0.12
            rotate_x = skill_params_ti[1] * (np.pi / 3)  # map [-1, 1] to [-pi/3, pi/3]
            n_step_move[None] = ti.abs(move_distance / (LINEAR_VELOCITY * DT_GLOBAL))
            n_step_rotate = ti.abs(rotate_x / (ANGULAR_VELOCITY * DT_GLOBAL))
            ti.atomic_max(n_step_move[None], n_step_rotate)
            n_step_move_int = ti.cast(n_step_move[None], ti.i32)
            if n_step_move_int > 0:
                move_delta_x[None] = move_distance / n_step_move[None]
                rotate_delta_x[None] = rotate_x / n_step_move[None]
            n_step_total[None] += n_step_move_int

            insert_angle = rotate_x + np.pi / 2
            insert_distance = (skill_params_ti[2] + 1) / 2 * 0.06  # map [-1, 1] to [0, 0.06]
            n_step_insert[None] = ti.abs(insert_distance / (LINEAR_VELOCITY * DT_GLOBAL))
            n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
            if n_step_insert_int > 0:
                insert_distance_x = insert_distance * ti.cos(insert_angle)
                insert_distance_z = insert_distance * ti.sin(insert_angle)
                insert_delta_x[None] = insert_distance_x / n_step_insert[None]
                insert_delta_z[None] = insert_distance_z / n_step_insert[None]
            n_step_total[None] += n_step_insert_int

            push_angle = (skill_params_ti[3] + 3) * np.pi / 3  # map [-1, 1] to [2*pi/3, 4*pi/3]
            push_distance = (skill_params_ti[4] + 1) * 0.1 + 0.04  # map [-1, 1] to [0.04, 0.24]
            n_step_push[None] = ti.abs(push_distance / (LINEAR_VELOCITY * DT_GLOBAL))
            n_step_push_int = ti.floor(n_step_push[None], ti.i32)
            # print('n_step_push', n_step_push[None])
            if n_step_push_int > 0:
                push_distance_x = push_distance * ti.cos(push_angle)
                # print('push_distance_x:', push_distance_x)
                push_distance_z = push_distance * ti.sin(push_angle)
                # print('push_distance_z:', push_distance_z)
                push_delta_x[None] = push_distance_x / n_step_push[None]
                push_delta_z[None] = push_distance_z / n_step_push[None]
            n_step_total[None] += n_step_push_int

            rotate_x_back = -rotate_x
            n_step_rotate_back = ti.abs(rotate_x / (ANGULAR_VELOCITY * DT_GLOBAL))
            move_up_distance = 0.1
            n_step_move_up = ti.abs(move_up_distance / (LINEAR_VELOCITY * DT_GLOBAL))
            n_step_return[None] = n_step_rotate_back
            ti.atomic_max(n_step_return[None], n_step_move_up)
            n_step_return_int = ti.cast(n_step_return[None], ti.i32)
            if n_step_return_int > 0:
                rotate_delta_x_back[None] = rotate_x_back / n_step_return[None]
                move_up_delta_z[None] = move_up_distance / n_step_return[None]
            n_step_total[None] += n_step_return_int

        @ti.kernel
        def fill_trajectory_10():
            for k in range(1):
                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                half_n_step_move_int = n_step_move_int // 2
                for j in range(half_n_step_move_int):
                    trajectory[j][0] = move_delta_x[None]
                    trajectory[j][3] = rotate_delta_x[None]

        @ti.kernel
        def fill_trajectory_11():
            for k in range(1):
                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                half_n_step_move_int = n_step_move_int // 2
                for j in range(n_step_move_int - half_n_step_move_int):
                    index = j + half_n_step_move_int
                    trajectory[index][0] = move_delta_x[None]
                    trajectory[index][3] = rotate_delta_x[None]

        @ti.kernel
        def fill_trajectory_2():
            for k in range(1):
                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                for j in range(n_step_insert_int):
                    index = j + n_step_move_int
                    trajectory[index][0] = insert_delta_x[None]
                    trajectory[index][2] = -insert_delta_z[None]

        @ti.kernel
        def fill_trajectory_3():
            for k in range(1):
                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                for j in range(n_step_push_int):
                    index = j + n_step_move_int
                    index = index + n_step_insert_int
                    trajectory[index][0] = push_delta_x[None]
                    trajectory[index][2] = push_delta_z[None]

        @ti.kernel
        def fill_trajectory_40():
            for k in range(1):
                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                half_n_step_return_int = n_step_return_int // 2
                for j in range(half_n_step_return_int):
                    index = j + n_step_move_int + n_step_insert_int + n_step_push_int
                    trajectory[index][3] = rotate_delta_x_back[None]
                    trajectory[index][2] = move_up_delta_z[None]

        @ti.kernel
        def fill_trajectory_41():
            for k in range(1):
                n_step_move_int = ti.cast(n_step_move[None], ti.i32)
                n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
                n_step_push_int = ti.cast(n_step_push[None], ti.i32)
                n_step_return_int = ti.cast(n_step_return[None], ti.i32)
                half_n_step_return_int = n_step_return_int // 2
                for j in range(half_n_step_return_int - half_n_step_return_int):
                    index = j + n_step_move_int + n_step_insert_int + n_step_push_int + half_n_step_return_int
                    trajectory[index][3] = rotate_delta_x_back[None]
                    trajectory[index][2] = move_up_delta_z[None]

        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False)

        set_parameters(mpm_env, SAND,
                       e=best_params['E'],
                       nu=best_params['nu'],
                       rho=best_params['rho'],
                       sand_friction_angle=best_params['sand_angle'])

        """forward pass"""
        mpm_env.set_state(init_state['state'], grad_enabled=True)
        # prepare trajectory
        skill_params_ti.from_numpy(skill_params_np.copy())
        reset_vars()
        abstraction_two_skill()
        fill_trajectory_10()
        fill_trajectory_11()
        fill_trajectory_2()
        fill_trajectory_3()
        fill_trajectory_40()
        fill_trajectory_41()
        trajectory_np = trajectory.to_numpy()
        trajectory_length = int(n_step_total[None])
        for i in range(trajectory_length):
            mpm_env.step(trajectory_np[i])
        loss_info = mpm_env.get_final_loss()

        logger.add_scalar(tag='loss/EMD', scalar_value=loss_info['emd_loss'], global_step=n_epoch)
        logger.add_scalar(tag='loss/Heightmap', scalar_value=loss_info['height_map_loss'], global_step=n_epoch)

        logger.add_scalar(tag='param/0-move_distance', scalar_value=skill_params_np[0], global_step=n_epoch)
        logger.add_scalar(tag='param/1-rotate_x', scalar_value=skill_params_np[1], global_step=n_epoch)
        logger.add_scalar(tag='param/2-insert_distance', scalar_value=skill_params_np[2], global_step=n_epoch)
        logger.add_scalar(tag='param/3-push_angle', scalar_value=skill_params_np[3], global_step=n_epoch)
        logger.add_scalar(tag='param/4-push_distance', scalar_value=skill_params_np[4], global_step=n_epoch)

        np.save(os.path.join(log_dir, 'final_skill_params.npy'), np.array(skill_params_np))
        print('===========> Finished training.')
        print('====> Final EMD Loss: %f' % loss_info['emd_loss'])
        print('====> Final Height map loss: %f' % loss_info['height_map_loss'])
        print(f'====> Final skill params: {skill_params_np}')
        logging.info('===========> Finished training.')
        logging.info('====> Final EMD Loss: %f' % loss_info['emd_loss'])
        logging.info('====> Final Height map loss: %f' % loss_info['height_map_loss'])
        logging.info(f'====> Final skill params: {skill_params_np}')
        mpm_env.simulator.clear_ckpt()
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient-based skill parameter optimisation")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default="5e6", help='Particle density.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--task-id', dest='task_id', type=int, default=0, help='task id')
    parser.add_argument('--demon', dest='demon', action='store_true', default=False, help='Use demonstration')
    parser.add_argument('--line-search', dest='line_search', action='store_true', default=False, help='Use line search')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--view-demon', dest='view_demon', action='store_true', default=False, help='View demonstration')
    parser.add_argument('--lr', dest='lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--sand', dest='sand', action='store_true', default=False, help='Use sand')
    parser.add_argument('--ms', dest='multi_skill', action='store_true', default=False, help='Use multi-skill')
    arguments = vars(parser.parse_args())
    main(arguments)
