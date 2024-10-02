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
SHOVEL_HEIGHT = 0.12
SOIL_HEIGHT = 0.085


def main(args):
    seed = args['seed']
    # np_rng = np.random.default_rng(seed=seed)
    task_id = args['task_id']
    ptcl_d = arguments['ptcl_density']
    subfix = ''
    if args['use_insertion_loss']:
        subfix += '-ins'
    if args['use_height_map_loss']:
        subfix += '-hm'
    if args['demon']:
        subfix += '-demo'
    if args['mini_batch']:
        subfix += '-batch'

    log_dir = os.path.join(script_path, '..', 'log-abs2-adam', f'd{ptcl_d}-task-{task_id}{subfix}', f'seed-{seed}')
    if args['compute_grad']:
        log_dir = log_dir[:-6] + 'grads'
    grad_mean_file_name = os.path.join(log_dir, 'grad_mean.npy')
    grad_std_file_name = os.path.join(log_dir, 'grad_std.npy')
    os.makedirs(log_dir, exist_ok=True)

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

    horizon = 600
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

    if args['eval']:
        with open(os.path.join(script_path, '..', 'log-abs2-adam',
                               f'best_params-task-{task_id}{subfix}.json')) as f:
            best_skill_params = json.load(f)[ptcl_d]["Parameters"]
        skill_params_np = np.asarray([
            best_skill_params['skill_params_0'],
            best_skill_params['skill_params_1'],
            best_skill_params['skill_params_2'],
            best_skill_params['skill_params_3'],
            best_skill_params['skill_params_4']
        ]).astype(DTYPE_NP).reshape((5,))
        if args['view_demon']:
            skill_params_np = np.asarray([1.0, 0.3, 0.8, 1.0, 0.3]).astype(DTYPE_NP)
        print("===> Loaded skill params for evaluation:", skill_params_np)
    else:
        if args['demon']:
            skill_params_np = np.asarray([1.0, 0.3, 0.8, 1.0, 0.3]).astype(DTYPE_NP)
        else:
            skill_params_np = np.random.uniform(-1, 1, size=5).astype(DTYPE_NP)
            if args['zero_init']:
                skill_params_np *= 0.0

    n_epoch = 150
    n_aborted_data = 0
    emd_loss = 0.0
    height_map_loss = 0.0
    grads_to_save = []
    if args['use_height_map_loss']:
        lrs = [0.02, 0.03, 0.03, 0.01, 0.15]
        if args['demon']:
            lrs = [0.01, 0.01, 0.05, 0.06, 0.05]
    else:
        lrs = [0.05, 0.03, 0.03, 0.07, 0.07]
    skill_params_optim_0 = Adam(parameters_shape=(1,),
                                cfg={'lr': lrs[0], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
    skill_params_optim_1 = Adam(parameters_shape=(1,),
                                cfg={'lr': lrs[1], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
    skill_params_optim_2 = Adam(parameters_shape=(1,),
                                cfg={'lr': lrs[2], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
    skill_params_optim_3 = Adam(parameters_shape=(1,),
                                cfg={'lr': lrs[3], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
    skill_params_optim_4 = Adam(parameters_shape=(1,),
                                cfg={'lr': lrs[4], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})

    for n in range(n_epoch):
        grads = []
        emd_losses = []
        height_map_losses = []
        if args['mini_batch']:
            n_inner_epoch = 3
        else:
            n_inner_epoch = 1
        for m in range(n_inner_epoch):
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
            insertion_loss_ti = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

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
                insertion_loss_ti.fill(0)

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
                insertion_loss_ti.grad.fill(0)

            @ti.kernel
            def abstraction_two_skill():
                move_distance = skill_params_ti[0] * 0.12
                rotate_x = skill_params_ti[1] * (np.pi / 2)  # map [-1, 1] to [-pi/2, pi/2]
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

                push_angle = (skill_params_ti[3] + 1) * np.pi / 2  # map [-1, 1] to [0, pi]
                push_distance = (skill_params_ti[4] + 1) * 0.1  # map [-1, 1] to [0, 0.2]
                n_step_push[None] = ti.abs(push_distance / (LINEAR_VELOCITY * DT_GLOBAL))
                n_step_push_int = ti.floor(n_step_push[None], ti.i32)
                if n_step_push_int > 0:
                    push_distance_x = push_distance * ti.cos(push_angle)
                    push_distance_z = push_distance * ti.sin(push_angle)
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
                        trajectory[index][5] = move_up_delta_z[None]

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
                        trajectory[index][5] = move_up_delta_z[None]

            @ti.kernel
            def insertion_loss():
                rotate_x = skill_params_ti[1] * (np.pi / 2)
                insert_angle = rotate_x + np.pi / 2
                insert_distance = (skill_params_ti[2] + 1) / 2 * 0.06
                insert_distance_z = insert_distance * ti.sin(insert_angle)
                new_ee_tip_z = 0.205 - SHOVEL_HEIGHT * ti.cos(rotate_x) - insert_distance_z
                if new_ee_tip_z > SOIL_HEIGHT:
                    insertion_loss_ti[None] = SOIL_HEIGHT - new_ee_tip_z

            def update_trajectory_grad(tr_grads, length):
                for k in range(length):
                    trajectory.grad[k][0] = tr_grads[k][0]
                    trajectory.grad[k][1] = tr_grads[k][1]
                    trajectory.grad[k][2] = tr_grads[k][2]
                    trajectory.grad[k][3] = tr_grads[k][3]
                    trajectory.grad[k][4] = tr_grads[k][4]
                    trajectory.grad[k][5] = tr_grads[k][5]

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
            # prepare trajectory
            skill_params_ti.from_numpy(skill_params_np.copy())
            reset_vars()
            if args['use_insertion_loss']:
                insertion_loss()
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
            if args['use_insertion_loss']:
                insertion_loss.grad()

            skill_params_grad_np = skill_params_ti.grad.to_numpy()

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
                if np.any(np.isnan(skill_params_grad_np)) or np.any(np.isinf(skill_params_grad_np)) or np.any(np.abs(skill_params_grad_np) > 1e10):
                    abort = True

            if not abort:
                num_zero_grad = 0
                for h in range(5):
                    if skill_params_grad_np[h] == 0.0:
                        num_zero_grad += 1
                if num_zero_grad > 4:
                    abort = True

            if abort:
                print(f'===> [Warning] Aborting epoch: {n}')
                print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                print(f'===> [Warning] Strange loss or gradient.')
                print(f'===> [Warning] Skill params: {skill_params_np}')
                print(f'===> [Warning] Skill params grad: {skill_params_grad_np}')
                print(f'===> [Warning] Loss info: {loss_info}')
                logging.error(f'===> [Warning] Aborting epoch: {n}')
                logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
                logging.error(f'===> [Warning] Strange loss or gradient.')
                logging.error(f'===> [Warning] Skill params: {skill_params_np}')
                logging.error(f'===> [Warning] Skill params grad: {skill_params_grad_np}')
                logging.error(f'===> [Warning] Loss info: {loss_info}')
                n_aborted_data += 1
                grads.append(np.ones_like(skill_params_grad_np)*1e-6)
            else:
                grads.append(skill_params_grad_np)
                emd_losses.append(loss_info['emd_loss'])
                height_map_losses.append(loss_info['height_map_loss'])

            mpm_env.simulator.clear_ckpt()

        grad = np.mean(grads, axis=0)
        emd_loss = np.mean(emd_losses)
        height_map_loss = np.mean(height_map_losses)
        if args['compute_grad']:
            grads_to_save.append(grad)
            skill_params_np = np.random.uniform(-1, 1, size=5).astype(DTYPE_NP)
            if args['demon']:
                skill_params_np = (np.asarray([1.0, 0.3, 0.8, 1.0, 0.3]).astype(DTYPE_NP) + np.random.uniform(-1, 1, size=5).astype(DTYPE_NP) * 0.5)
                skill_params_np = np.clip(skill_params_np, -1, 1)

            print(f'=====> Epoch: {n}')
            print(f'=====> EMD Loss: {emd_loss}')
            print(f'=====> Height map loss: {height_map_loss}')
            print(f'=====> Grad: {grad}')
            print(f'=====> Skill params: {skill_params_np}')
            print(f"=====> Num. aborted data so far: {n_aborted_data}")
            logging.info(f'=====> Epoch: {n}')
            logging.info(f'=====> EMD Loss: {emd_loss}')
            logging.info(f'=====> Height map loss: {height_map_loss}')
            logging.info(f'=====> Grad: {grad}')
            logging.info(f'=====> Skill params: {skill_params_np}')
            logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")
        else:
            skill_params_np_0 = skill_params_optim_0.step(skill_params_np.copy()[0], grad[0])
            skill_params_np_1 = skill_params_optim_1.step(skill_params_np.copy()[1], grad[1])
            skill_params_np_2 = skill_params_optim_2.step(skill_params_np.copy()[2], grad[2])
            skill_params_np_3 = skill_params_optim_3.step(skill_params_np.copy()[3], grad[3])
            skill_params_np_4 = skill_params_optim_4.step(skill_params_np.copy()[4], grad[4])
            skill_params_np = np.array([skill_params_np_0,
                                        skill_params_np_1,
                                        skill_params_np_2,
                                        skill_params_np_3,
                                        skill_params_np_4]).reshape((5,))
            skill_params_np = np.clip(skill_params_np, -1, 1)

            print(f'=====> Epoch: {n}')
            print(f'=====> EMD Loss: {emd_loss}')
            print(f'=====> Height map loss: {height_map_loss}')
            print(f'=====> Grad: {grad}')
            print(f'=====> Skill params: {skill_params_np}')
            print(f"=====> Num. aborted data so far: {n_aborted_data}")
            logging.info(f'=====> Epoch: {n}')
            logging.info(f'=====> EMD Loss: {emd_loss}')
            logging.info(f'=====> Height map loss: {height_map_loss}')
            logging.info(f'=====> Grad: {grad}')
            logging.info(f'=====> Skill params: {skill_params_np}')
            logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")

            logger.add_scalar(tag='loss/EMD', scalar_value=emd_loss, global_step=n)
            logger.add_scalar(tag='loss/Heightmap', scalar_value=height_map_loss, global_step=n)
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
        np.save(os.path.join(log_dir, 'final_skill_params.npy'), np.array(skill_params_np))
        print('====> Finished training.')
        print('====> Final EMD loss: ', emd_loss)
        print('====> Final height map loss: ', height_map_loss)
        print('====> Final skill params: ', skill_params_np)
        logging.info('====> Finished training.')
        logging.info(f'====> Final EMD loss: {emd_loss}')
        logging.info(f'====> Final height map loss: {height_map_loss}')
        logging.info(f'====> Final skill params: {skill_params_np}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient-based skill parameter optimisation")
    parser.add_argument('--seed', dest='seed', type=int, default=0, help='Random seed')
    parser.add_argument('--com_grad', dest='compute_grad', action='store_true', default=False, help='Compute gradient')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default="5e6",
                        help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str,
                        help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--task-id', dest='task_id', type=int, default=0, help='task id')
    parser.add_argument('--zero-init', dest='zero_init', action='store_true', default=False, help='Initialise parameters to zero')
    parser.add_argument('--demon', dest='demon', action='store_true', default=False, help='Use demonstration')
    parser.add_argument('--mini-batch', dest='mini_batch', action='store_true', default=False, help='Use mini-batch')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--insert-loss', dest='use_insertion_loss', action='store_true', default=False, help='Use insertion loss')
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('--view-demon', dest='view_demon', action='store_true', default=False, help='View demonstration')
    arguments = vars(parser.parse_args())
    main(arguments)
