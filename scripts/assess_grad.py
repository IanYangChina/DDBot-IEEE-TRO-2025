import os
import json
import logging
import argparse
import numpy as np
import taichi as ti
from torch.utils.tensorboard import SummaryWriter
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.envs.planting_env import make_env
from doma.engine.utils.misc import set_parameters
from doma.engine.configs.macros import DTYPE_NP, SAND, DTYPE_TI
cam_cfg = {
    'pos': (0.2, 0.8, 0.7),
    'lookat': (0.2, 0.2, 0.03),
    'euler': (180+np.rad2deg(np.arctan(1.0/(0.9-0.03))), 0, 180),
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
SHOVEL_HEIGHT = 0.12
SOIL_HEIGHT = 0.085 + 0.005


def main(args):
    d_str = args['ptcl_density']
    case = f'd{d_str}'
    if args['soft_contact']:
        case += '-soft'
    if args['toi_contact']:
        case += '-toi'
    if args['use_height_map_loss']:
        case += '-hm'

    if args['grad_norm']:
        case += '-gnorm'
    elif args['grad_dy_scale']:
        case += '-gdys'
    elif args['grad_clip']:
        case += '-gclip'
    else:
        case += '-gnone'

    case += f'-res{args["res"]}'
    result_path = os.path.join(script_path, '..', 'log-grad-analysis', case)
    if args['substep']:
        result_path = os.path.join(script_path, '..', 'log-substep-grad-analysis', case)

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu

    motion_id = 0
    dt_global = 0.01
    env_cfg = {
        'p_density': float(args['ptcl_density']),
        'material_id': SAND,
        'grid_scale': 1,
        'horizon': 600,
        'dt_global': dt_global,
        'n_substeps': 20,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }
    if args['soft_contact']:
        assert not args['toi_contact']
        env_cfg['collide_type'] = 'soft'
    elif args['toi_contact']:
        assert not args['soft_contact']
        env_cfg['collide_type'] = 'toi'
    if args['grad_norm']:
        env_cfg['grad_op'] = 'normalize'
    elif args['grad_dy_scale']:
        env_cfg['grad_op'] = 'dynamic-scale'
    elif args['grad_clip']:
        env_cfg['grad_op'] = 'clip'
    else:
        env_cfg['grad_op'] = 'none'

    loss_cfg = {
        'use_height_map_loss': args['use_height_map_loss'],
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                        f'pcd_{motion_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'height_grid_res': args['res'],
    }

    E_range = (2.5e5, 1e6)
    rho_range = (1200, 2200)
    nu_range = (0.1, 0.4)
    sand_angle_range = (10, 40)
    if args['seed'] != -1:
        seeds = [args['seed']]
    else:
        seeds = [0, 1, 2, 3, 4]

    for seed in seeds:
        print(f'Running seed {seed}')
        if not args['test']:
            log_dir = os.path.join(result_path, f'seed-{seed}')
            os.makedirs(log_dir, exist_ok=True)
            logger = SummaryWriter(log_dir=log_dir)

        # Initialising parameters
        E = np.asarray(np.random.uniform(E_range[0], E_range[1]), dtype=DTYPE_NP).reshape((1,))  # Young's modulus
        nu = np.asarray(np.random.uniform(nu_range[0], nu_range[1]), dtype=DTYPE_NP).reshape((1,))  # Poisson's ratio
        rho = np.asarray(np.random.uniform(rho_range[0], rho_range[1]), dtype=DTYPE_NP).reshape((1,))  # Density
        sand_angle = np.asarray(np.random.uniform(sand_angle_range[0], sand_angle_range[1]), dtype=DTYPE_NP).reshape((1,))  # Sand friction angle

        skill_params_np = np.asarray([1.0, 0.45, 0.8, 0.0, -0.1]).astype(DTYPE_NP)

        ti.reset()
        ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=ti.f32, fast_math=True, random_seed=args['seed'])

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

        trajectory = ti.Vector.field(n=6, dtype=DTYPE_TI, shape=600, needs_grad=True)

        new_ee_tip_z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
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

            insertion_loss_ti.grad.fill(1)

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

        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg,
                                            debug_grad=False, logger=logging)
        if not args['test']:
            mpm_env.simulator.tb_logger = logger
        set_parameters(mpm_env, material_id=SAND, e=E.copy(), nu=nu.copy(), rho=rho.copy(),
                       sand_friction_angle=sand_angle.copy(),
                       manipulator_friction=0.5, container_friction=0.5)

        """forward pass"""
        mpm_env.set_state(init_state['state'], grad_enabled=True)
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

        mpm_env.simulator.trajectory_length = trajectory_length

        for i in range(trajectory_length):
            mpm_env.step(trajectory_np[i])
        loss_info = mpm_env.get_final_loss()

        """backward pass"""
        mpm_env.simulator.log_substep_grad = args['substep']
        reset_grads()
        mpm_env.reset_grad()
        mpm_env.get_final_loss_grad()
        for i in range(trajectory_length - 1, -1, -1):
            action = trajectory[i]
            mpm_env.step_grad(action=action)

            # This is a trick that prevents faulty gradient computation
            # It works for unknown reasons
            _ = mpm_env.simulator.particle_param.grad[SAND].E

            if args['test']:
                print(f"Grads at step {i}")
                mpm_env.simulator.print_grads()
            elif not args['substep']:
                mpm_env.simulator.reset_particle_grid_grad_records()
                mpm_env.simulator.get_min_max_p_grad(mpm_env.simulator.cur_substep_local)
                mpm_env.simulator.get_min_max_grid_grad(mpm_env.simulator.cur_substep_local)

                logger.add_scalar('Grad-p/x_min', mpm_env.simulator.debug_info[None].min_particle_grad_x, trajectory_length-1-i)
                logger.add_scalar('Grad-p/x_max', mpm_env.simulator.debug_info[None].max_particle_grad_x, trajectory_length-1-i)
                logger.add_scalar('Grad-p/v_min', mpm_env.simulator.debug_info[None].min_particle_grad_v, trajectory_length-1-i)
                logger.add_scalar('Grad-p/v_max', mpm_env.simulator.debug_info[None].max_particle_grad_v, trajectory_length-1-i)
                logger.add_scalar('Grad-p/F_min', mpm_env.simulator.debug_info[None].min_particle_grad_F, trajectory_length-1-i)
                logger.add_scalar('Grad-p/F_max', mpm_env.simulator.debug_info[None].max_particle_grad_F, trajectory_length-1-i)
                logger.add_scalar('Grad-grid/mass_min', mpm_env.simulator.debug_info[None].min_grid_grad_mass, trajectory_length-1-i)
                logger.add_scalar('Grad-grid/mass_max', mpm_env.simulator.debug_info[None].max_grid_grad_mass, trajectory_length-1-i)
                logger.add_scalar('Grad-grid/v_in_min', mpm_env.simulator.debug_info[None].min_grid_grad_v_in, trajectory_length-1-i)
                logger.add_scalar('Grad-grid/v_in_max', mpm_env.simulator.debug_info[None].max_grid_grad_v_in, trajectory_length-1-i)
                logger.add_scalar('Grad-grid/v_out_min', mpm_env.simulator.debug_info[None].min_grid_grad_v_out, trajectory_length-1-i)
                logger.add_scalar('Grad-grid/v_out_max', mpm_env.simulator.debug_info[None].max_grid_grad_v_out, trajectory_length-1-i)

                logger.add_scalar('Grad_param/E', mpm_env.simulator.particle_param.grad[SAND].E, trajectory_length-1-i)
                logger.add_scalar('Grad_param/nu', mpm_env.simulator.particle_param.grad[SAND].nu, trajectory_length-1-i)
                logger.add_scalar('Grad_param/rho', mpm_env.simulator.particle_param.grad[SAND].rho, trajectory_length-1-i)
                logger.add_scalar('Grad_param/sand_angle', mpm_env.simulator.system_param.grad[None].sand_friction_angle, trajectory_length-1-i)

                trajectory_grads = mpm_env.agent.get_grad(trajectory_length)
                logger.add_scalar('Grad_action/0_max', np.max(trajectory_grads[:, 0]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/0_min', np.min(trajectory_grads[:, 0]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/1_max', np.max(trajectory_grads[:, 1]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/1_min', np.min(trajectory_grads[:, 1]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/2_max', np.max(trajectory_grads[:, 2]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/2_min', np.min(trajectory_grads[:, 2]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/3_max', np.max(trajectory_grads[:, 3]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/3_min', np.min(trajectory_grads[:, 3]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/4_max', np.max(trajectory_grads[:, 4]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/4_min', np.min(trajectory_grads[:, 4]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/5_max', np.max(trajectory_grads[:, 5]), trajectory_length-1-i)
                logger.add_scalar('Grad_action/5_min', np.min(trajectory_grads[:, 5]), trajectory_length-1-i)
            else:
                if i <= (trajectory_length - 5):
                    break

        if not args['substep']:
            trajectory_grads = mpm_env.agent.get_grad(trajectory_length)
            update_trajectory_grad(trajectory_grads, trajectory_length)
            fill_trajectory_41.grad()
            fill_trajectory_40.grad()
            fill_trajectory_3.grad()
            fill_trajectory_2.grad()
            fill_trajectory_11.grad()
            fill_trajectory_10.grad()
            abstraction_two_skill.grad()
            print(f"Skill params grad: {skill_params_ti.grad}")
            if not args['test'] and not args['substep']:
                logger.add_scalar('Grad_skill/0', skill_params_ti.grad[0], 0)
                logger.add_scalar('Grad_skill/1', skill_params_ti.grad[1], 0)
                logger.add_scalar('Grad_skill/2', skill_params_ti.grad[2], 0)
                logger.add_scalar('Grad_skill/3', skill_params_ti.grad[3], 0)
                logger.add_scalar('Grad_skill/4', skill_params_ti.grad[4], 0)

            mpm_env.simulator.clear_ckpt()
        print(f'Seed {seed} done')
        if args['test']:
            exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="System identification for soil")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=5e6, help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--grad-clip', dest='grad_clip', action='store_true', default=False, help='Use gradient clipping')
    parser.add_argument('--grad-norm', dest='grad_norm', action='store_true', default=False, help='Use gradient normalisation')
    parser.add_argument('--grad-dy-scale', dest='grad_dy_scale', action='store_true', default=False, help='Use gradient dynamic scaling')
    parser.add_argument('--hm', dest='use_height_map_loss', action='store_true', default=False, help='Use height map loss')
    parser.add_argument('--soft-contact', dest='soft_contact', action='store_true', default=False, help='Use soft contact')
    parser.add_argument('--toi-contact', dest='toi_contact', action='store_true', default=False, help='Use time-of-impact contact')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str, help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    parser.add_argument('--eval', dest='eval', action='store_true', default=False, help='Evaluate the model')
    parser.add_argument('--test', dest='test', action='store_true', default=False, help='Run test')
    parser.add_argument('--substep', dest='substep', action='store_true', default=False, help='Log substep grads')
    parser.add_argument('--res', dest='res', default=60, type=int, help='Height map resolution')
    arguments = vars(parser.parse_args())
    main(arguments)
