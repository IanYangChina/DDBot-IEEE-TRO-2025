import os
import yaml
import logging
import argparse
import numpy as np
import taichi as ti
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


def get_skill_params(params=None):
    if params is None:
        with open(os.path.join(script_path, '..', 'data', 'skills.yaml'), 'r') as f:
            skill_params = yaml.safe_load(f)
        params = skill_params.copy()

    return [params['insert']['xy_location'][0],
            params['insert']['xy_location'][1],
            params['insert']['distance'],
            params['insert']['angle'] / 180 * np.pi,
            params['pullout']['distance'],
            params['push-forward']['distance']]


def main(args):
    seed = args['seed']
    # np_rng = np.random.default_rng(seed=seed)
    log_dir = os.path.join(script_path, '..', 'log-abs2-adam', f'seed-{seed}')
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

    if args['backend'] == 'opengl':
        backend = ti.opengl
    elif args['backend'] == 'cuda':
        backend = ti.cuda
    elif args['backend'] == 'vulkan':
        backend = ti.vulkan
    else:
        backend = ti.cpu
    ti.reset()
    ti.init(arch=backend, device_memory_GB=args['cuda_GB'], default_fp=DTYPE_TI,
            fast_math=True, random_seed=args['seed'])

    horizon = 600

    skill_params_optim = Adam(parameters_shape=(5,),
                              cfg={'lr': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
    skill_params_ti = ti.field(dtype=DTYPE_TI, shape=5, needs_grad=True)
    n_step_total = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

    trajectory = ti.Vector.field(n=6, dtype=DTYPE_TI, shape=horizon, needs_grad=True)

    def reset_grads():
        skill_params_ti.grad.fill(0)
        trajectory.grad.fill(0)

    @ti.kernel
    def abstraction_two_skill():
        move_distance = skill_params_ti[0] * 0.12  # map [-1, 1] to [-0.12, 0.12]
        rotate_x = skill_params_ti[1] * (np.pi / 2)  # map [-1, 1] to [-pi/2, pi/2]

        n_step_move = ti.abs(ti.floor(move_distance / (LINEAR_VELOCITY * DT_GLOBAL)))
        n_step_rotate = ti.abs(ti.floor(rotate_x / (ANGULAR_VELOCITY * DT_GLOBAL)))
        n_step = n_step_move
        ti.atomic_max(n_step, n_step_rotate)
        n_step_int = int(n_step)
        if n_step_int > 0:
            move_delta_x = move_distance / n_step
            for j in range(n_step_int):
                trajectory[j][0] = move_delta_x
            rotate_delta_x = rotate_x / n_step
            for j in range(n_step_int):
                trajectory[j][3] = rotate_delta_x

        insert_angle = rotate_x + np.pi / 2
        insert_distance = (skill_params_ti[2] + 1) / 2 * 0.06  # map [-1, 1] to [0, 0.06]
        n_step_insert = ti.floor(insert_distance / (LINEAR_VELOCITY * DT_GLOBAL))
        n_step_insert_int = int(n_step_insert)
        if n_step_insert_int > 0:
            insert_distance_x = insert_distance * ti.cos(insert_angle)
            insert_distance_z = insert_distance * ti.sin(insert_angle)
            insert_delta_x = insert_distance_x / n_step_insert
            insert_delta_z = insert_distance_z / n_step_insert
            for j in range(n_step_int, n_step_int + n_step_insert_int):
                trajectory[j][0] = insert_delta_x
                trajectory[j][2] = -insert_delta_z

        push_angle = (skill_params_ti[3] + 1) * np.pi / 2  # map [-1, 1] to [0, pi]
        push_distance = (skill_params_ti[4] + 1) * 0.1  # map [-1, 1] to [0, 0.2]
        n_step_push = ti.floor(push_distance / (LINEAR_VELOCITY * DT_GLOBAL))
        n_step_push_int = int(n_step_push)
        if n_step_push_int > 0:
            push_distance_x = push_distance * ti.cos(push_angle)
            push_distance_z = push_distance * ti.sin(push_angle)
            push_delta_x = push_distance_x / n_step_push
            push_delta_z = push_distance_z / n_step_push
            for j in range(n_step_int + n_step_insert_int, n_step_int + n_step_insert_int + n_step_push_int):
                trajectory[j][0] = push_delta_x
                trajectory[j][2] = push_delta_z

        rotate_x_back = -rotate_x
        n_step_rotate_back = n_step_rotate
        move_up_distance = 0.1
        n_step_move_up = ti.floor(move_up_distance / (LINEAR_VELOCITY * DT_GLOBAL))
        n_step_return = n_step_rotate_back
        ti.atomic_max(n_step_return, n_step_move_up)
        n_step_return_int = int(n_step_return)
        if n_step_return_int > 0:
            rotate_delta_x_back = rotate_x_back / n_step_return
            move_up_delta_z = move_up_distance / n_step_return
            for j in range(n_step_int + n_step_insert_int + n_step_push_int,
                           n_step_int + n_step_insert_int + n_step_push_int + n_step_return_int):
                trajectory[j][3] = rotate_delta_x_back
                trajectory[j][5] = move_up_delta_z

        n_step_total[None] = n_step_int + n_step_insert_int + n_step_push_int + n_step_return_int

    def update_trajectory_grad(grads, length):
        for m in range(length):
            trajectory.grad[m][0] = grads[m][0]
            trajectory.grad[m][1] = grads[m][1]
            trajectory.grad[m][2] = grads[m][2]
            trajectory.grad[m][3] = grads[m][3]
            trajectory.grad[m][4] = grads[m][4]
            trajectory.grad[m][5] = grads[m][5]

    task_id = 0
    env_cfg = {
        'p_density': float(args['ptcl_density']),
        'material_id': SAND,
        'grid_scale': 1,
        'horizon': trajectory.shape[0],
        'dt_global': DT_GLOBAL,
        'n_substeps': 20,
        'agent_init_pos': (0.2, 0.2, 0.205),
        'agent_init_euler': (0, 180, 90),
    }
    loss_cfg = {
        'target_pcd_height_map_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                                   f'pcd_{task_id}_cropped_norm_z_aligned_height_map-res60-vdsize0.001.npy'),
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'sys_id_target_pcds',
                                        f'pcd_{task_id}_cropped_norm_z_aligned.ply'),
        'target_pcd_offset': [0.2, 0.2, 0],
        'down_sample_voxel_size': 0.007,
    }
    n_epoch = 150
    n_aborted_data = 0
    losses = []
    skill_params_np = np.random.uniform(-1, 1, size=5).astype(DTYPE_NP)
    env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)
    set_parameters(mpm_env, SAND, e=5e5, nu=0.3, rho=2000., sand_friction_angle=30.,
                   manipulator_friction=0.5, container_friction=0.5)
    for n in range(n_epoch):
        """forward pass"""
        mpm_env.set_state(init_state['state'], grad_enabled=True)
        # prepare trajectory
        skill_params_ti.from_numpy(skill_params_np.copy())
        trajectory.fill(0)
        abstraction_two_skill()
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
        abstraction_two_skill.grad()
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
            if np.any(np.isnan(skill_params_grad_np)) or np.any(np.isinf(skill_params_grad_np)) or np.any(np.abs(skill_params_grad_np) > 1e6):
                abort = True

        if not abort:
            num_zero_grad = 0
            for n in range(5):
                if skill_params_grad_np[n] == 0.0:
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
        else:
            skill_params_np = skill_params_optim.step(skill_params_np.copy(), skill_params_grad_np)
            skill_params_np = np.clip(skill_params_np, -1, 1)

        print(f'=====> Epoch: {n}')
        print(f'=====> Loss: {mpm_env.loss.total_loss[None]}')
        print(f'=====> Grad: {skill_params_grad_np}')
        print(f"=====> Num. aborted data so far: {n_aborted_data}")
        logging.info(f'=====> Epoch: {n}')
        logging.info(f'=====> Loss: {mpm_env.loss.total_loss[None]}')
        logging.info(f'=====> Grad: {skill_params_grad_np}')
        logging.info(f"=====> Num. aborted data so far: {n_aborted_data}")

        losses.append(loss_info['total_loss'])
        logger.add_scalar(tag='loss/EMD', scalar_value=loss_info['emd_loss'], global_step=n)
        logger.add_scalar(tag='param/0-move_distance', scalar_value=skill_params_np[0], global_step=n)
        logger.add_scalar(tag='param/1-rotate_x', scalar_value=skill_params_np[1], global_step=n)
        logger.add_scalar(tag='param/2-insert_distance', scalar_value=skill_params_np[2], global_step=n)
        logger.add_scalar(tag='param/3-push_angle', scalar_value=skill_params_np[3], global_step=n)
        logger.add_scalar(tag='param/4-push_distance', scalar_value=skill_params_np[4], global_step=n)

    mpm_env.simulator.clear_ckpt()

    logger.close()
    print('====> Finished training.')
    print('====> Final loss: ', losses[-1])
    print('====> Final skill params: ', skill_params_np)
    logging.info('====> Finished training.')
    logging.info('====> Final loss: ', losses[-1])
    logging.info('====> Final skill params: ', skill_params_np)
    np.save(os.path.join(log_dir, 'final_skill_params_.npy'), np.array(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient-based skill parameter optimisation")
    parser.add_argument('--seed', dest='seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--ptcl_d', dest='ptcl_density', type=str, default=1e7,
                        help='Particle density, use scientific notation like \'5e6\'.')
    parser.add_argument('--backend', dest='backend', default='cuda', type=str,
                        help='Computation backend: cuda, opengl, or cpu')
    parser.add_argument('--cuda_GB', dest='cuda_GB', default=5, type=int, help='preallocated GPU memory in GB')
    arguments = vars(parser.parse_args())
    main(arguments)
