import os
import yaml
import logging
import argparse
import numpy as np
import taichi as ti
from torch.utils.tensorboard import SummaryWriter
from time import time
from scipy.spatial.transform import Rotation
script_path = os.path.dirname(os.path.realpath(__file__))

from doma.optimiser.adam import Adam, GD
from doma.envs.planting_env import make_env
cam_cfg = {
    'pos': (-0.2, -0.2, 0.7),
    'lookat': (0.2, 0.2, 0.03),
    'fov': 30,
    'lights': [{'pos': (-1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
               {'pos': (-1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
               {'pos': (-1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
    'particle_radius': 0.001,
    'res': (640, 640)
}
SOIL_HEIGHT = 0.095
LINEAR_VELOCITY = 0.1  # m/s
ANGULAR_VELOCITY = 0.5  # rad/s
DT_GLOBAL = 0.02  # sec


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
    log_dir = os.path.join(script_path, '..', 'log-diff_skill')
    os.makedirs(log_p_dir, exist_ok=True)
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

    horizon = args['horizon']

    skill_params_np = np.array(get_skill_params())
    skill_params_optim = Adam(parameters_shape=skill_params_np.shape,
                              cfg={'lr': training_config['lr_manipulator_friction'], 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-8})
    skill_params_ti = ti.Vector.field(n=6, dtype=ti.f32, shape=(), needs_grad=True)
    skill_params_ti[None] = skill_params_np.copy()

    init_pose = ti.Vector.field(n=6, dtype=ti.f32, shape=(), needs_grad=True)
    insert_n_timestep = ti.field(ti.i32, shape=(), needs_grad=True)
    insert_delta_x = ti.field(ti.f32, shape=(), needs_grad=True)
    insert_delta_z = ti.field(ti.f32, shape=(), needs_grad=True)
    pullout_n_timestep = ti.field(ti.i32, shape=(), needs_grad=True)
    pullout_delta_x = ti.field(ti.f32, shape=(), needs_grad=True)
    pullout_delta_z = ti.field(ti.f32, shape=(), needs_grad=True)
    push_forward_n_timestep = ti.field(ti.i32, shape=(), needs_grad=True)
    push_forward_delta_x = ti.field(ti.f32, shape=(), needs_grad=True)

    trajectory = ti.Vector.field(n=6, dtype=ti.f32, shape=horizon, needs_grad=True)
    trajectory.fill(0)

    def reset_grads():
        skill_params_ti.grad.fill(1)
        init_pose.grad.fill(1)
        insert_n_timestep.grad.fill(1)
        insert_delta_x.grad.fill(1)
        insert_delta_z.grad.fill(1)
        pullout_n_timestep.grad.fill(1)
        pullout_delta_x.grad.fill(1)
        pullout_delta_z.grad.fill(1)
        push_forward_n_timestep.grad.fill(1)
        push_forward_delta_x.grad.fill(1)
        trajectory.grad.fill(1)

    @ti.kernel
    def calculate_skill_parameters():
        # set the initial pose of the end effector
        init_pose[None] = [skill_params_ti[None][0], skill_params_ti[None][1], SOIL_HEIGHT,
                           0, 180+skill_params_ti[None][3], 90]
        # insert
        insert_distance = skill_params_ti[None][2]
        insert_n_timestep[None] = ti.floor(
            ((insert_distance / LINEAR_VELOCITY) / DT_GLOBAL), ti.i32)
        insert_angle = skill_params_ti[None][3]
        insert_delta_x[None] = -insert_distance * ti.sin(insert_angle) / insert_n_timestep[None]
        insert_delta_z[None] = -insert_distance * ti.cos(insert_angle) / insert_n_timestep[None]
        # pullout
        pullout_distance = skill_params_ti[None][4]
        pullout_n_timestep[None] = ti.floor(
            ((pullout_distance / LINEAR_VELOCITY) / DT_GLOBAL), ti.i32)
        pullout_delta_x[None] = pullout_distance * ti.sin(insert_angle) / pullout_n_timestep[None]
        pullout_delta_z[None] = pullout_distance * ti.cos(insert_angle) / pullout_n_timestep[None]
        # push-forward
        push_forward_distance = skill_params_ti[None][5]
        push_forward_n_timestep[None] = ti.floor(
            ((push_forward_distance / LINEAR_VELOCITY) / DT_GLOBAL), ti.i32)
        push_forward_delta_x[None] = -push_forward_distance / push_forward_n_timestep[None]

    @ti.kernel
    def fill_trajectory():
        for n in ti.static(range(horizon)):
            if n < insert_n_timestep[None]:
                trajectory[n][0] = insert_delta_x[None]
                trajectory[n][2] = insert_delta_z[None]
            elif n < insert_n_timestep[None] + pullout_n_timestep[None]:
                trajectory[n][0] = pullout_delta_x[None]
                trajectory[n][2] = pullout_delta_z[None]
            elif n < insert_n_timestep[None] + pullout_n_timestep[None] + push_forward_n_timestep[None]:
                trajectory[n][0] = push_forward_delta_x[None]
            else:
                pass

    def update_trajectory_grad(grads, trajectory_length):
        for n in range(trajectory_length):
            trajectory.grad[n][0] = grads[n][0]
            trajectory.grad[n][1] = grads[n][1]
            trajectory.grad[n][2] = grads[n][2]
            trajectory.grad[n][3] = grads[n][3]
            trajectory.grad[n][4] = grads[n][4]
            trajectory.grad[n][5] = grads[n][5]

    env_cfg = {
        'p_density': arguments['ptcl_density'],
        'horizon': horizon,
        'dt_global': DT_GLOBAL,
        'n_substeps': 50,
        'agent_init_euler': agent_init_euler,
    }
    loss_cfg = {
        'target_pcd_path': os.path.join(script_path, '..', 'data', 'target_pcds', 'test_file.ply'),
        'target_pcd_offset': [0, 0, 0],
        'down_sample_voxel_size': arguments['down_sample_voxel_size'],
    }
    n_epoch = arguments['n_epoch']
    n_aborted_data = 0
    losses = []
    for n in range(n_epoch):
        ti.reset()
        ti.init(arch=backend, device_memory_GB=15, default_fp=ti.f32, fast_math=True, random_seed=arguments['seed'])
        env, mpm_env, init_state = make_env(env_cfg, loss_cfg, cam_cfg=cam_cfg, debug_grad=False, logger=logging)
        """forward pass"""
        mpm_env.set_state(init_state['state'], grad_enabled=True)
        # prepare trajectory
        calculate_skill_parameters()
        mpm_env.apply_agent_action_p(init_pose.to_numpy())
        fill_trajectory()
        trajectory_np = trajectory.to_numpy()
        trajectory_length = int(insert_n_timestep[None] + pullout_n_timestep[None] + push_forward_n_timestep[None])
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
        mpm_env.apply_agent_action_p_grad(init_pose.to_numpy())
        trajectory_grads = mpm_env.agent.get_grad(trajectory_length)
        update_trajectory_grad(trajectory_grads, trajectory_length)
        fill_trajectory.grad()
        calculate_skill_parameters.grad()
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
            for n in range(6):
                if skill_params_grad_np[n] == 0.0:
                    num_zero_grad += 1
            if num_zero_grad > 4:
                abort = True

        if abort:
            print(f'===> [Warning] Aborting epoch: {epoch}')
            print(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
            print(f'===> [Warning] Strange loss or gradient.')
            print(f'===> [Warning] Skill params: {skill_params_np}')
            print(f'===> [Warning] Skill params grad: {skill_params_grad_np}')
            print(f'===> [Warning] Loss info: {loss_info}')
            logging.error(f'===> [Warning] Aborting epoch: {epoch}')
            logging.error(f'===> [Warning] Particle has nan or inf: {particle_has_naninf}')
            logging.error(f'===> [Warning] Strange loss or gradient.')
            logging.error(f'===> [Warning] Skill params: {skill_params_np}')
            logging.error(f'===> [Warning] Skill params grad: {skill_params_grad_np}')
            logging.error(f'===> [Warning] Loss info: {loss_info}')
            n_aborted_data += 1
        else:
            skill_params_np = skill_params_optim.step(skill_params_np.copy(), skill_params_grad_np)

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
        logger.add_scalar(tag='param/insert_x', scalar_value=skill_params_np[0], global_step=n)
        logger.add_scalar(tag='param/insert_y', scalar_value=skill_params_np[1], global_step=n)
        logger.add_scalar(tag='param/insert_distance', scalar_value=skill_params_np[2], global_step=n)
        logger.add_scalar(tag='param/insert_angle', scalar_value=skill_params_np[3], global_step=n)
        logger.add_scalar(tag='param/pullout_distance', scalar_value=skill_params_np[4], global_step=n)
        logger.add_scalar(tag='param/push_forward_distance', scalar_value=skill_params_np[5], global_step=n)

        mpm_env.simulator.clear_ckpt()

    logger.close()
    print('====> Finished training.')
    print('====> Final loss: ', losses[-1])
    print('====> Final skill params: ', skill_params_np)
    loggin.info('====> Finished training.')
    loggin.info('====> Final loss: ', losses[-1])
    loggin.info('====> Final skill params: ', skill_params_np)
    np.save(os.path.join(log_dir, 'final_skill_params_.npy'), np.array(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill', type=str, default='insert', help='Skill to be executed')
    arguments = vars(parser.parse_args())
    main(arguments)
