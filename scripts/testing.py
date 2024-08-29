import taichi as ti
import numpy as np
LINEAR_VELOCITY = 0.2  # m/s
ANGULAR_VELOCITY = np.pi / 4  # rad/s
DT_GLOBAL = 0.01  # sec

ti.reset()
ti.init(arch=ti.vulkan, device_memory_GB=5, default_fp=ti.f32,
        fast_math=True, random_seed=0)

horizon = 400

skill_params_ti = ti.field(dtype=ti.f32, shape=5, needs_grad=True)
n_step_total = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

trajectory = ti.Vector.field(n=6, dtype=ti.f32, shape=horizon, needs_grad=True)
target = ti.Vector.field(n=6, dtype=ti.f32, shape=horizon, needs_grad=True)
target.fill(1)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def loss_func():
    for i in range(int(n_step_total[None])):
        for j in ti.static(range(6)):
            loss[None] += ti.abs((trajectory[i][j] - target[i][j]) ** 2)


def reset_grads():
    skill_params_ti.grad.fill(0)
    trajectory.grad.fill(0)


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

    insert_distance = (skill_params_ti[2] + 1) / 2 * 0.06  # map [-1, 1] to [0, 0.06]
    insert_angle = rotate_x + np.pi / 2
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
    n_step_push_int = ti.cast(n_step_push[None], ti.i32)
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
def fill_trajectory():
    for k in range(1):
        n_step_move_int = ti.cast(n_step_move[None], ti.i32)
        for i in range(n_step_move_int):
            trajectory[i][0] = move_delta_x[None]
            trajectory[i][3] = rotate_delta_x[None]

        n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
        for i in range(n_step_insert_int):
            index = i+n_step_move_int
            trajectory[index][0] = insert_delta_x[None]
            trajectory[index][2] = -insert_delta_z[None]

        n_step_push_int = ti.cast(n_step_push[None], ti.i32)
        for i in range(n_step_push_int):
            index = i+n_step_move_int
            index = index+n_step_insert_int
            trajectory[index][0] = push_delta_x[None]
            trajectory[index][2] = push_delta_z[None]

        n_step_return_int = ti.cast(n_step_return[None], ti.i32)
        for i in range(n_step_return_int):
            index = i+n_step_move_int+n_step_insert_int+n_step_push_int
            trajectory[index][3] = rotate_delta_x_back[None]
            trajectory[index][5] = move_up_delta_z[None]


skill_params_ti.from_numpy(np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32))
trajectory.fill(0)

abstraction_two_skill()
fill_trajectory()
total_step = int(n_step_total[None])
print(trajectory.to_numpy()[:total_step, :])
print(total_step)
loss_func()
print('loss:', loss[None])

loss.grad.fill(1)
skill_params_ti.grad.fill(0)
trajectory.grad.fill(0)

loss_func.grad()
print('trajectory.grad:', trajectory.grad.to_numpy()[:total_step, :])
print('skill_params_ti.grad:', skill_params_ti.grad.to_numpy())

fill_trajectory.grad()
print('trajectory.grad:', trajectory.grad.to_numpy()[:total_step, :])
print('skill_params_ti.grad:', skill_params_ti.grad.to_numpy())

abstraction_two_skill.grad()
print('trajectory.grad:', trajectory.grad.to_numpy()[:total_step, :])
print('skill_params_ti.grad:', skill_params_ti.grad.to_numpy())

