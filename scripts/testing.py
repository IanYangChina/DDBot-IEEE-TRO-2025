import taichi as ti
import numpy as np
LINEAR_VELOCITY = 0.2  # m/s
ANGULAR_VELOCITY = np.pi / 4  # rad/s
DT_GLOBAL = 0.01  # sec

ti.reset()
ti.init(arch=ti.cuda, device_memory_GB=5, default_fp=ti.f32,
        fast_math=True, random_seed=0)

horizon = 300

skill_params_ti = ti.field(dtype=ti.f32, shape=5, needs_grad=True)
n_step_total = ti.field(dtype=ti.f32, shape=(), needs_grad=False)

trajectory = ti.Vector.field(n=6, dtype=ti.f32, shape=horizon, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def loss_func():
    for i in range(int(n_step_total[None])):
        for j in ti.static(range(6)):
            loss[None] += ((trajectory[i][j] - 5) ** 2)


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
        half_n_step_move_int = n_step_move_int//2
        for i in range(half_n_step_move_int):
            trajectory[i][0] = move_delta_x[None]
            trajectory[i][3] = rotate_delta_x[None]


@ti.kernel
def fill_trajectory_11():
    for k in range(1):
        n_step_move_int = ti.cast(n_step_move[None], ti.i32)
        half_n_step_move_int = n_step_move_int//2
        for i in range(n_step_move_int-half_n_step_move_int):
            index = i+half_n_step_move_int
            trajectory[index][0] = move_delta_x[None]
            trajectory[index][3] = rotate_delta_x[None]


@ti.kernel
def fill_trajectory_2():
    for k in range(1):
        n_step_move_int = ti.cast(n_step_move[None], ti.i32)
        n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
        for i in range(n_step_insert_int):
            index = i+n_step_move_int
            trajectory[index][0] = insert_delta_x[None]
            trajectory[index][2] = -insert_delta_z[None]


@ti.kernel
def fill_trajectory_3():
    for k in range(1):
        n_step_move_int = ti.cast(n_step_move[None], ti.i32)
        n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
        n_step_push_int = ti.cast(n_step_push[None], ti.i32)
        for i in range(n_step_push_int):
            index = i+n_step_move_int
            index = index+n_step_insert_int
            trajectory[index][0] = push_delta_x[None]
            trajectory[index][2] = push_delta_z[None]


@ti.kernel
def fill_trajectory_40():
    for k in range(1):
        n_step_move_int = ti.cast(n_step_move[None], ti.i32)
        n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
        n_step_push_int = ti.cast(n_step_push[None], ti.i32)
        n_step_return_int = ti.cast(n_step_return[None], ti.i32)
        half_n_step_return_int = n_step_return_int//2
        for i in range(half_n_step_return_int):
            index = i+n_step_move_int+n_step_insert_int+n_step_push_int
            trajectory[index][3] = rotate_delta_x_back[None]
            trajectory[index][5] = move_up_delta_z[None]


@ti.kernel
def fill_trajectory_41():
    for k in range(1):
        n_step_move_int = ti.cast(n_step_move[None], ti.i32)
        n_step_insert_int = ti.cast(n_step_insert[None], ti.i32)
        n_step_push_int = ti.cast(n_step_push[None], ti.i32)
        n_step_return_int = ti.cast(n_step_return[None], ti.i32)
        half_n_step_return_int = n_step_return_int//2
        for i in range(n_step_return_int-half_n_step_return_int):
            index = i+n_step_move_int+n_step_insert_int+n_step_push_int+half_n_step_return_int
            trajectory[index][3] = rotate_delta_x_back[None]
            trajectory[index][5] = move_up_delta_z[None]


for c in range(200):
    skill_params_np = np.asarray([1.0, 0.3, 0.8, 1.0, 0.3]).astype(np.float32) + np.random.uniform(-1, 1, size=5).astype(
        np.float32) * 0.5
    skill_params_np = np.clip(skill_params_np, -1, 1)
    skill_params_ti.from_numpy(skill_params_np)
    reset_vars()

    abstraction_two_skill()
    # print('n_step_total:', n_step_total[None])
    # print('n_step_move:', n_step_move[None])
    # print('n_step_insert:', n_step_insert[None])
    # print('n_step_push:', n_step_push[None])
    # print('n_step_return:', n_step_return[None])
    fill_trajectory_10()
    fill_trajectory_11()
    fill_trajectory_2()
    fill_trajectory_3()
    fill_trajectory_40()
    fill_trajectory_41()
    total_step = int(n_step_total[None])
    print('total_step:', total_step)
    # loss_func()
    # print('loss:', loss[None])
    #
    # loss.grad.fill(1)
    # skill_params_ti.grad.fill(0)
    # trajectory.grad.fill(0)
    # move_delta_x.grad.fill(0)
    # rotate_delta_x.grad.fill(0)
    # n_step_move.grad.fill(0)
    # insert_delta_x.grad.fill(0)
    # insert_delta_z.grad.fill(0)
    # n_step_insert.grad.fill(0)
    # push_delta_x.grad.fill(0)
    # push_delta_z.grad.fill(0)
    # n_step_push.grad.fill(0)
    # rotate_delta_x_back.grad.fill(0)
    # move_up_delta_z.grad.fill(0)
    # n_step_return.grad.fill(0)
    #
    # loss_func.grad()
    # fill_trajectory_41.grad()
    # fill_trajectory_40.grad()
    # fill_trajectory_3.grad()
    # fill_trajectory_2.grad()
    # fill_trajectory_11.grad()
    # fill_trajectory_10.grad()
    # abstraction_two_skill.grad()
    # print('skill_params_ti.grad:', skill_params_ti.grad.to_numpy())
    #
