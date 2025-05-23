import taichi as ti
ti.init(arch=ti.cuda, fast_math=True)

theta = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
z = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def loss_func():
    loss[None] = ti.abs(z[None]) / theta[None]


theta[None] = 2
z[None] = -0.1

theta.grad.fill(0)
z.grad.fill(0)
loss[None] = 0
loss.grad[None] = 1

loss_func()
loss_func.grad()
print(f'loss = {loss[None]: 0.3f}, z = {z[None]}, theta.grad = {theta.grad}, z.grad = {z.grad}')
