import taichi as ti


ti.reset()
ti.init(arch=ti.cuda, device_memory_GB=5, default_fp=ti.f32,
        fast_math=True)

t = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
a = ti.field(dtype=ti.f32, shape=10, needs_grad=True)
k = ti.field(dtype=ti.f32, shape=10, needs_grad=True)
n = ti.field(dtype=ti.f32, shape=10, needs_grad=True)
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.func
def divide(f, low, high):
    for i in range(20):
        k[f] = (high - low) / 2 + low
        a[f] = k[f] ** 2 - t[None]
        if a[f] < -1e-5:
            n[f] += 1
            low = k[f]
        if a[f] > 1e-5:
            n[f] += 1
            high = k[f]


@ti.kernel
def compute_loss():
    for i in range(10):
        divide(i, 0.0, 1.0)
        loss[None] += a[i] ** 2


t[None] = 0.81
a.fill(-1.)
k.fill(0.0)
n.fill(0)
loss.fill(0.0)
compute_loss()
print(k)
print(a)
print(n)
print(loss[None], '\n')

loss.grad.fill(1)
a.grad.fill(0)
k.grad.fill(0)
t.grad.fill(0)
compute_loss.grad()
print(loss.grad, a.grad, k.grad, t.grad)
