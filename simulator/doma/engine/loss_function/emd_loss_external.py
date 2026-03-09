import numpy as np
import taichi as ti
from doma.engine.configs.macros import DTYPE_TI
from scipy.optimize import linear_sum_assignment
import trimesh


def compute_emd_loss_external(x1, x2, d=None):
    n_points_1 = x1.shape[0]
    n_points_2 = x2.shape[0]
    assert n_points_2 >= n_points_1, "The first point cloud must not have more points than the second one for valid EMD result."
    x1_ti = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_points_1, needs_grad=False)
    x2_ti = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_points_2, needs_grad=False)
    x1_ti.from_numpy(x1)
    x2_ti.from_numpy(x2)
    emd_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
    emd_ind_pairs = ti.Vector.field(2, dtype=ti.i32, shape=n_points_1, needs_grad=False)
    emd_distance_matrix = ti.field(dtype=DTYPE_TI,
                                   shape=(n_points_1, n_points_2),
                                   needs_grad=False)
    emd_ind_pairs.fill(-1)
    emd_distance_matrix.fill(10000.0)
    compute_emd_distance_matrix(x1_ti, x2_ti, emd_distance_matrix)
    compute_emd_distance_bijection(emd_distance_matrix, emd_ind_pairs)
    compute_emd_euclidean_distance(emd_loss, x1_ti, x2_ti, emd_ind_pairs)
    return -emd_loss[None]


def compute_emd_loss_with_res(particles, p_radius, pcd, pcd_heightmap, res):
    n_particles = particles.shape[0]
    x1_ti = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_particles, needs_grad=False)
    x1_ti.from_numpy(particles)
    height_map_1 = ti.field(dtype=DTYPE_TI, shape=(res, res), needs_grad=False)
    height_map_1.fill(0)
    calculate_height_map(x1_ti, height_map_1, n_particles, res, 0.2, 0.2, 0.24)
    point_id_1 = ti.field(dtype=ti.i32, shape=res * res)
    point_id_1.fill(-1)
    compute_height_map_masks(x1_ti, height_map_1, n_particles, point_id_1, res, 0.2, 0.2, 0.24)

    x2_ti = ti.Vector.field(3, dtype=DTYPE_TI, shape=pcd.shape[0], needs_grad=False)
    x2_ti.from_numpy(pcd)

    emd_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
    emd_loss.fill(0)
    emd_ind_pairs = ti.Vector.field(2, dtype=ti.i32, shape=x1_ti.shape[0], needs_grad=False)
    emd_distance_matrix = ti.field(dtype=DTYPE_TI, shape=(res*res, res*res), needs_grad=False)
    emd_ind_pairs.fill(-1)
    emd_distance_matrix.fill(10000.0)
    compute_emd_distance_matrix_with_ids(x1_ti, point_id_1, x2_ti, emd_distance_matrix)
    compute_emd_distance_bijection(emd_distance_matrix, emd_ind_pairs)
    compute_emd_euclidean_distance(emd_loss, x1_ti, x2_ti, emd_ind_pairs)

    height_map_1_radius = ti.field(dtype=DTYPE_TI, shape=(res, res), needs_grad=False)
    height_map_1_radius.fill(0.0)
    calculate_height_map_with_radius(x1_ti, height_map_1_radius, p_radius, n_particles,
                                     res, 0.2, 0.2, 0.24)
    height_map_2 = ti.field(dtype=DTYPE_TI, shape=(res, res), needs_grad=False)
    height_map_2.from_numpy(pcd_heightmap)
    height_map_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
    height_map_loss.fill(0)
    compute_height_map_loss(height_map_1, height_map_2, height_map_loss)
    return emd_loss[None], height_map_loss[None], height_map_1.to_numpy(), height_map_1_radius.to_numpy()


@ti.func
def compute_euclidean_distance(a, b):
    return ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))


@ti.kernel
def compute_emd_distance_matrix(x1: ti.template(), x2: ti.template(), d_mat: ti.template()):
    for i in x1:
        for j in range(x2.shape[0]):
            d_mat[i, j] = compute_euclidean_distance(x1[i], x2[j])


@ti.kernel
def compute_emd_distance_matrix_with_ids(particles: ti.template(), id1: ti.template(), x2: ti.template(), d_mat: ti.template()):
    for ind in id1:
        i = id1[ind]
        for j in range(x2.shape[0]):
            d_mat[i, j] = compute_euclidean_distance(particles[i], x2[j])


def compute_emd_distance_bijection(d_mat, emd_ind_pairs):
    mat = d_mat.to_numpy()
    attempt = 0
    done = False
    while not done:
        attempt += 1
        try:
            ind1, ind2 = linear_sum_assignment(mat)
            indexes = np.stack((ind1, ind2), axis=-1).astype(np.int32)
            done = True
        except:
            print('Error: linear_sum_assignment failed')
            print(f'D mat shape: {mat.shape}')
            print(f'D mat NaN: {np.isnan(mat).any()}, Inf: {np.isinf(mat).any()}, Max: {np.max(mat)}, Min: {np.min(mat)}')
        if attempt > 5:
            print('Linear assignment failed for 5 times, exiting calculation.')
            done = True
    if done:
        emd_ind_pairs.from_numpy(indexes)


@ti.kernel
def compute_emd_euclidean_distance(emd_loss: ti.template(),
                                   x1: ti.template(),
                                   x2: ti.template(),
                                   emd_ind_pairs: ti.template()):
    for n in emd_ind_pairs:
        i = emd_ind_pairs[n][0]
        j = emd_ind_pairs[n][1]
        d = compute_euclidean_distance(x1[i], x2[j])
        emd_loss[None] += d


@ti.func
def from_xy_to_uv(x: DTYPE_TI, y: DTYPE_TI,
                  res: DTYPE_TI, offset_x: DTYPE_TI, offset_y: DTYPE_TI, hm_size: DTYPE_TI):
    # this does not need to be differentiable as the loss is connected to the z values of the particles/points
    u = (x - offset_x) / (hm_size / res) + res / 2
    v = (y - offset_y) / (hm_size / res) + res / 2
    return ti.floor(u, ti.i32), ti.floor(v, ti.i32)


@ti.kernel
def calculate_height_map(points: ti.template(), height_map: ti.template(), n_particles: ti.i32,
                         res: ti.i32, offset_x: DTYPE_TI, offset_y: DTYPE_TI, hm_size: DTYPE_TI):
    for p in range(n_particles):
        u, v = from_xy_to_uv(points[p][0], points[p][1], res, offset_x, offset_y, hm_size)
        ti.atomic_max(height_map[u, v], (points[p][2]))


def get_heightmap_np_with_radius(points, particle_radius, height_map_res, height_map_size,
                                 xy_offset=(0.25, 0.25), median_filter=False):
    n_cur_particles = points.shape[0]
    cur_particles_ti = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_cur_particles)
    cur_particles_ti.from_numpy(points)
    height_map = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=True)
    height_map.fill(0)

    calculate_height_map_with_radius(cur_particles_ti, height_map, particle_radius, n_cur_particles,
                                     height_map_res, xy_offset[0], xy_offset[1], height_map_size)
    if median_filter:
        height_map_out = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=True)
        height_map_out.fill(0)
        apply_median_filter(height_map, height_map_out, height_map_res)
        return height_map_out.to_numpy()
    else:
        return height_map.to_numpy()


@ti.kernel
def calculate_height_map_with_radius(points: ti.template(), height_map: ti.template(),
                                     radius: DTYPE_TI, n_particles: ti.i32,
                                     res: ti.i32, offset_x: DTYPE_TI, offset_y: DTYPE_TI, hm_size: DTYPE_TI):
    for p in range(n_particles):
        if (points[p][0]+(hm_size / res) <= (offset_x + hm_size/2) and points[p][0]-(hm_size / res) >= (offset_x - hm_size/2) and
                points[p][1]+(hm_size / res) <= (offset_y + hm_size/2) and points[p][1]-(hm_size / res) >= (offset_y - hm_size/2)):
            u, v = from_xy_to_uv(points[p][0], points[p][1], res, offset_x, offset_y, hm_size)
            u_0, v_0 = from_xy_to_uv(points[p][0] - radius,
                                     points[p][1] - radius, res, offset_x, offset_y, hm_size)
            u_1, v_1 = from_xy_to_uv(points[p][0] + radius,
                                     points[p][1] + radius, res, offset_x, offset_y, hm_size)
            ti.atomic_max(height_map[u, v], (points[p][2]))
            ti.atomic_max(height_map[u_0, v_0], (points[p][2]))
            ti.atomic_max(height_map[u_1, v_1], (points[p][2]))
            ti.atomic_max(height_map[u_0, v_1], (points[p][2]))
            ti.atomic_max(height_map[u_1, v_0], (points[p][2]))


def get_median_filtered_hm(height_map, height_map_res):
    height_map_ti = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=True)
    height_map_ti.from_numpy(height_map)
    height_map_out = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=True)
    height_map_out.fill(0)
    apply_median_filter(height_map_ti, height_map_out, height_map_res)

    return height_map_out.to_numpy()


@ti.kernel
def apply_median_filter(height_map: ti.template(), height_map_out: ti.template(), height_map_res: ti.i32):
    for i in range(1, height_map_res - 1):
        for j in range(1, height_map_res - 1):
            pixel = height_map[i - 1, j - 1] + height_map[i - 1, j] + height_map[i - 1, j + 1] + \
                    height_map[i, j - 1] + height_map[i, j] + height_map[i, j + 1] + \
                    height_map[i + 1, j - 1] + height_map[i + 1, j] + height_map[i + 1, j + 1]
            pixel = pixel / 9.0
            height_map_out[i, j] = pixel


def get_height_difference_np(cur_hm, target_hm, height_map_res, median_filter=False):
    hm_diff = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=False)
    height_map_cur_particles_ti = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=False)
    target_hm_ti = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=False)
    height_map_cur_particles_ti.from_numpy(cur_hm)
    target_hm_ti.from_numpy(target_hm)
    compute_hm_difference(hm_diff, height_map_cur_particles_ti, target_hm_ti, height_map_res)
    if median_filter:
        height_map_out = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=True)
        height_map_out.fill(0)
        apply_median_filter(hm_diff, height_map_out, height_map_res)
        return hm_diff.to_numpy()
    else:
        return hm_diff.to_numpy()

@ti.kernel
def compute_hm_difference(hm_diff: ti.template(),
                          height_map_cur_particles: ti.template(),
                          target_hm_ti: ti.template(),
                          height_map_res: ti.i32):
    for i in range(height_map_res):
        for j in range(height_map_res):
            hm_diff[i, j] = height_map_cur_particles[i, j] - target_hm_ti[i, j]


@ti.kernel
def compute_height_map_masks(points: ti.template(), height_map: ti.template(), n_particles: ti.i32,
                             p_id: ti.template(), res: ti.i32,
                             offset_x: DTYPE_TI, offset_y: DTYPE_TI, hm_size: DTYPE_TI):
    for p in range(n_particles):
        u, v = from_xy_to_uv(points[p][0], points[p][1], res, offset_x, offset_y, hm_size)
        if points[p][2] >= height_map[u, v]:
            p_id[u * res + v] = p


@ti.kernel
def compute_height_map_loss(height_map: ti.template(), target_pcd_heightmap_ti: ti.template(), height_map_loss: ti.template()):
    for i, j in target_pcd_heightmap_ti:
        d = ti.sqrt((target_pcd_heightmap_ti[i, j] - height_map[i, j]) ** 2)
        height_map_loss[None] += d


def calculate_mesh_volume(mesh_path):
    mesh = trimesh.load(mesh_path)
    triangles = mesh.triangles
    triangles_ti = ti.Vector.field(3, dtype=ti.f32, shape=(triangles.shape[0], triangles.shape[1]))
    triangles_ti.from_numpy(triangles)
    volume = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
    volume.fill(0)
    calculate_mesh_volume_ti(triangles_ti, volume, triangles.shape[0])
    return volume[None]


@ti.kernel
def calculate_mesh_volume_ti(triangles: ti.template(), volume: ti.template(), n_triangles: ti.i32):
    for t in range(n_triangles):
        v0 = triangles[t][0]
        v1 = triangles[t][1]
        v2 = triangles[t][2]
        volume_t = ti.math.dot(ti.math.cross(v0, v1), v2) / 6.0
        volume[None] += volume_t
