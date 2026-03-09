import taichi as ti
from doma.engine.configs.macros import DTYPE_TI, DTYPE_NP
from .loss import Loss
import open3d as o3d
import numpy as np
from scipy.optimize import linear_sum_assignment
from .emd_loss_external import compute_emd_loss_external


@ti.data_oriented
class PointCloudEMDHMLosses(Loss):
    def __init__(
            self,
            use_height_map_loss,
            matching_mat,
            target_pcd_path,
            target_pcd_offset,
            height_grid_size,
            height_grid_res,
            **kwargs,
    ):
        super(PointCloudEMDHMLosses, self).__init__(**kwargs)
        self.use_height_map_loss = use_height_map_loss
        self.matching_mat = matching_mat
        self.target_pcd_path = target_pcd_path[:-4] + f'_res{height_grid_res}.ply'
        self.target_pcd_offset = np.array(target_pcd_offset).astype(DTYPE_NP)
        self.target_heightmap_path = target_pcd_path[:-4] + f'_height_map-res{height_grid_res}.npy'

        self.linear_assignment_failed = False
        self.is_nan_particle = 0

        self.height_grid_size = height_grid_size  # m

        self.height_grid_xy_offset = (self.target_pcd_offset[0], self.target_pcd_offset[1])  # centre point of the height map
        self.height_grid_res = height_grid_res
        self.height_grid_pixel_size = self.height_grid_size / self.height_grid_res

        self.compute_emd_loss_external_func = compute_emd_loss_external

    def build(self, sim):
        self.n_particles_matching_mat = sim.n_particles_per_mat[self.matching_mat]
        self.particle_radius = sim.bodies_i[0].p_radius

        target_pcd_original = o3d.io.read_point_cloud(self.target_pcd_path)
        self.target_pcd_points_np = np.asarray(target_pcd_original.points, dtype=DTYPE_NP) + self.target_pcd_offset
        self.n_target_pcd_original_points = self.target_pcd_points_np.shape[0]
        self.target_pcd_points = ti.Vector.field(3, dtype=DTYPE_TI,
                                                 shape=self.n_target_pcd_original_points, needs_grad=False)
        self.target_pcd_points.from_numpy(self.target_pcd_points_np)

        self.height_map_pcd_target = ti.field(dtype=DTYPE_TI,
                                              shape=(self.height_grid_res, self.height_grid_res), needs_grad=False)
        self.height_map_pcd_target.from_numpy(np.load(self.target_heightmap_path).astype(DTYPE_NP))
        self.height_map_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.height_map = ti.field(dtype=ti.f32, shape=(self.height_grid_res, self.height_grid_res), needs_grad=True)

        self.height_grid = ti.field(dtype=ti.f32, shape=(self.height_grid_res, self.height_grid_res), needs_grad=True)
        self.particle_id = ti.field(dtype=ti.i32, shape=self.height_grid_res*self.height_grid_res, needs_grad=False)
        self.surface_particles = ti.Vector.field(3, dtype=ti.f32,
                                                 shape=self.height_grid_res*self.height_grid_res, needs_grad=True)

        self.emd_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)
        self.emd_ind_pairs = ti.Vector.field(2, dtype=ti.i32, shape=self.height_grid_res*self.height_grid_res, needs_grad=False)
        self.emd_distance_matrix = ti.field(dtype=DTYPE_TI,
                                            shape=(self.height_grid_res*self.height_grid_res,
                                                   self.height_grid_res*self.height_grid_res),
                                            needs_grad=False)

        super(PointCloudEMDHMLosses, self).build(sim)

    def reset_grad(self):
        super(PointCloudEMDHMLosses, self).reset_grad()
        self.emd_loss.grad.fill(0)
        self.height_map_loss.grad.fill(0)

    @ti.kernel
    def clear_losses(self):
        self.height_map.fill(0)
        self.height_grid.fill(0)
        self.particle_id.fill(-1)
        self.emd_ind_pairs.fill(-1)
        self.emd_distance_matrix.fill(10000.0)
        self.height_map_loss.fill(0)
        self.emd_loss.fill(0)
        self.emd_loss.grad.fill(0)

    def compute_step_loss(self, s, f):
        pass

    def compute_step_loss_grad(self, s, f):
        pass

    """EMD loss"""
    @ti.func
    def from_xy_to_uv(self, x: DTYPE_TI, y: DTYPE_TI):
        # this does not need to be differentiable as the loss is connected to the z values of the particles/points
        u = (x - self.height_grid_xy_offset[0]) / self.height_grid_pixel_size + self.height_grid_res / 2
        v = (y - self.height_grid_xy_offset[1]) / self.height_grid_pixel_size + self.height_grid_res / 2
        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

    @ti.kernel
    def calculate_height_map(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                u, v = self.from_xy_to_uv(self.particle_x[f, p][0], self.particle_x[f, p][1])
                u_0, v_0 = self.from_xy_to_uv(self.particle_x[f, p][0] - self.particle_radius,
                                              self.particle_x[f, p][1] - self.particle_radius)
                u_1, v_1 = self.from_xy_to_uv(self.particle_x[f, p][0] + self.particle_radius,
                                              self.particle_x[f, p][1] + self.particle_radius)
                ti.atomic_max(self.height_map[u, v], (self.particle_x[f, p][2]))
                ti.atomic_max(self.height_map[u_0, v_0], (self.particle_x[f, p][2]))
                ti.atomic_max(self.height_map[u_1, v_1], (self.particle_x[f, p][2]))
                ti.atomic_max(self.height_map[u_0, v_1], (self.particle_x[f, p][2]))
                ti.atomic_max(self.height_map[u_1, v_0], (self.particle_x[f, p][2]))

    @ti.kernel
    def calculate_height_grid(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                u, v = self.from_xy_to_uv(self.particle_x[f, p][0], self.particle_x[f, p][1])
                ti.atomic_max(self.height_grid[u, v], (self.particle_x[f, p][2]))

    @ti.kernel
    def compute_height_grid_masks(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_mat[p] == self.matching_mat:
                u, v = self.from_xy_to_uv(self.particle_x[f, p][0], self.particle_x[f, p][1])
                if self.particle_x[f, p][2] >= self.height_grid[u, v]:
                    self.particle_id[u * self.height_grid_res + v] = p

    @ti.kernel
    def gather_surface_particles(self, f: ti.i32):
        for i in range(self.height_grid_res*self.height_grid_res):
            p = self.particle_id[i]
            if p != -1:
                self.surface_particles[i] = self.particle_x[f, p]

    @ti.func
    def compute_euclidean_distance(self, a, b):
        return ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

    @ti.kernel
    def compute_emd_distance_matrix(self, f: ti.i32):
        for i in range(self.height_grid_res * self.height_grid_res):
            for j in range(self.height_grid_res * self.height_grid_res):
                self.emd_distance_matrix[i, j] = self.compute_euclidean_distance(self.surface_particles[i],
                                                                                 self.target_pcd_points[j])

    def compute_emd_distance_bijection(self):
        mat = self.emd_distance_matrix.to_numpy()
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
                print(
                    f'D mat NaN: {np.isnan(mat).any()}, Inf: {np.isinf(mat).any()}, Max: {np.max(mat)}, Min: {np.min(mat)}')

            if attempt > 5:
                print('Linear assignment failed for 5 times, exiting calculation.')
                if self.logger is not None:
                    self.logger.error('Linear assignment failed for 5 times, exiting calculation.')
                done = True
                self.linear_assignment_failed = True
        if done:
            self.emd_ind_pairs.from_numpy(indexes)

    @ti.kernel
    def compute_emd_euclidean_distance(self, f: ti.i32):
        for n in range(self.height_grid_res*self.height_grid_res):
            i = self.emd_ind_pairs[n][0]
            p = self.particle_id[i]
            j = self.emd_ind_pairs[n][1]
            d = self.compute_euclidean_distance(self.particle_x[f, p],
                                                self.target_pcd_points[j])
            self.emd_loss[None] += d

    @ti.kernel
    def compute_height_map_loss(self):
        for i, j in self.height_map_pcd_target:
            d = ti.sqrt((self.height_map_pcd_target[i, j] - self.height_map[i, j]) ** 2)
            self.height_map_loss[None] += d

    def compute_emd_loss(self, f):
        self.calculate_height_grid(f)
        self.compute_height_grid_masks(f)
        self.gather_surface_particles(f)
        self.compute_emd_distance_matrix(f)

        self.linear_assignment_failed = False
        self.compute_emd_distance_bijection()
        if not self.linear_assignment_failed:
            self.compute_emd_euclidean_distance(f)

    def compute_emd_loss_grad(self, f):
        self.compute_emd_euclidean_distance.grad(f)

    @ti.kernel
    def get_final_loss_kernel(self):
        if ti.static(self.use_height_map_loss):
            self.total_loss[None] += self.height_map_loss[None]
        else:
            # self.emd_loss[None] /= self.n_target_pcd_points
            self.total_loss[None] += self.emd_loss[None]

    def validate_nan_inf_particles(self, f):
        particles = self.particle_x.to_numpy()[f]
        if np.any(np.isnan(particles)) or np.any(np.isinf(particles)):
            return True
        else:
            return False

    def get_final_loss(self):
        particle_has_naninf = self.validate_nan_inf_particles(self.sim.cur_substep_local)
        if not particle_has_naninf:
            self.clear_losses()
            self.calculate_height_map(self.sim.cur_substep_local)
            self.compute_height_map_loss()
            self.compute_emd_loss(self.sim.cur_substep_local)

            self.get_final_loss_kernel()

        loss_info = {
            # 'height_map': self.height_map.to_numpy(),
            # 'height_map_target': self.height_map_pcd_target.to_numpy(),
            'particle_has_naninf': particle_has_naninf,
            'height_map_loss': self.height_map_loss[None],
            'emd_loss': self.emd_loss[None],
            'total_loss': self.total_loss[None],
        }
        return loss_info

    def get_final_loss_grad(self):
        self.get_final_loss_kernel.grad()
        if self.use_height_map_loss:
            self.compute_height_map_loss.grad()
            self.calculate_height_map.grad(self.sim.cur_substep_local)
        else:
            self.compute_emd_loss_grad(self.sim.cur_substep_local)
