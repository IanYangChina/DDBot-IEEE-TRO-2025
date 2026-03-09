import os
import taichi as ti
from doma.engine.configs.macros import DTYPE_TI, DTYPE_NP
from .loss import Loss
from doma.engine.utils.mesh_ops import generate_particles_from_mesh
import open3d as o3d
import numpy as np
from scipy.optimize import linear_sum_assignment
from .emd_loss_external import compute_emd_loss_external, calculate_height_map_with_radius


@ti.data_oriented
class PRTEMDHMLosses(Loss):
    def __init__(
            self,
            matching_mat,
            target_mesh_path,
            target_prt_path,
            target_pcd_offset,
            target_hm_path,
            height_grid_size,
            height_grid_res,
            centralised_before_linear_assignment=False,
            **kwargs,
    ):
        super(PRTEMDHMLosses, self).__init__(**kwargs)
        self.matching_mat = matching_mat
        self.target_pcd_offset = np.array(target_pcd_offset).astype(DTYPE_NP)
        self.target_mesh_path = target_mesh_path
        self.target_prt_path = target_prt_path
        self.target_hm_path = target_hm_path

        self.linear_assignment_failed = False
        self.is_nan_particle = 0
        self.compute_linear_assignment = True

        self.height_grid_size = height_grid_size  # m

        self.height_grid_xy_offset = (0.25, 0.25)  # centre point of the height map
        self.height_grid_res = height_grid_res
        self.height_grid_pixel_size = self.height_grid_size / self.height_grid_res

        self.centralised_before_linear_assignment = centralised_before_linear_assignment
        self.compute_emd_loss_external_func = compute_emd_loss_external

    def build(self, sim):
        self.n_particles_matching_mat = sim.n_particles_per_mat[self.matching_mat]
        self.particle_radius = sim.bodies_i[0].p_radius

        reconstruct_target = True
        if os.path.exists(self.target_prt_path):
            self.target_particles_from_mesh_np = np.load(self.target_prt_path)
            reconstruct_target = False
            if self.target_particles_from_mesh_np.shape[0] > self.n_particles_matching_mat:
                reconstruct_target = True

        if reconstruct_target:
            # due to the linear assignment algorithm
            # num. of target particles should be smaller than num. of simulated particles for computing EMD loss
            done = False
            ptcl_d = 1e7
            while not done:
                self.target_particles_from_mesh_np = generate_particles_from_mesh(file=self.target_mesh_path,
                                                                                  voxelize_res=1080,
                                                                                  particle_density=ptcl_d,
                                                                                  pos=self.target_pcd_offset)
                if self.target_particles_from_mesh_np.shape[0] < self.n_particles_matching_mat:
                    done = True
                else:
                    ptcl_d *= 0.95
            np.save(self.target_prt_path, self.target_particles_from_mesh_np)

        self.target_particles_from_mesh_np = self.target_particles_from_mesh_np.astype(DTYPE_NP)
        self.target_particles_from_mesh_np_centre = np.mean(self.target_particles_from_mesh_np, axis=0)
        self.target_particles_from_mesh_np_centre_ti = ti.Vector(self.target_particles_from_mesh_np_centre.tolist())
        self.n_target_prt_points = self.target_particles_from_mesh_np.shape[0]
        self.target_prt_points = ti.Vector.field(3, dtype=DTYPE_TI,
                                                     shape=self.n_target_prt_points, needs_grad=False)
        self.target_prt_points.from_numpy(self.target_particles_from_mesh_np)
        self.particle_centre_ti = ti.Vector.field(3, dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.particle_centre_ti.fill(0)
        self.emd_ind_pairs_prt = ti.Vector.field(2, dtype=ti.i32, shape=self.n_target_prt_points, needs_grad=False)
        self.emd_distance_matrix_prt = ti.field(dtype=DTYPE_TI,
                                            shape=(self.n_target_prt_points, self.n_particles_matching_mat),
                                            needs_grad=False)

        self.target_obj_centre = np.mean(self.target_particles_from_mesh_np, axis=0)

        # heightmaps
        self.obj_particles = ti.Vector.field(3, dtype=DTYPE_TI,
                                             shape=self.n_particles_matching_mat, needs_grad=False)
        self.obj_particles.fill(0)
        self.obj_J = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)
        self.obj_hm = ti.field(dtype=DTYPE_TI, shape=(self.height_grid_res, self.height_grid_res), needs_grad=False)
        self.obj_hm.fill(0)
        self.target_hm = ti.field(dtype=DTYPE_TI, shape=(self.height_grid_res, self.height_grid_res), needs_grad=False)
        self.target_hm.fill(0)
        if os.path.exists(self.target_hm_path):
            self.target_hm_np = np.load(self.target_hm_path)
            self.target_hm.from_numpy(self.target_hm_np)
        else:
            calculate_height_map_with_radius(self.target_prt_points, self.target_hm,
                                             self.particle_radius,
                                             self.n_target_prt_points,
                                             self.height_grid_res,
                                             0.25,
                                             0.25,
                                             self.height_grid_size)
            self.target_hm_np = self.target_hm.to_numpy()
            np.save(self.target_hm_path, self.target_hm_np)

        self.target_obj_avg_height = np.mean(self.target_hm_np[self.target_hm_np > 1e-6])
        self.emd_loss = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=True)

        super(PRTEMDHMLosses, self).build(sim)

    def reset_grad(self):
        super(PRTEMDHMLosses, self).reset_grad()
        self.emd_loss.grad.fill(0)

    def clear_losses(self):
        self.emd_ind_pairs_prt.fill(-1)
        self.emd_distance_matrix_prt.fill(10000.0)
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

    @ti.func
    def compute_euclidean_distance(self, a, b):
        return ti.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

    @ti.kernel
    def compute_emd_distance_matrix_prt(self, f: ti.i32):
        for i in range(self.n_target_prt_points):
            for j in range(self.n_particles):
                if self.particle_mat[j] == self.matching_mat:
                    self.emd_distance_matrix_prt[i, j] = self.compute_euclidean_distance(self.particle_x[f, j],
                                                                                         self.target_prt_points[i])

    @ti.kernel
    def compute_particle_centre(self, f: ti.i32):
        for j in range(self.n_particles):
            if self.particle_mat[j] == self.matching_mat:
                self.particle_centre_ti[None] += self.particle_x[f, j] / self.n_particles_matching_mat

    @ti.kernel
    def compute_emd_distance_matrix_prt_centralised(self, f: ti.i32):
        for i in range(self.n_target_prt_points):
            for j in range(self.n_particles):
                if self.particle_mat[j] == self.matching_mat:
                    px = self.particle_x[f, j] - self.particle_centre_ti[None]
                    tx = self.target_prt_points[i] - self.target_particles_from_mesh_np_centre_ti
                    self.emd_distance_matrix_prt[i, j] = self.compute_euclidean_distance(px, tx)

    def compute_emd_distance_bijection_prt(self):
        mat = self.emd_distance_matrix_prt.to_numpy()
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
            self.emd_ind_pairs_prt.from_numpy(indexes)

    @ti.kernel
    def compute_emd_euclidean_distance_prt(self, f: ti.i32):
        for n in range(self.n_target_prt_points):
            i = self.emd_ind_pairs_prt[n][0]
            j = self.emd_ind_pairs_prt[n][1]
            d = self.compute_euclidean_distance(self.particle_x[f, j],
                                                self.target_prt_points[i])
            self.emd_loss[None] += d

    def load_linear_assignment_solution(self, ind_pairs):
        self.emd_ind_pairs_prt.from_numpy(ind_pairs)

    def get_linear_assignment_solution(self):
        return self.emd_ind_pairs_prt.to_numpy()

    def compute_emd_loss(self, f):
        if self.compute_linear_assignment:
            self.linear_assignment_failed = False
            if self.centralised_before_linear_assignment:
                self.compute_particle_centre(f)
                self.compute_emd_distance_matrix_prt_centralised(f)
            else:
                self.compute_emd_distance_matrix_prt(f)
            self.compute_emd_distance_bijection_prt()

        if not self.linear_assignment_failed:
            self.compute_emd_euclidean_distance_prt(f)

    def compute_emd_loss_grad(self, f):
        self.compute_emd_euclidean_distance_prt.grad(f)

    @ti.kernel
    def get_final_loss_kernel(self):
        self.total_loss[None] += self.emd_loss[None]

    def validate_nan_inf_particles(self, f):
        particles = self.particle_x.to_numpy()[f]
        if np.any(np.isnan(particles)) or np.any(np.isinf(particles)):
            return True
        else:
            return False

    @ti.kernel
    def get_final_particles_kernel(self, f: ti.i32):
        for j in range(self.n_particles):
            if self.particle_mat[j] == self.matching_mat:
                self.obj_particles[j] = self.particle_x[f, j]

    @ti.kernel
    def get_final_particle_j(self, f: ti.i32):
        for j in range(self.n_particles):
            if self.particle_mat[j] == self.matching_mat:
                self.obj_J[None] += self.particle_J[f, j] / self.n_particles_matching_mat

    def get_final_loss(self):
        particle_has_naninf = self.validate_nan_inf_particles(self.sim.cur_substep_local)
        if not particle_has_naninf:
            self.compute_emd_loss(self.sim.cur_substep_local)
            self.get_final_loss_kernel()
            self.get_final_particles_kernel(self.sim.cur_substep_local)
            self.get_final_particle_j(self.sim.cur_substep_local)

        particles = self.obj_particles.to_numpy()
        x = np.mean(particles[:, 0])
        y = np.mean(particles[:, 1])
        z = np.max(particles[:, 2])
        calculate_height_map_with_radius(self.obj_particles, self.obj_hm,
                                         self.particle_radius,
                                         self.n_particles_matching_mat,
                                         self.height_grid_res,
                                         0.25,
                                         0.25,
                                         self.height_grid_size)
        obj_hm_np = self.obj_hm.to_numpy()
        obj_avg_height = np.mean(obj_hm_np[obj_hm_np > 1e-6])

        loss_info = {
            'particle_has_naninf': particle_has_naninf,
            'p_radius': self.particle_radius,
            'emd_loss': self.emd_loss[None],
            'avg_emd_loss': self.emd_loss[None] / self.n_target_prt_points,
            'total_loss': self.total_loss[None],
            'particles': particles,
            'obj_hm': obj_hm_np,
            'obj_centre': np.mean(particles, axis=0),
            'obj_avg_height': obj_avg_height,
            'obj_J': self.obj_J[None],
            'target_hm': self.target_hm_np,
            'target_obj_centre': self.target_obj_centre,
            'target_obj_avg_height': self.target_obj_avg_height,
            'new_agent_pos': np.array([x, y, z], dtype=DTYPE_NP).tolist(),
        }
        n_nonzero = np.sum(self.target_hm_np > 1e-6)
        loss_info['hm_loss'] = np.sqrt(np.sum((loss_info['target_hm'] - loss_info['obj_hm']) ** 2))
        loss_info['avg_hm_loss'] = loss_info['hm_loss'] / n_nonzero
        loss_info['obj_centre_distance'] = np.linalg.norm(loss_info['obj_centre'][:-1] - loss_info['target_obj_centre'][:-1])

        return loss_info

    def get_final_loss_grad(self):
        self.get_final_loss_kernel.grad()
        self.compute_emd_loss_grad(self.sim.cur_substep_local)
