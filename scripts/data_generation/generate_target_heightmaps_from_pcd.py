import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import taichi as ti
from doma.engine.utils.misc import get_gpu_memory
import psutil
import argparse

DTYPE_NP = np.float32
DTYPE_TI = ti.f32


def run(args):
    ti.init(arch=ti.opengl,
            # offline_cache=False, log_level=ti.TRACE,
            default_fp=DTYPE_TI, default_ip=ti.i32,
            fast_math=False, random_seed=1)

    height_map_res = args['hmr']
    height_map_size = 0.24  # meter
    height_map_xy_offset = (0.2, 0.2)
    height_map_pixel_size = height_map_size / height_map_res
    height_map_pcd_target = ti.field(dtype=DTYPE_TI, shape=(height_map_res, height_map_res), needs_grad=False)
    down_sample_voxel_size = args['vds']

    @ti.func
    def from_xy_to_uv(x, y):
        u = (x - height_map_xy_offset[0]) / height_map_pixel_size + height_map_res / 2
        v = (y - height_map_xy_offset[1]) / height_map_pixel_size + height_map_res / 2
        return ti.floor(u, ti.i32), ti.floor(v, ti.i32)

    process = psutil.Process(os.getpid())
    script_path = os.path.dirname(os.path.realpath(__file__))
    script_path = os.path.join(script_path, '..')
    for data_ind in [0, 1, 2, 3, 4]:
        data_path = os.path.join(script_path, '..', 'data', 'task_target_pcds')
        # hm = np.load(os.path.join(data_path, f'target_pcd_height_map-{data_ind}-res{str(height_map_res)}-vdsize{str(down_sample_voxel_size)}.npy'))
        # plt.imshow(hm, cmap='Greys')
        # plt.show()
        # plt.close()
        # continue

        target_pcd_path = os.path.join(data_path, f'pcd_{data_ind}_cropped_norm_z_aligned.ply')
        pcd_offset = (0.2, 0.2, 0.0)

        print(f'===> CPU memory occupied before create particles/points: {process.memory_percent()} %')
        print(f'===> GPU memory before create particles/points: {get_gpu_memory()}')

        target_pcd = o3d.io.read_point_cloud(target_pcd_path) #.voxel_down_sample(voxel_size=down_sample_voxel_size)
        target_pcd_points_np = np.asarray(target_pcd.points, dtype=DTYPE_NP) + pcd_offset
        n_target_pcd_points = target_pcd_points_np.shape[0]
        print(f'===>  {n_target_pcd_points:7d} target points loaded.')
        target_pcd_points = ti.Vector.field(3, dtype=DTYPE_TI, shape=n_target_pcd_points)
        target_pcd_points.from_numpy(target_pcd_points_np)
        height_map_pcd_target.fill(0)

        @ti.kernel
        def compute_height_map_pcd():
            for i in range(n_target_pcd_points):
                u, v = from_xy_to_uv(target_pcd_points[i][0], target_pcd_points[i][1])
                ti.atomic_max(height_map_pcd_target[u, v], target_pcd_points[i][2])

        compute_height_map_pcd()
        point_id_pcd_target = ti.field(dtype=ti.i32, shape=height_map_res * height_map_res)

        @ti.kernel
        def compute_height_grid_masks_pcd():
            for i in range(n_target_pcd_points):
                u, v = from_xy_to_uv(target_pcd_points[i][0], target_pcd_points[i][1])
                if target_pcd_points[i][2] >= height_map_pcd_target[u, v]:
                    point_id_pcd_target[u * height_map_res + v] = i

        compute_height_grid_masks_pcd()
        target_pcd_points_downed = ti.Vector.field(3, dtype=DTYPE_TI, shape=height_map_res * height_map_res)

        @ti.kernel
        def gather_target_pcd_points():
            for i in range(height_map_res * height_map_res):
                point_id = point_id_pcd_target[i]
                target_pcd_points_downed[i] = target_pcd_points[point_id]

        gather_target_pcd_points()
        target_pcd_downed = o3d.geometry.PointCloud()
        target_pcd_downed.points = o3d.utility.Vector3dVector(target_pcd_points_downed.to_numpy())
        target_pcd_downed.translate(-np.array(pcd_offset))
        print(target_pcd_downed)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([frame, target_pcd_downed], width=800, height=600)
        o3d.io.write_point_cloud(os.path.join(data_path, f'pcd_{data_ind}_cropped_norm_z_aligned_res{height_map_res}.ply'),
                                 target_pcd_downed)

        # plt.imshow(height_map_pcd, cmap='YlOrBr')
        # plt.show()
        #
        # np.save(
        #     os.path.join(data_path, f'pcd_{data_ind}_cropped_norm_z_aligned_height_map-res{str(height_map_res)}.npy'), height_map_pcd)
        # print(f'height map saved as:\n'
        #       f'{os.path.join(data_path, f"pcd_{data_ind}_cropped_norm_z_aligned_height_map-res{str(height_map_res)}.npy")}')


if __name__ == '__main__':
    description = "This script generates target height maps from the fused point clouds."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--vds', dest='vds', default=0.001, type=float, help="Voxel down sample size")
    parser.add_argument('--hmr', dest='hmr', default=20, type=int, help="Height map resolution")
    arguments = vars(parser.parse_args())
    run(arguments)
