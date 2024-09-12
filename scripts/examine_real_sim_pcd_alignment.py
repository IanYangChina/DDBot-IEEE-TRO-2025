import os
import open3d as o3d
import numpy as np
import taichi as ti
script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'data')

ti.init(arch=ti.cuda, device_memory_GB=5, default_fp=ti.f32, fast_math=False)
from doma.envs import PlantingEnv

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

env = PlantingEnv(ptcl_density=3e6, horizon=500, agent_cfg_file='shovel_eef.yaml',
                  has_loss=False, dt_global=0.01, n_substeps=50,
                  camera_cfg=cam_cfg,
                  agent_init_pos=(0.4, 0.2, 0.2), agent_init_euler=(0, 180, 90))
env.reset()

env.mpm_env.step(np.array([0., 0., 0., 0., 0., 0.]))

x = env.render(mode='point_cloud')
sim_particles = o3d.utility.Vector3dVector(x)
sim_particles = o3d.geometry.PointCloud(sim_particles).paint_uniform_color([1.0, 0.0, 0.0])

pcd_id = 0
pcd_path = os.path.join(data_path, 'task_target_pcds', f'pcd_{pcd_id}_cropped_norm_z_aligned.ply')
pcd_in_cam_frame = o3d.io.read_point_cloud(pcd_path)
pcd_in_cam_frame.translate([0.2, 0.2, 0.0])
pcd_in_cam_frame.paint_uniform_color([0.0, 0.0, 1.0])

world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

o3d.visualization.draw_geometries([
    world_frame, sim_particles, pcd_in_cam_frame
], width=800, height=600)
