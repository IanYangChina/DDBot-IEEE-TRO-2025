import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=5, default_fp=ti.f32, fast_math=False)
from doma.envs import PlantingEnv, ClayEnv, SysIDEnv
from doma.assets import asset_mesh_dir

cam_cfg = {
    'pos': (0.45, 0.11, 0.55),
    'lookat': (0.11, 0.11, 0.03),
    'fov': 30,
    'lights': [{'pos': (1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
               {'pos': (1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
    'particle_radius': 0.001,
    'res': (640, 640)
}

env = PlantingEnv(ptcl_density=3e7, horizon=500, agent_cfg_file='shovel_eef.yaml',
                  has_loss=False, dt_global=0.02, n_substeps=50,
                  camera_cfg=cam_cfg,
                  agent_init_pos=(0.11, 0.01, 0.12), agent_init_euler=(30, 180, 180))
env.reset()
env.render(mode='human')
done = False
keyboard_ctrl = False
# env.mpm_env.apply_agent_action_p(np.array((1., 1., 1.)))
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
while env.t < 500:
    if keyboard_ctrl:
        key = input("ad: left right, ws: forward backward, qe: up down,"
                    "fh: rotate about y, tg: rotate about x, ry: rotate about z")
        if key == 'w':
            env.mpm_env.step(np.array([-0.1, 0., 0., 0., 0., 0.]))
        elif key == 's':
            env.mpm_env.step(np.array([0.1, 0., 0., 0., 0., 0.]))
        elif key == 'a':
            env.mpm_env.step(np.array([0., 0.1, 0., 0., 0., 0.]))
        elif key == 'd':
            env.mpm_env.step(np.array([0., -0.1, 0., 0., 0., 0.]))
        elif key == 'q':
            env.mpm_env.step(np.array([0., 0., 0.1, 0., 0., 0.]))
        elif key == 'e':
            env.mpm_env.step(np.array([0., 0., -0.1, 0., 0., 0.]))
        elif key == 'f':
            env.mpm_env.step(np.array([0., 0., 0., 0., 0.5, 0.]))
        elif key == 'h':
            env.mpm_env.step(np.array([0., 0., 0., 0., -0.5, 0.]))
        elif key == 't':
            env.mpm_env.step(np.array([0., 0., 0., 0.5, 0., 0.]))
        elif key == 'g':
            env.mpm_env.step(np.array([0., 0., 0., -0.5, 0., 0.]))
        elif key == 'r':
            env.mpm_env.step(np.array([0., 0., 0., 0., 0., 0.5]))
        elif key == 'y':
            env.mpm_env.step(np.array([0., 0., 0., 0., 0., -0.5]))
        else:
            print("Empty action.")
            env.mpm_env.step(np.array([0., 0., 0., 0., 0., 0.0]))
    else:
        if env.t <= 30:
            env.mpm_env.step(np.array([0., 0.1, 0., 0., 0., 0.0]))
        elif 30 < env.t <= 100:
            env.mpm_env.step(np.array([0., 0.0, 0.01, 0.4, 0.0, 0.0]))

    env.render(mode='human')

    if env.t == 30 or env.t == 100 or env.t == 150:
        x, x_ = env.render(mode='point_cloud')
        obj_vec = o3d.utility.Vector3dVector(x)
        obj_pcd = o3d.geometry.PointCloud(obj_vec)
        o3d.visualization.draw_geometries([frame,
                                           obj_pcd
                                           ], width=800, height=600)
        d = env.render(mode='depth_array')
        plt.imshow(d)
        plt.show()