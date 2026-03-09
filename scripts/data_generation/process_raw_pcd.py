import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
import yaml
import time
import open3d as o3d
import numpy as np
from paths import data_calibration_path, render_output_dir, result_dir_from_legacy_log

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
data_path = os.path.join(script_path, '..', 'data')

# Create bounding box of the inner space of the real soil box
workspace_bounding_box_array = np.array([
 [-0.117,  0.445, 0.01],
 [0.152,  0.445, 0.01],
 [0.152,  0.7, 0.01],
 [-0.117,  0.7, 0.01],
 [-0.117,  0.445, 0.15],
 [0.152,  0.445, 0.15],
 [0.152,  0.7, 0.15],
 [-0.117,  0.7, 0.15]])
workspace_bounding_box_array = o3d.utility.Vector3dVector(workspace_bounding_box_array)
workspace_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points=workspace_bounding_box_array)
workspace_bounding_box.color = (0, 1, 0)

# Create a plane of the same height as the real table surface
table_surface = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
table_surface.translate([-0.25, 0.35, -0.0011884])
soil_box_frame_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,
                                                                            origin=[0.03006989, 0.49788191, -0.11920619])

# Load the camera extrinsics
transform_world_to_cam = np.load(data_calibration_path('cam_extrinsics_fine_tuned.npy'))
transform_cam_to_world = np.linalg.inv(transform_world_to_cam)

task = 'task'
mat = ''
test = False
pcd_folder_path = os.path.join(data_path, f'{task}_target_pcds{mat}')
folder = result_dir_from_legacy_log(f'log-abs2{mat}')
saving_path = render_output_dir(f'abs2{mat}')
dirs = os.listdir(folder)
# for pcd_id in [0, 1, 2]:
for data_path in dirs:
    pcd_id = 0
    pcd_folder_path = os.path.join(folder, data_path)
    if 'task-1' not in pcd_folder_path:
        continue
    target_pcd_path = os.path.join(pcd_folder_path, f'pcd_{pcd_id}.ply')
    if not os.path.isfile(target_pcd_path):
        continue
    print(f'Processing {target_pcd_path}')
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    pcd_path = os.path.join(target_pcd_path)
    pcd_in_cam_frame = o3d.io.read_point_cloud(pcd_path)
    pcd_in_world_frame = pcd_in_cam_frame.transform(transform_world_to_cam)
    o3d.visualization.draw_geometries([
        frame, table_surface, soil_box_frame_in_world,
        workspace_bounding_box,
        pcd_in_world_frame,
    ], width=800, height=600)
    if not test:
        crop_pcd_in_world_frame = pcd_in_world_frame.crop(workspace_bounding_box)
        # o3d.visualization.draw_geometries([
        #     frame,
        #     crop_pcd_in_world_frame,
        # ], width=800, height=600)
        points = np.asarray(crop_pcd_in_world_frame.points)
        centre = np.array([0.03, 0.58])
        points[:, :2] -= centre[:2]
        points[:, 2] += 0.0011884
        crop_pcd_norm_z_aligned = o3d.geometry.PointCloud()
        crop_pcd_norm_z_aligned.points = o3d.utility.Vector3dVector(points)
    else:
        crop_pcd_norm_z_aligned = o3d.io.read_point_cloud(
            os.path.join(pcd_folder_path, f'pcd_{pcd_id}_cropped_norm_z_aligned.ply'))

    print(crop_pcd_norm_z_aligned)
    o3d.visualization.draw_geometries([
        # frame,
        crop_pcd_norm_z_aligned,
    ], width=800, height=600)

    if not test:
        # Save the cropped point cloud
        crop_pcd_norm_z_aligned_path = os.path.join(
            saving_path, data_path, f'pcd_{pcd_id}_cropped_norm_z_aligned.ply')
        o3d.io.write_point_cloud(
            crop_pcd_norm_z_aligned_path, crop_pcd_norm_z_aligned)
