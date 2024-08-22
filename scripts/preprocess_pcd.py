import os
import yaml
import open3d as o3d
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'data')

# Create bounding box of the inner space of the real soil box
workspace_bounding_box_array = np.array([
 [-0.104,  0.37,  -0.098],
 [ 0.168,  0.37,  -0.098],
 [ 0.168,  0.62,  -0.098],
 [-0.104,  0.62,  -0.098],
 [-0.104,  0.37,   0.095],
 [ 0.168,  0.37,   0.095],
 [ 0.168,  0.62,   0.095],
 [-0.104,  0.62,   0.095]])
workspace_bounding_box_array = o3d.utility.Vector3dVector(workspace_bounding_box_array)
workspace_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points=workspace_bounding_box_array)
workspace_bounding_box.color = (0, 1, 0)

# Create a plane of the same height as the real table surface
table_surface = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
table_surface.translate([-0.25, 0.35, -0.11920619])
soil_box_frame_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1,
                                                                            origin=[0.03006989, 0.49788191, -0.11920619])

# Load the camera extrinsics
with open(os.path.join(data_path, 'zivid_cam_extrinsics_fine_tuned.yml'), 'r') as f:
    cam_extrinsics = yaml.load(f, Loader=yaml.FullLoader)
transform_world_to_cam = cam_extrinsics['matrix']
transform_cam_to_world = np.linalg.inv(transform_world_to_cam)

task = 'sys_id'
pcd_id = 1
test = True
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
pcd_path = os.path.join(data_path, f'{task}_target_pcds', f'pcd_{pcd_id}.ply')
pcd_in_cam_frame = o3d.io.read_point_cloud(pcd_path)
# o3d.visualization.draw_geometries([
#     frame, pcd_in_cam_frame,
# ], width=800, height=600)
#
pcd_in_world_frame = pcd_in_cam_frame.transform(transform_world_to_cam)
# o3d.visualization.draw_geometries([
#     frame, table_surface, soil_box_frame_in_world,
#     workspace_bounding_box,
#     pcd_in_world_frame,
# ], width=800, height=600)
if not test:
    crop_pcd_in_world_frame = pcd_in_world_frame.crop(workspace_bounding_box)
    points = np.asarray(crop_pcd_in_world_frame.points)
    centre = np.mean([points.min(0), points.max(0)], axis=0)
    points[:, :2] -= centre[:2]
    points[:, 2] += 0.11920619
    crop_pcd_norm_z_aligned = o3d.geometry.PointCloud()
    crop_pcd_norm_z_aligned.points = o3d.utility.Vector3dVector(points)
else:
    crop_pcd_norm_z_aligned = o3d.io.read_point_cloud(os.path.join(data_path, f'{task}_target_pcds', f'pcd_{pcd_id}_cropped_norm_z_aligned.ply'))

cam_frame_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
cam_frame_in_world.transform(transform_world_to_cam)

crop_pcd_norm_z_aligned = crop_pcd_norm_z_aligned.voxel_down_sample(0.007)
print(crop_pcd_norm_z_aligned)
o3d.visualization.draw_geometries([
    frame, cam_frame_in_world,
    crop_pcd_norm_z_aligned,
], width=800, height=600)

if not test:
    # Save the cropped point cloud
    ans = input("Save the cropped point cloud? (y/n): ")
    if ans == 'y':
        crop_pcd_norm_z_aligned_path = os.path.join(data_path, f'{task}_target_pcds', f'pcd_{pcd_id}_cropped_norm_z_aligned.ply')
        o3d.io.write_point_cloud(crop_pcd_norm_z_aligned_path, crop_pcd_norm_z_aligned)