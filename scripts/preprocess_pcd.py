import os
import yaml
import open3d as o3d
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', 'data')

# Create bounding box of the inner space of the real soil box
workspace_bounding_box_array = np.load(os.path.join(data_path, 'inner_soil_box_bounding_box.npy'))
workspace_bounding_box_array = o3d.utility.Vector3dVector(workspace_bounding_box_array)
workspace_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(points=workspace_bounding_box_array)
workspace_bounding_box.color = (0, 1, 0)

# Create a plane of the same height as the real table surface
table_surface = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
table_surface.translate([-0.25, 0.35, -0.11312])

# Load the camera extrinsics
with open(os.path.join(data_path, 'zivid_cam_extrinsics_fine_tuned.yml'), 'r') as f:
    cam_extrinsics = yaml.load(f, Loader=yaml.FullLoader)
transform_world_to_cam = cam_extrinsics['matrix']
transform_cam_to_world = np.linalg.inv(transform_world_to_cam)

pcd_id = 1
test = False
if not test:
    pcd_path = os.path.join(data_path, 'target_pcds', f'pcd_{pcd_id}.ply')
    pcd_in_cam_frame = o3d.io.read_point_cloud(pcd_path)
    pcd_in_world_frame = pcd_in_cam_frame.transform(transform_world_to_cam)
    crop_pcd_in_world_frame = pcd_in_world_frame.crop(workspace_bounding_box)
    points = np.asarray(crop_pcd_in_world_frame.points)
    centre = np.mean([points.min(0), points.max(0)], axis=0)
    points[:, :2] -= centre[:2]
    points[:, 2] += 0.11312
    crop_pcd_norm_z_aligned = o3d.geometry.PointCloud()
    crop_pcd_norm_z_aligned.points = o3d.utility.Vector3dVector(points)
else:
    crop_pcd_norm_z_aligned = o3d.io.read_point_cloud(os.path.join(data_path, 'target_pcds', f'pcd_{pcd_id}_cropped_norm_z_aligned.ply'))

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
cam_frame_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
cam_frame_in_world.transform(transform_world_to_cam)
soil_box_frame_in_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.010, 0.600, -0.113])

o3d.visualization.draw_geometries([
    frame,
    crop_pcd_norm_z_aligned
], width=800, height=600)

# Save the cropped point cloud
crop_pcd_norm_z_aligned_path = os.path.join(data_path, 'target_pcds', f'pcd_{pcd_id}_cropped_norm_z_aligned.ply')
o3d.io.write_point_cloud(crop_pcd_norm_z_aligned_path, crop_pcd_norm_z_aligned)