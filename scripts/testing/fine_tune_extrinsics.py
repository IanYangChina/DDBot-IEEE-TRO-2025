import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
import yaml
import open3d as o3d
import numpy as np
np.printoptions(suppress=True, precision=4)
from copy import deepcopy as dcp
from paths import system_identification_target_dir, data_calibration_path

# matrix = [
#     [-9.98126040e-01, 1.01789242e-03, -6.11830817e-02, 4.19549581e-02],
#     [3.64607836e-02, 8.12880127e-01, -5.81288664e-01, 1.28920082e+00],
#     [4.91428199e-02, -5.82430132e-01, -8.11393937e-01, 9.40518766e-01],
#     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
# ]
#
# # Suppress scientific notation and convert to a list of lists with formatted strings
# formatted_matrix = [[f'{number:.8f}' for number in row] for row in matrix]
#
# # Optionally, convert back to floats if you need the numbers in float format
# formatted_matrix_floats = [[float(number) for number in row] for row in formatted_matrix]
#
# print(formatted_matrix_floats)
# exit()


def construct_homogeneous_transform_matrix(translation, orientation):
    translation = np.array(translation).reshape((3, 1))  # xyz
    if len(orientation) == 4:
        rotation = o3d.geometry.get_rotation_matrix_from_quaternion(np.array(orientation).reshape((4, 1)))  # wxyz
    else:
        assert len(orientation) == 3, 'orientation should be a quaternion or 3 axis angles'
        rotation = np.radians(np.array(orientation).astype("float")).reshape((3, 1))  # CBA in radians
        rotation = o3d.geometry.get_rotation_matrix_from_zyx(rotation)
    transformation = np.append(rotation, translation, axis=1)
    transformation = np.append(transformation, np.array([[0, 0, 0, 1]]), axis=0)
    return transformation


script_path = os.path.dirname(os.path.realpath(__file__))
# Load the camera extrinsics
with open(data_calibration_path('zivid_cam_extrinsics.yml'), 'r') as f:
    cam_extrinsics = yaml.load(f, Loader=yaml.FullLoader)
transform_world_to_cam = cam_extrinsics['matrix']
transform_cam_to_world = np.linalg.inv(transform_world_to_cam)

# Create a plane
mesh_box = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
mesh_box.translate([-0.25, 0.35, -0.0011884])

# Create frames
world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
cam_frame.transform(transform_world_to_cam)

# Load PCD (use system-identification target as reference point cloud)
pcd = o3d.io.read_point_cloud(os.path.join(system_identification_target_dir(''), 'pcd_0.ply'))
pcd_in_world_frame = pcd.transform(transform_world_to_cam)

# Visualize
o3d.visualization.draw_geometries([
    world_frame,
    cam_frame,
    mesh_box,
    pcd_in_world_frame
], width=800, height=600)

done = False
while not done:
    delta_x = float(input("Enter delta x: "))
    delta_y = float(input("Enter delta y: "))
    delta_z = float(input("Enter delta z: "))
    delta_rx = float(input("Enter delta rx: "))
    delta_ry = float(input("Enter delta ry: "))
    delta_rz = float(input("Enter delta rz: "))
    print(f"Alternate the tranformation matrix by the following deltas: \n"
          f"x {delta_x}, y {delta_y}, z {delta_z}, rx {delta_rx}, ry {delta_ry}, rz {delta_rz}.")
    T = construct_homogeneous_transform_matrix([delta_x, delta_y, delta_z], [delta_rz, delta_ry, delta_rx])
    new_transform_world_to_cam = T @ transform_world_to_cam
    pcd = dcp(pcd_in_world_frame)
    pcd = pcd.transform(T)
    new_cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    new_cam_frame.transform(new_transform_world_to_cam)
    o3d.visualization.draw_geometries([
        world_frame,
        new_cam_frame,
        mesh_box,
        pcd
    ], width=800, height=600)
    done = input("Done? (y/n): ") == 'y'

print(f"  - {new_transform_world_to_cam[0]}")
print(f"  - {new_transform_world_to_cam[1]}")
print(f"  - {new_transform_world_to_cam[2]}")
print(f"  - {new_transform_world_to_cam[3]}")
np.save(data_calibration_path('cam_extrinsics_fine_tuned.npy'), new_transform_world_to_cam)