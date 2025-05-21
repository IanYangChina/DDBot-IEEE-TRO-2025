import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

mat = ''
script_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(script_path, '..', '..', 'data', 'task_target_pcds'+mat)
result_path = os.path.join(script_path, '..', '..', 'render_test', 'abs2'+mat)
pcd_offset = (0.2, 0.2, 0.0)
height_map_size = 0.24  # meter
height_map_res = 40
height_map_pixel_size = height_map_size / height_map_res
# ground_level = 0.072  # Soil targets
ground_level = 0.073  # Sand targets

def median_filter(hm_raw):
    hm_raw_filtered = hm_raw.copy()
    for i in range(1, hm_raw.shape[0] - 1):
        for j in range(1, hm_raw.shape[0] - 1):
            pixel = hm_raw[i - 1, j - 1] + hm_raw[i - 1, j] + hm_raw[i - 1, j + 1] + \
                    hm_raw[i, j - 1] + hm_raw[i, j] + hm_raw[i, j + 1] + \
                    hm_raw[i + 1, j - 1] + hm_raw[i + 1, j] + hm_raw[i + 1, j + 1]
            pixel = pixel / 9.0
            hm_raw_filtered[i, j] = pixel
    return hm_raw_filtered

dirs_sand = [
    'd5e6-task-0-hm-ls-lr0.03',
    'd5e6-task-1-hm-ls-demo-lr0.03',
    'd5e6-task-2-hm-ls-demo-lr0.03'
]
dirs_soil = [
    'd5e6-task-0-ls-lr0.03',
    'd5e6-task-1-ls-demo-lr0.03',
    'd5e6-task-2-ls-demo-lr0.03'
]

for n in range(3):
    # Load the point cloud
    if mat == '':
        pcd_path = os.path.join(result_path, dirs_soil[n],
                                'pcd_0_cropped_norm_z_aligned_height_map-res40.npy')
    else:
        pcd_path = os.path.join(result_path, dirs_sand[n],
                                'pcd_0_cropped_norm_z_aligned_height_map-res40.npy')
    hm = np.load(pcd_path)
    hm = median_filter(hm)
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    hole_mask = hm < ground_level

    x_indices, y_indices = np.where(hole_mask)
    hole_centre_x = (np.mean(x_indices) - height_map_res / 2) * height_map_pixel_size
    hole_centre_y = (np.mean(y_indices) - height_map_res / 2) * height_map_pixel_size
    depth = ground_level - np.min(hm[hole_mask])
    planar_area = np.sum(hole_mask) * (height_map_pixel_size ** 2)
    print(f"Results: ({hole_centre_x*100:.2f}, {hole_centre_y*100:.2f}) & {depth*100:.2f} & {planar_area*10000:.2f}")

    ax[0].imshow(hm, vmin=0.002, vmax=0.09)
    ax[0].set_axis_off()
    ax[1].imshow(hole_mask)
    ax[1].set_axis_off()

    target_pcd_path = os.path.join(data_path,
                            f'pcd_{n}_cropped_norm_z_aligned_height_map-res40.npy')
    target_hm = np.load(target_pcd_path)
    target_hm = median_filter(target_hm)
    target_hole_masks = target_hm < ground_level

    target_x_indices, target_y_indices = np.where(target_hole_masks)
    target_hole_centre_x = (np.mean(target_x_indices) - height_map_res / 2) * height_map_pixel_size
    target_hole_centre_y = (np.mean(target_y_indices) - height_map_res / 2) * height_map_pixel_size
    target_depth = ground_level - np.min(target_hm[target_hole_masks])
    target_planar_area = np.sum(target_hole_masks) * (height_map_pixel_size ** 2)
    print(f"Target: ({target_hole_centre_x*100:.2f}, {target_hole_centre_y*100:.2f}) & "
          f"{target_depth*100:.2f} & {target_planar_area*10000:.2f}")

    diff_hole_x = abs(hole_centre_x - target_hole_centre_x)
    diff_hole_y = abs(hole_centre_y - target_hole_centre_y)
    diff_depth = abs(depth - target_depth)
    diff_planar_area = abs(planar_area - target_planar_area)
    print(f"Diff: ({diff_hole_x*100:.2f}, {diff_hole_y*100:.2f}) & "
          f"{diff_depth*100:.2f} & {diff_planar_area*10000:.2f}")

    # ax[2].imshow(target_hm, vmin=0.002, vmax=0.09)
    # ax[2].set_axis_off()
    # ax[3].imshow(target_hole_masks)
    # ax[3].set_axis_off()
    # ax[1].set_title(f"Hole centre: ({hole_centre_x:.2f}, {hole_centre_y:.2f})\n"
    #                 f"Depth: {depth:.2f} m, Planar area: {planar_area:.5f} $m^2$")
    # plt.show()
