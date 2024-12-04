import matplotlib.pyplot as plt
import numpy as np
import os


script_path = os.path.dirname(os.path.realpath(__file__))
hm = np.load(os.path.join(script_path, '..', '..', 'data', 'task_target_pcds', 'pcd_0_cropped_norm_z_aligned_height_map-res40.npy'))
plt.imshow(hm, cmap='hot')
plt.show()