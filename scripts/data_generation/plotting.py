import os
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
colour_pool = ['#a42423', '#e89da0',
               '#a55d35', '#efbf6a',
               '#566d40', '#abde70',
               '#22326f', '#4296d2',
               '#662d5e', '#ad3bad']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"
plt.rcParams.update({'font.size': 10})


def plot_trajectory(fig_path):
    sim_pic_ids = [160, 200, 240, 280, 320, 360, 400, 440]
    fig, ax = plt.subplots(2, 10, figsize=(10*2, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(8):
        img = iio.imread(os.path.join(fig_path, f'{i}.png'))
        ax[0, i].imshow(img[:, 180:-180, :])
        ax[0, i].axis('off')
        img_ = iio.imread(os.path.join(fig_path, f'img_{sim_pic_ids[i]}.png'))
        ax[1, i].imshow(img_[:, 100:-100, :])
        ax[1, i].axis('off')

    hm = iio.imread(os.path.join(fig_path, 'height_map.png'))
    ax[0, 8].imshow(np.rot90(np.rot90(hm[324:533, 49:258, :])))
    ax[0, 8].axis('off')
    ax[1, 8].imshow(np.rot90(np.rot90(hm[72:281, 49:258, :])))
    ax[1, 8].axis('off')

    pcd_ = iio.imread(os.path.join(fig_path, 'pcd_target.png'))
    ax[0, 9].imshow(np.rot90(np.rot90(pcd_)))
    ax[0, 9].axis('off')
    pcd = iio.imread(os.path.join(fig_path, 'pcd.png'))
    ax[1, 9].imshow(np.rot90(np.rot90(pcd)))
    ax[1, 9].axis('off')

    plt.savefig(os.path.join(fig_path, 'trajectory.pdf'), bbox_inches='tight', pad_inches=0)


plot_trajectory(fig_path=os.path.join(script_path, '..', 'render_test', 'sysid', 'imgs_sand'))