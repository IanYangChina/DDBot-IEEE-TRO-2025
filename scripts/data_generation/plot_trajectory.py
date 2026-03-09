import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
import numpy as np
import imageio.v3 as iio
import matplotlib.pyplot as plt
from PIL import Image
from paths import render_output_dir

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
plt.rcParams.update({'font.size': 14})


def plot_trajectory(dirs, mat=''):
    folder = render_output_dir(f'abs2{mat}')
    fig, ax = plt.subplots(2*3, 12, figsize=(11*2+0.3, 4*3),
                           gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 1, 1, 0.3, 1, 1, 1]})
    plt.subplots_adjust(wspace=0.02, hspace=0)
    for task in range(3):
        case = dirs[task]
        fig_path = os.path.join(folder, case, f'imgs{mat}')
        if task == 0:
            sim_pic_ids = [0, 20, 40, 60, 80, 100, 120, 140]
            real_pic_margin = (120, -10)
        elif task == 1:
            sim_pic_ids = [0, 20, 40, 60, 80, 100, 120, 140]
            real_pic_margin = (50, -20)
        else:
            sim_pic_ids = [0, 20, 40, 60, 80, 100, 120, 140]
            real_pic_margin = (70, -30)
        for i in range(len(sim_pic_ids)):
            img = iio.imread(os.path.join(fig_path, f'{i+1}.png'))[:, real_pic_margin[0]:real_pic_margin[1], :]
            img = Image.fromarray(img).resize((1600, 1600))
            ax[task*2+0, i].imshow(img)
            img_ = iio.imread(os.path.join(fig_path, f'img_{sim_pic_ids[i]}.png'))[:, 90:-70, :]
            img_ = Image.fromarray(img_).resize((1600, 1600))
            ax[task*2+1, i].imshow(img_)
        ax[task*2+0, 0].set_ylabel('Real Trajectory')
        ax[task*2+1, 0].set_ylabel('Sim Trajectory')

        hm = iio.imread(os.path.join(fig_path, '..', f'height_map{mat}.png'))
        if task == 0:
            ax[task*2+1, 9].imshow(np.rot90(np.rot90(hm)))
        else:
            ax[task*2+1, 9].imshow(np.rot90(np.rot90(np.rot90(hm))))

        ax[task*2+1, 9].set_ylabel('Height Map')
        pcd = iio.imread(os.path.join(fig_path, '..', 'pcd_sim.png'))
        ax[task*2+0, 9].imshow(pcd[110:-110, 150:-150, :])
        ax[task*2+0, 9].set_ylabel('Point Cloud')

        hm = iio.imread(os.path.join(fig_path, '..', 'pcd_0_cropped_norm_z_aligned_height_map-res40.png'))
        ax[task*2+1, 10].imshow(np.rot90(np.rot90(np.rot90(hm))))
        pcd_ = iio.imread(os.path.join(fig_path, '..', 'pcd_real.png'))
        ax[task*2+0, 10].imshow(pcd_[50:-50, 50:-50, :])
        if task == 1:
            ax[task * 2 + 0, 10].imshow(pcd_[70:-70, 70:-70, :])

        hm_target = iio.imread(os.path.join(fig_path, '..', '..',
                                            f'pcd_{task}_cropped_norm_z_aligned_height_map-res40.png'))
        ax[task*2+1, 11].imshow(np.rot90(np.rot90(np.rot90(hm_target))))
        pcd_target = iio.imread(os.path.join(fig_path, '..', '..', f'task_{task}_pcd.png'))
        ax[task*2+0, 11].imshow(pcd_target[50:-50, 50:-50, :])

        for n in range(12):
            ax[task*2+0, n].set_xticks([])
            ax[task*2+0, n].set_yticks([])
            ax[task*2+0, n].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
            ax[task*2+1, n].set_xticks([])
            ax[task*2+1, n].set_yticks([])
            ax[task*2+1, n].spines[['right', 'top', 'bottom', 'left']].set_visible(False)

        ax[0, 9].set_title('Simulation')
        ax[0, 10].set_title('Real')
        ax[0, 11].set_title('Target')

    if mat == '':
        mat = '_soil'
    plt.savefig(os.path.join(folder, f'trajectory{mat}.pdf'), dpi=500,
                bbox_inches='tight', pad_inches=0.01)


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

plot_trajectory(dirs=dirs_soil, mat='')
