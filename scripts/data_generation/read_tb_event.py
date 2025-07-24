import os
import json
import numpy as np
from copy import deepcopy as dcp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow.python.summary.summary_iterator import summary_iterator

script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
colour_pool = ['#a42423',
               '#566d40',
               '#efbf6a',
               '#22326f', #'#4296d2',
               '#662d5e', #'#ad3bad',
               ]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"
plt.rcParams.update({'font.size': 10})


def find_best_parameters(hm=False, grad_norm=False, grad_dy_scale=False, grad_clip=False,
                         line_search=False, man_init=False, ptcl='5e6', mat='_sand',
                         resolutions=[10, 20, 30, 40, 50, 60]):
    """generate mean and deviation data from tensorboard logs"""
    case = f'd{ptcl}'
    if hm:
        case += '-hm'

    if grad_norm:
        case += '-gnorm'
    elif grad_dy_scale:
        case += '-gdys'
    elif grad_clip:
        case += '-gclip'
    else:
        case += '-gnone'

    if line_search:
        case += '-ls'
    if man_init:
        case += '-man-init'

    for res in resolutions:
        case_folder = os.path.join(script_path, '..', f'log-sys_id{mat}', case + f'-res{res}')
        best_loss = np.inf
        best_loss_info = {}
        for seed in range(5):
            folder = os.path.join(case_folder, f'seed-{seed}')
            data_dict = {
                'Loss': {
                    'emd_loss': [],
                    'height_map_loss': [],
                },
                'Validation Loss': {
                    'emd_loss': [],
                    'height_map_loss': [],
                    'total_loss': [],
                },
                'Parameters': {
                    'E': [],
                    'nu': [],
                    'rho': [],
                    'sand_angle': [],
                }
            }

            for filename in os.listdir(folder):
                if filename[:5] == 'event':
                    for event in summary_iterator(os.path.join(folder, filename)):
                        for v in event.summary.value:
                            if v.tag[:5] == 'Loss/':
                                if v.tag[5:] == 'emd_loss':
                                    data_dict['Loss']['emd_loss'].append(v.simple_value)
                                elif v.tag[5:] == 'height_map_loss':
                                    data_dict['Loss']['height_map_loss'].append(v.simple_value)
                                else:
                                    pass
                            elif v.tag[:16] == 'Validation Loss/':
                                if v.tag[16:] == 'emd_loss':
                                    data_dict['Validation Loss']['emd_loss'].append(v.simple_value)
                                elif v.tag[16:] == 'height_map_loss':
                                    data_dict['Validation Loss']['height_map_loss'].append(v.simple_value)
                            elif v.tag[:6] == 'Param/':
                                if v.tag[6:] == 'E':
                                    data_dict['Parameters']['E'].append(v.simple_value)
                                elif v.tag[6:] == 'nu':
                                    data_dict['Parameters']['nu'].append(v.simple_value)
                                elif v.tag[6:] == 'rho':
                                    data_dict['Parameters']['rho'].append(v.simple_value)
                                elif v.tag[6:] == 'sand_angle':
                                    data_dict['Parameters']['sand_angle'].append(v.simple_value)
                                else:
                                    pass
                            else:
                                pass

            data_dict['Validation Loss']['total_loss'] = (
                        np.asarray(data_dict['Validation Loss']['height_map_loss']) / (40 * 40) +
                        np.asarray(data_dict['Validation Loss']['emd_loss']) / (40 * 40)).tolist()
            json.dump(data_dict, open(os.path.join(folder, 'raw_data.json'), 'w'))
            min_loss_id = np.argmin(data_dict['Validation Loss']['total_loss'])
            best_loss_seed = {
                'Step': float(min_loss_id),
                'Loss': {
                    'emd_loss': data_dict['Loss']['emd_loss'][min_loss_id],
                    'height_map_loss': data_dict['Loss']['height_map_loss'][min_loss_id],
                },
                'Validation Loss': {
                    'emd_loss': data_dict['Validation Loss']['emd_loss'][min_loss_id],
                    'height_map_loss': data_dict['Validation Loss']['height_map_loss'][min_loss_id],
                    'total_loss': data_dict['Validation Loss']['total_loss'][min_loss_id],
                },
                'Parameters': {
                    'E': data_dict['Parameters']['E'][min_loss_id],
                    'nu': data_dict['Parameters']['nu'][min_loss_id],
                    'rho': data_dict['Parameters']['rho'][min_loss_id],
                    'sand_angle': data_dict['Parameters']['sand_angle'][min_loss_id],
                }}
            json.dump(best_loss_seed, open(os.path.join(folder, 'best_loss.json'), 'w'))

            if best_loss_seed['Validation Loss']['total_loss'] < best_loss:
                best_loss = best_loss_seed['Validation Loss']['total_loss']
                best_loss_info = dcp(best_loss_seed)

        open(os.path.join(case_folder, 'best_params.json'), 'w').write(json.dumps(best_loss_info))


# find_best_parameters(ptcl='1e7', grad_clip=True, line_search=True, man_init=True, hm=True, resolutions=[40])
# find_best_parameters(ptcl='2e7', grad_clip=True, line_search=True, man_init=True, hm=True, resolutions=[40])
# find_best_parameters(grad_clip=True, line_search=True, man_init=False, hm=True, resolutions=[40])
# find_best_parameters(grad_clip=True, line_search=True, man_init=False, hm=False, resolutions=[40])
# find_best_parameters(grad_clip=True, line_search=False, man_init=False, hm=True, resolutions=[40])  # validation res 60 for soil
# find_best_parameters(grad_clip=True, line_search=False, man_init=False, hm=False, resolutions=[40])  # validation res 60 for soil
# find_best_parameters(grad_clip=True, line_search=True, man_init=True, hm=True, resolutions=[40])
# find_best_parameters(grad_clip=True, line_search=True, man_init=True, hm=False, resolutions=[40])
# find_best_parameters(grad_norm=True, line_search=True, man_init=True, hm=True, resolutions=[40])
# find_best_parameters(grad_norm=True, line_search=True, man_init=True, hm=False, resolutions=[40])
# find_best_parameters(grad_dy_scale=True, line_search=True, man_init=True, hm=True, resolutions=[40])
# find_best_parameters(grad_dy_scale=True, line_search=True, man_init=True, hm=False, resolutions=[40])


def plot_si_scatter(mat='_sand'):
    plt.figure(figsize=(2, 4))
    plt.rcParams.update({'font.size': 11})
    cases = ['gclip', 'hm-gclip',
             'gclip-ls', 'hm-gclip-ls',
             'gclip-ls-man-init', 'hm-gclip-ls-man-init',
             'gnorm-ls-man-init', 'hm-gnorm-ls-man-init',
             'gdys-ls-man-init', 'hm-gdys-ls-man-init']
    casenames = ['ClipGrad', 'ClipGrad-HM',
                 'ClipGrad-LS', 'ClipGrad-LS-HM',
                 'ClipGrad-LS-ManInit', 'ClipGrad-LS-HM-ManInit',
                 'NormGrad-LS-ManInit', 'NormGrad-LS-HM-ManInit',
                 'DyScaleGrad-LS-ManInit', 'DyScaleGrad-LS-HM-ManInit']
    for case_id in range(len(cases)):
        x = []
        x_std = []
        y = []
        y_std = []
        case = cases[case_id]
        for res in [40]:
            case_folder = os.path.join(script_path, '..', f'log-sys_id{mat}', f'd5e6-{case}-res{res}')
            best_loss = []
            earliest_epoch = []
            for seed in range(5):
                folder = os.path.join(case_folder, f'seed-{seed}')
                with open(os.path.join(folder, 'best_loss.json')) as f:
                    data = json.load(f)
                    earliest_epoch.append(data['Step'])
                    best_loss.append(data['Validation Loss']['total_loss'])
            y.append(np.mean(best_loss))
            y_std.append(np.std(best_loss))
            x.append(np.mean(earliest_epoch))
            x_std.append(np.std(earliest_epoch))
        plt.errorbar(y, len(cases) - case_id - 1, label=casenames[case_id], color=colour_pool[case_id], xerr=y_std,
                     fmt='h',
                     capsize=4, markersize=6, elinewidth=1, capthick=1)

    plt.grid(True, axis='x')
    casenames.reverse()
    if mat == '':
        plt.xlim(0.0235, 0.0295)
        plt.xticks([0.0245, 0.0265, 0.0285])
    plt.yticks([])
    plt.xlabel('Best validation loss')
    plt.savefig(os.path.join(script_path, '..', 'figs', f'sys_id{mat}.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.01)


# plot_si_scatter(mat='')
# plot_si_scatter(mat='_sand')


def plot_si_loss_curve(mat=''):
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(2, 5, figsize=(12, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    cases = ['gclip', 'hm-gclip',
             'gclip-ls', 'hm-gclip-ls',
             'gclip-ls-man-init', 'hm-gclip-ls-man-init',
             'gnorm-ls-man-init', 'hm-gnorm-ls-man-init',
             'gdys-ls-man-init', 'hm-gdys-ls-man-init']
    casenames = ['ClipGrad-EMD', 'ClipGrad-HMD',
                 'ClipGrad-LS-EMD', 'ClipGrad-LS-HMD',
                 'ClipGrad-LS-EMD-ManInit', 'ClipGrad-LS-HMD-ManInit',
                 'NormGrad-LS-EMD-ManInit', 'NormGrad-LS-HMD-ManInit',
                 'DyScaleGrad-LS-EMD-ManInit', 'DyScaleGrad-LS-HMD-ManInit']
    window = 2
    for case_id in range(len(cases)):
        case = cases[case_id]
        if 'hm' in case:
            row = 1
            column = int((case_id - 1) / 2)
        else:
            row = 0
            column = int(case_id / 2)

        for res in [40]:
            case_folder = os.path.join(script_path, '..', f'log-sys_id{mat}', f'd5e6-{case}-res{res}')
            losses = []
            for seed in range(5):
                folder = os.path.join(case_folder, f'seed-{seed}')
                with open(os.path.join(folder, 'raw_data.json')) as f:
                    data = json.load(f)
                    losses.append(np.asarray(data['Validation Loss']['total_loss']))
            mean_loss = np.mean(losses, axis=0)
            running_avg = np.empty(mean_loss.shape[0])
            for n in range(mean_loss.shape[0]):
                running_avg[n] = np.mean(mean_loss[max(0, n - window):(n + 1)])
            xs = np.arange(len(running_avg))
            ax[row, column].plot(xs, running_avg, label=casenames[case_id], color=colour_pool[case_id], linewidth=2)
            for l in losses:
                running_avg_l = np.empty(l.shape[0])
                for n in range(l.shape[0]):
                    running_avg_l[n] = np.mean(l[max(0, n - window):(n + 1)])
                ax[row, column].plot(xs, running_avg_l, color=colour_pool[case_id], linestyle='--', linewidth=1)

            if mat == '_sand':
                ax[row, column].set_ylim([0.0215, 0.0245])
                ax[row, column].set_yticks([0.022, 0.023, 0.024])
            else:
                ax[row, column].set_ylim([0.023, 0.033])
                ax[row, column].set_yticks([0.024, 0.026, 0.028, 0.030, 0.032])
            ax[row, column].set_xlim([-2, 21])
            ax[row, column].set_xticks([0, 4, 9, 14, 19])
            ax[row, column].set_xticklabels(['1', '5', '10', '15', '20'])
            ax[row, column].grid(True)
            if column == 0:
                ax[row, column].set_ylabel('Validation loss')
                ax[row, column].spines[['right']].set_visible(False)
            elif column == 4:
                for tick in ax[row, column].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            else:
                ax[row, column].spines[['right']].set_visible(False)
                for tick in ax[row, column].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            if row == 0:
                ax[row, column].spines[['bottom']].set_visible(False)
                for tick in ax[row, column].xaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

            ax[row, column].set_xlabel('Epoch')
            handle = Line2D([0], [0], color=colour_pool[case_id], linewidth=5)
            ax[row, column].legend([handle], [casenames[case_id]], loc='upper left', fontsize=8,
                                   handlelength=0.1, frameon=True)

    plt.savefig(os.path.join(script_path, '..', 'figs', f'sys_id_loss_curve{mat}.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.01)


# plot_si_loss_curve(mat='')
# plot_si_loss_curve(mat='_sand')


def plot_task_pcds():
    data_path = os.path.join(script_path, '..', 'data')
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(4, 3, figsize=(2*3, 2*4))
    plt.subplots_adjust(wspace=0, hspace=-0.6)
    for mat in ['soil', 'sand']:
        if mat == 'soil':
            path_subfix = ''
            pcd_row = 0
            real_row = 1
        else:
            path_subfix = '_sand'
            pcd_row = 2
            real_row = 3
        for task_id in range(3):
            pcd_img_path = os.path.join(data_path, f'task_target_pcds{path_subfix}',
                                        f't{task_id}-{mat}.png')
            real_img_path = os.path.join(data_path, f'task_target_pcds{path_subfix}',
                                         f't{task_id}-{mat}.JPG')
            pcd_img = plt.imread(pcd_img_path)
            real_img = plt.imread(real_img_path)
            ax[pcd_row, task_id].imshow(pcd_img)
            ax[pcd_row, task_id].set_xticks([])
            ax[pcd_row, task_id].set_yticks([])
            ax[pcd_row, task_id].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
            ax[real_row, task_id].imshow(real_img)
            ax[real_row, task_id].set_xticks([])
            ax[real_row, task_id].set_yticks([])
            ax[real_row, task_id].spines[['right', 'top', 'bottom', 'left']].set_visible(False)

    # mt0_pcd_soil = plt.imread(os.path.join(data_path, 'task_target_pcds', 'mt0-soil.png'))
    # mt0_real_soil = plt.imread(os.path.join(data_path, 'task_target_pcds', 'mt0-soil.JPG'))
    # mt0_pcd_sand = plt.imread(os.path.join(data_path, 'task_target_pcds_sand', 'mt0-sand.png'))
    # mt0_real_sand = plt.imread(os.path.join(data_path, 'task_target_pcds_sand', 'mt0-sand.JPG'))
    # ax[0, 3].imshow(mt0_pcd_soil)
    # ax[0, 3].set_xticks([])
    # ax[0, 3].set_yticks([])
    # ax[0, 3].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    # ax[1, 3].imshow(mt0_real_soil)
    # ax[1, 3].set_xticks([])
    # ax[1, 3].set_yticks([])
    # ax[1, 3].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    # ax[2, 3].imshow(mt0_pcd_sand)
    # ax[2, 3].set_xticks([])
    # ax[2, 3].set_yticks([])
    # ax[2, 3].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    # ax[3, 3].imshow(mt0_real_sand)
    # ax[3, 3].set_xticks([])
    # ax[3, 3].set_yticks([])
    # ax[3, 3].spines[['right', 'top', 'bottom', 'left']].set_visible(False)

    for i in range(3):
        ax[0, i].set_title(f'Task {i+1}')
    # ax[0, 3].set_title(f'Multi-skill Task')
    ax[0, 0].set_ylabel('Soil PCD', labelpad=0)
    ax[2, 0].set_ylabel('Sand PCD', labelpad=0)
    ax[1, 0].set_ylabel('Soil Real', labelpad=0)
    ax[3, 0].set_ylabel('Sand Real', labelpad=0)

    plt.savefig(os.path.join(script_path, '..', 'figs', 'task_pcds.pdf'),
                dpi=200, bbox_inches='tight', pad_inches=0.01)


# plot_task_pcds()


def find_best_skill_parameters(mat='', sac=False, abs0=False, cmamae=False, rg=False):
    if sac:
        log_folder = f'log-abs2-sac{mat}'
    elif cmamae:
        log_folder = f'log-abs2-cmamae{mat}'
    elif abs0:
        log_folder = f'log-abs0{mat}'
    elif rg:
        log_folder = f'log-abs2-rg{mat}'
    else:
        log_folder = f'log-abs2{mat}'
    dirs = os.listdir(os.path.join(script_path, '..', log_folder))
    for p_dir in dirs:
        p_folder = os.path.join(script_path, '..', log_folder, p_dir)
        if os.path.isfile(os.path.join(p_folder, 'best_loss.json')):
            continue
        best_loss = np.inf
        best_loss_info = {}
        for seed in [0, 1, 2, 3, 4]:
            folder = os.path.join(p_folder, f'seed-{seed}')
            if sac:
                folder = os.path.join(folder, 'data')
            data_dict = {
                'Loss': {
                    'emd_loss': [],
                    'height_map_loss': [],
                    'total_loss': [],
                },
                'Parameters': {
                    'skill_params_0': [],
                    'skill_params_1': [],
                    'skill_params_2': [],
                    'skill_params_3': [],
                    'skill_params_4': [],
                },
                'SAC': {
                    'actor_loss': [],
                    'temperature': [],
                    'policy_entropy': [],
                    'critic_1_loss': [],
                    'critic_2_loss': [],
                    'emd_loss': [],
                    'height_map_loss': [],
                    'test_emd_loss': [],
                    'test_height_map_loss': [],
                }
            }
            for filename in os.listdir(folder):
                if filename[:5] == 'event':
                    for event in summary_iterator(os.path.join(folder, filename)):
                        if cmamae:
                            for v in event.summary.value:
                                if v.tag[:7] == 'loss_0/':
                                    if v.tag[7:] == 'EMD':
                                        data_dict['Loss']['emd_loss'].append(v.simple_value)
                                    elif v.tag[7:] == 'Heightmap':
                                        data_dict['Loss']['height_map_loss'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:8] == 'param_0/':
                                    if v.tag[8] == '0':
                                        data_dict['Parameters']['skill_params_0'].append([v.simple_value])
                                    elif v.tag[8] == '1':
                                        data_dict['Parameters']['skill_params_1'].append([v.simple_value])
                                    elif v.tag[8] == '2':
                                        data_dict['Parameters']['skill_params_2'].append([v.simple_value])
                                    elif v.tag[8] == '3':
                                        data_dict['Parameters']['skill_params_3'].append([v.simple_value])
                                    elif v.tag[8] == '4':
                                        data_dict['Parameters']['skill_params_4'].append([v.simple_value])
                                    else:
                                        pass
                                else:
                                    pass
                        elif sac:
                            for v in event.summary.value:
                                if v.tag[:6] == 'Actor/':
                                    if v.tag[6:] == 'actor_loss':
                                        data_dict['SAC']['actor_loss'].append(v.simple_value)
                                    elif v.tag[6:] == 'alpha':
                                        data_dict['SAC']['temperature'].append(v.simple_value)
                                    elif v.tag[6:] == 'policy_entropy':
                                        data_dict['SAC']['policy_entropy'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:7] == 'Critic/':
                                    if v.tag[7:] == 'critic_1_loss':
                                        data_dict['SAC']['critic_1_loss'].append(v.simple_value)
                                    elif v.tag[7:] == 'critic_2_loss':
                                        data_dict['SAC']['critic_2_loss'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:5] == 'Task/':
                                    if v.tag[5:] == 'emd_loss':
                                        data_dict['SAC']['emd_loss'].append(v.simple_value)
                                    elif v.tag[5:] == 'heightmap_loss':
                                        data_dict['SAC']['height_map_loss'].append(v.simple_value)
                                    elif v.tag[5:] == 'test_emd_loss':
                                        data_dict['Loss']['emd_loss'].append(v.simple_value)
                                        data_dict['SAC']['test_emd_loss'].append(v.simple_value)
                                    elif v.tag[5:] == 'test_heightmap_loss':
                                        data_dict['Loss']['height_map_loss'].append(v.simple_value)
                                        data_dict['SAC']['test_height_map_loss'].append(v.simple_value)
                                    else:
                                        pass
                        else:
                            for v in event.summary.value:
                                if v.tag[:5] == 'loss/':
                                    if v.tag[5:] == 'EMD':
                                        data_dict['Loss']['emd_loss'].append(v.simple_value)
                                    elif v.tag[5:] == 'Heightmap':
                                        data_dict['Loss']['height_map_loss'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:6] == 'param/':
                                    if v.tag[6] == '0':
                                        data_dict['Parameters']['skill_params_0'].append([v.simple_value])
                                    elif v.tag[6] == '1':
                                        data_dict['Parameters']['skill_params_1'].append([v.simple_value])
                                    elif v.tag[6] == '2':
                                        data_dict['Parameters']['skill_params_2'].append([v.simple_value])
                                    elif v.tag[6] == '3':
                                        data_dict['Parameters']['skill_params_3'].append([v.simple_value])
                                    elif v.tag[6] == '4':
                                        data_dict['Parameters']['skill_params_4'].append([v.simple_value])
                                    else:
                                        pass
                                else:
                                    pass

            data_dict['Loss']['total_loss'] = (
                        np.asarray(data_dict['Loss']['height_map_loss']) / (40 * 40) +
                        np.asarray(data_dict['Loss']['emd_loss']) / (40 * 40)).tolist()
            json.dump(data_dict, open(os.path.join(folder, 'raw_data.json'), 'w'))
            if cmamae:
                min_loss_id = np.argmin(data_dict['Loss']['total_loss'][:19])
            else:
                min_loss_id = np.argmin(data_dict['Loss']['total_loss'])
            best_loss_seed = {
                'Step': float(min_loss_id),
                'Loss': {
                    'emd_loss': data_dict['Loss']['emd_loss'][min_loss_id],
                    'height_map_loss': data_dict['Loss']['height_map_loss'][min_loss_id],
                    'total_loss': data_dict['Loss']['total_loss'][min_loss_id],
                }}
            if not sac and not abs0:
                best_loss_seed.update({
                    'Parameters': {
                        'skill_params_0': data_dict['Parameters']['skill_params_0'][min_loss_id],
                        'skill_params_1': data_dict['Parameters']['skill_params_1'][min_loss_id],
                        'skill_params_2': data_dict['Parameters']['skill_params_2'][min_loss_id],
                        'skill_params_3': data_dict['Parameters']['skill_params_3'][min_loss_id],
                        'skill_params_4': data_dict['Parameters']['skill_params_4'][min_loss_id],
                    }})
            json.dump(best_loss_seed, open(os.path.join(folder, 'best_loss.json'), 'w'))

            if data_dict['Loss']['total_loss'][min_loss_id] < best_loss:
                best_loss = data_dict['Loss']['total_loss'][min_loss_id]
                best_loss_info = dcp(best_loss_seed)

        open(os.path.join(p_folder, 'best_loss.json'), 'w').write(json.dumps(best_loss_info))


# find_best_skill_parameters(mat='')
# find_best_skill_parameters(mat='_sand')
# find_best_skill_parameters(mat='', rg=True)
# find_best_skill_parameters(mat='', cmamae=True)


def plot_so_loss_curve(mat='', baseline=False, rg=False, sompc=False):
    plt.rcParams.update({'font.size': 16})
    handle_fontsize = 9.5
    cases = [
        'ls-lr0.03',
        'hm-ls-lr0.03',
        'ls-demo-lr0.03',
        'hm-ls-demo-lr0.03'
    ]
    casenames = [
        'EMD-LS',
        'HMD-LS',
        'EMD-LS-Demo',
        'HMD-LS-Demo',
    ]
    case_p_folders = [
        os.path.join(script_path, '..', f'log-abs2{mat}'),
        os.path.join(script_path, '..', f'log-abs2{mat}'),
        os.path.join(script_path, '..', f'log-abs2{mat}'),
        os.path.join(script_path, '..', f'log-abs2{mat}'),
    ]
    width_ratios = [0.6, 0.6, 0.6, 0.6, 0.05, 0.5]
    if rg:
        cases = [
            'ls-demo-lr0.03',
            'ls-demo-lr0.03',
            'hm-ls-demo-lr0.03',
            'hm-ls-demo-lr0.03'
        ]
        casenames = [
            'EMD-Rounded',
            'EMD-Unrounded',
            'HMD-Rounded',
            'HMD-Unrounded'
        ]
        case_p_folders = [
            os.path.join(script_path, '..', f'log-abs2{mat}'),
            os.path.join(script_path, '..', f'log-abs2-rg{mat}'),
            os.path.join(script_path, '..', f'log-abs2{mat}'),
            os.path.join(script_path, '..', f'log-abs2-rg{mat}')
        ]
    if baseline:
        cases = [
            'ls-demo-lr0.03',
            'ls-demo-lr0.004',
            'demo-em2-bs10',
            'her-demo'
        ]
        casenames = [
            'SO-EMD-LS',
            'TR-EMD-LS',
            'CMAMAE-EMD',
            'SAC-EMD-HER'
        ]
        case_p_folders = [
            os.path.join(script_path, '..', f'log-abs2-rg{mat}'),
            os.path.join(script_path, '..', f'log-abs0'),
            os.path.join(script_path, '..', f'log-abs2-cmamae'),
            os.path.join(script_path, '..', f'log-abs2-sac')
        ]
    if sompc:
        cases = [
            'ls-lr0.03',
            'ls-demo-lr0.03',
            'em2-bs10',
            'demo-em2-bs10'
        ]
        casenames = [
            'SO-EMD-LS',
            'SO-EMD-LS-Demo',
            'CMAMAE-EMD',
            'CMAMAE-EMD-Demo'
        ]
        case_p_folders = [
            os.path.join(script_path, '..', f'log-abs2-rg{mat}'),
            os.path.join(script_path, '..', f'log-abs2-rg{mat}'),
            os.path.join(script_path, '..', f'log-abs2-cmamae'),
            os.path.join(script_path, '..', f'log-abs2-cmamae')
        ]

    fig, ax = plt.subplots(3, len(cases)+2, figsize=(len(cases)*2+2.05, 3*1.5),
                           gridspec_kw={'width_ratios': width_ratios,
                                        'height_ratios': [1, 1, 1]})
    plt.subplots_adjust(wspace=0, hspace=0.01)
    window = 2
    for case_id in range(len(cases)):
        case = cases[case_id]
        column = case_id
        for task in range(3):
            case_folder = os.path.join(case_p_folders[case_id], f'd5e6-task-{task}-{case}')
            losses = []
            for seed in range(5):
                folder = os.path.join(case_folder, f'seed-{seed}')
                with open(os.path.join(folder, 'raw_data.json')) as f:
                    data = json.load(f)
                    losses.append(np.asarray(data['Loss']['total_loss']))
            mean_loss = np.mean(losses, axis=0)
            running_avg = np.empty(mean_loss.shape[0])
            for n in range(mean_loss.shape[0]):
                running_avg[n] = np.mean(mean_loss[max(0, n - window):(n + 1)])
            if not baseline:
                xs = np.arange(20)
                running_avg = running_avg[:20]
            elif case_id > 2:
                xs = np.arange(39)
                running_avg = running_avg[:39]
            else:
                xs = np.arange(20)
                running_avg = running_avg[:20]
            ax[task, column].plot(xs, running_avg, label=casenames[case_id], color=colour_pool[case_id], linewidth=2)
            for l in losses:
                running_avg_l = np.empty(l.shape[0])
                for n in range(l.shape[0]):
                    running_avg_l[n] = np.mean(l[max(0, n - window):(n + 1)])
                if not baseline:
                    running_avg_l = running_avg_l[:20]
                elif case_id > 2:
                    running_avg_l = running_avg_l[:39]
                else:
                    running_avg_l = running_avg_l[:20]
                ax[task, column].plot(xs, running_avg_l, color=colour_pool[case_id], linestyle='--', linewidth=1)

            if task == 0:
                if mat == '_sand':
                    ax[task, column].set_ylim([0.0115, 0.0205])
                    ax[task, column].set_yticks([0.013, 0.015, 0.017, 0.019])
                elif baseline:
                    ax[task, column].set_ylim([0.011, 0.026])
                    ax[task, column].set_yticks([0.013, 0.016, 0.019, 0.022])
                else:
                    ax[task, column].set_ylim([0.012, 0.021])
                    ax[task, column].set_yticks([0.013, 0.015, 0.017, 0.019])
            elif task == 1:
                if mat == '_sand':
                    ax[task, column].set_ylim([0.011, 0.017])
                    ax[task, column].set_yticks([0.012, 0.014, 0.016])
                elif baseline:
                    ax[task, column].set_ylim([0.0145, 0.0225])
                    ax[task, column].set_yticks([0.015, 0.017, 0.019, 0.021])
                else:
                    ax[task, column].set_ylim([0.0155, 0.0185])
                    ax[task, column].set_yticks([0.016, 0.017, 0.018])
            else:
                if mat == '_sand':
                    ax[task, column].set_ylim([0.011, 0.018])
                    ax[task, column].set_yticks([0.012, 0.014, 0.016])
                elif baseline:
                    ax[task, column].set_ylim([0.0175, 0.0295])
                    ax[task, column].set_yticks([0.019, 0.022, 0.025, 0.028])
                else:
                    ax[task, column].set_ylim([0.018, 0.025])
                    ax[task, column].set_yticks([0.020, 0.022, 0.024])
            if not baseline:
                ax[task, column].set_xlim([-2, 21])
                ax[task, column].set_xticks([0, 4, 9, 14, 19])
                ax[task, column].set_xticklabels(['1', '5', '10', '15', '20'])
            elif case_id == 3:
                ax[task, column].set_xlim([-3, 42])
                ax[task, column].set_xticks([0, 9, 19, 29, 39])
                ax[task, column].set_xticklabels(['1', '50', '100', '150', '200'])
            else:
                ax[task, column].set_xlim([-2, 21])
                ax[task, column].set_xticks([0, 4, 9, 14, 19])
                ax[task, column].set_xticklabels(['1', '5', '10', '15', '20'])

            if task != 2:
                for tick in ax[task, column].xaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            ax[task, column].grid(True)
            if column == 0:
                ax[task, column].set_ylabel(f'Task {task+1}')
                ax[task, column].spines[['right']].set_visible(False)
            elif column == len(cases)-1:
                for tick in ax[task, column].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            else:
                ax[task, column].spines[['right']].set_visible(False)
                for tick in ax[task, column].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

            ax[task, column].set_xlabel('Epoch')
            handle = Line2D([0], [0], color=colour_pool[case_id], linewidth=5)
            ax[task, column].legend([handle], [casenames[case_id]], loc='upper left', fontsize=handle_fontsize,
                                    handlelength=0.1, frameon=True)

    for case_id in range(len(cases)):
        for task in range(3):
            y = []
            y_std = []
            case = cases[case_id]
            case_folder = os.path.join(case_p_folders[case_id], f'd5e6-task-{task}-{case}')

            best_loss = []
            for seed in range(5):
                folder = os.path.join(case_folder, f'seed-{seed}')
                with open(os.path.join(folder, 'best_loss.json')) as f:
                    data = json.load(f)
                    best_loss.append(data['Loss']['total_loss'])
            y.append(np.mean(best_loss))
            y_std.append(np.std(best_loss))

            ax[task, len(cases)+1].errorbar(case_id, y, color=colour_pool[case_id], yerr=y_std,
                                            fmt='h', capsize=4, markersize=6, elinewidth=1, capthick=1)

    for task in range(3):
        ax[task, len(cases)+1].grid(True, axis='y')
        if task == 0:
            if mat == '_sand':
                ax[task, len(cases) + 1].set_ylim([0.0115, 0.0205])
                ax[task, len(cases)+1].set_yticks([0.013, 0.015, 0.017, 0.019])
            elif baseline:
                ax[task, len(cases)+1].set_ylim([0.011, 0.026])
                ax[task, len(cases)+1].set_yticks([0.013, 0.016, 0.019, 0.022])
            else:
                ax[task, len(cases) + 1].set_ylim([0.012, 0.021])
                ax[task, len(cases) + 1].set_yticks([0.013, 0.015, 0.017, 0.019])
        elif task == 1:
            if mat == '_sand':
                ax[task, len(cases) + 1].set_ylim([0.011, 0.017])
                ax[task, len(cases) + 1].set_yticks([0.012, 0.014, 0.016])
            elif baseline:
                ax[task, len(cases)+1].set_ylim([0.0145, 0.0225])
                ax[task, len(cases)+1].set_yticks([0.015, 0.017, 0.019, 0.021])
            else:
                ax[task, len(cases)+1].set_ylim([0.0155, 0.0185])
                ax[task, len(cases)+1].set_yticks([0.016, 0.017, 0.018])
        else:
            if mat == '_sand':
                ax[task, len(cases) + 1].set_ylim([0.011, 0.018])
                ax[task, len(cases) + 1].set_yticks([0.012, 0.014, 0.016])
            elif baseline:
                ax[task, len(cases)+1].set_ylim([0.0175, 0.0295])
                ax[task, len(cases)+1].set_yticks([0.019, 0.022, 0.025, 0.028])
            else:
                ax[task, len(cases) + 1].set_ylim([0.018, 0.025])
                ax[task, len(cases) + 1].set_yticks([0.020, 0.022, 0.024])

        for tick in ax[task, len(cases)+1].yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)
            tick.label1.set_visible(False)
            tick.label2.set_visible(False)
        ax[task, len(cases)+1].set_xlim([-0.5, len(cases)-0.5])
        ax[task, len(cases)+1].set_xticks([])
        if task == 2:
            ax[task, len(cases)+1].set_xlabel('Best validation \nloss (Mean ± Std)')

        ax[task, len(cases)].axis('off')
    if baseline:
        mat = '_baseline'
    if mat == '':
        mat = '_soil'
    if rg:
        mat += '_rg'
    if sompc:
        mat += '_sompc'
    plt.savefig(os.path.join(script_path, '..', 'figs', f'task{mat}.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.01)


# plot_so_loss_curve(mat='', baseline=True, rg=False)
# plot_so_loss_curve(mat='', baseline=False, rg=False)
# plot_so_loss_curve(mat='_sand', baseline=False, rg=False)
# plot_so_loss_curve(mat='', baseline=False, rg=True)
plot_so_loss_curve(mat='', baseline=False, rg=False, sompc=True)


def plot_rl_loss_curve():
    plt.rcParams.update({'font.size': 11})
    data_name = ['actor_loss', 'critic_loss',
                 'temperature', 'policy_entropy',
                 'emd_loss', 'test_emd_loss',
                 'height_map_loss', 'test_height_map_loss']
    data_name_legend = ['Actor loss (Q value)', 'Critic loss',
                        r'Temperature $\alpha$', 'Policy entropy',
                        'EMD loss', 'Test EMD loss (every 5 training epochs)',
                        'Height map loss', 'Test height map loss (every 5 training epochs)']
    linestyles = ['-', '--', '-.', ':']
    for task_id in range(3):
        fig, ax = plt.subplots(len(data_name)//2, 2, figsize=(1*12, len(data_name)*1))
        plt.subplots_adjust(wspace=0.12, hspace=0.03)
        data_dict = {
            'actor_loss': [],
            'critic_loss': [],
            'temperature': [],
            'policy_entropy': [],
            'emd_loss': [],
            'height_map_loss': [],
            'test_emd_loss': [],
            'test_height_map_loss': [],
        }
        for seed in range(5):
            d_folder = os.path.join(script_path, '..', 'log-abs2-sac',
                                    f'd5e6-task-{task_id}-her-demo', f'seed-{seed}')
            with open(os.path.join(d_folder, 'raw_data.json')) as f:
                data = json.load(f)
            data_dict['actor_loss'].append(data['SAC']['actor_loss'])
            data_dict['critic_loss'].append(data['SAC']['critic_1_loss'])
            data_dict['critic_loss'].append(data['SAC']['critic_2_loss'])
            data_dict['temperature'].append(data['SAC']['temperature'])
            data_dict['policy_entropy'].append(data['SAC']['policy_entropy'])
            data_dict['emd_loss'].append(data['SAC']['emd_loss'])
            data_dict['height_map_loss'].append(data['SAC']['height_map_loss'])
            data_dict['test_emd_loss'].append(data['SAC']['test_emd_loss'])
            data_dict['test_height_map_loss'].append(data['SAC']['test_height_map_loss'])
        for n in range(len(data_name)):
            row = n // 2
            col = n % 2
            key = data_name[n]
            upper = np.max(data_dict[key], axis=0)
            lower = np.min(data_dict[key], axis=0)
            data_dict[key] = np.mean(data_dict[key], axis=0)
            running_avg = np.empty(data_dict[key].shape[0])
            running_avg_upper = np.empty(data_dict[key].shape[0])
            running_avg_lower = np.empty(data_dict[key].shape[0])
            for i in range(data_dict[key].shape[0]):
                running_avg[i] = np.mean(data_dict[key][max(0, i - 2):(i + 1)])
                running_avg_upper[i] = np.mean(upper[max(0, i - 2):(i + 1)])
                running_avg_lower[i] = np.mean(lower[max(0, i - 2):(i + 1)])
            if n > 3:
                running_avg /= (40 * 40)
                running_avg_upper /= (40 * 40)
                running_avg_lower /= (40 * 40)
            xs = np.arange(len(running_avg))
            ax[row, col].plot(xs, running_avg, label=key, color=colour_pool[n], linewidth=2)
            ax[row, col].fill_between(xs, running_avg_lower, running_avg_upper,
                               color=colour_pool[n], alpha=0.2)
            ax[row, col].grid(True)
            handle = Line2D([0], [0], color=colour_pool[n], linewidth=5)
            ax[row, col].legend([handle], [data_name_legend[n]], loc='upper left', fontsize=10,
                                    handlelength=0.1, frameon=True)
            if n in [5, 7]:
                ax[row, col].set_xlim([-3, 39.2])
                ax[row, col].set_xticks([-2, 7.85, 17.9, 28, 38])
            elif n in [4, 6]:
                ax[row, col].set_xlim([-5, 205])
                ax[row, col].set_xticks([0, 49, 99, 149, 199])
            else:
                ax[row, col].set_xlim([-10, 200])
                ax[row, col].set_xticks([-5, 44, 94, 144, 194])

            if n != 6 and n != 7:
                for tick in ax[row, col].xaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            else:
                ax[row, col].set_xlabel('Epoch')
                ax[row, col].set_xticklabels(['1', '50', '100', '150', '200'])
        plt.savefig(os.path.join(script_path, '..', 'figs', f'rl_task{task_id}.pdf'),
                    dpi=300, bbox_inches='tight', pad_inches=0.01)


# plot_rl_loss_curve()