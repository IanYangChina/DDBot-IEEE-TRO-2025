import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorflow.python.summary.summary_iterator import summary_iterator
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
colour_pool = ['#a42423', '#ff8a00', '#003153', '#436850', '#7cc2c0', '#b7bc56', '#7a81fc', '#7f4a88']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"
plt.rcParams.update({'font.size': 10})


def find_best_parameters(hm=False, grad_norm=False, grad_dy_scale=False, grad_clip=False,
                         line_search=False, man_init=False,
                         resolutions=[10, 20, 30, 40, 50, 60]):
    """generate mean and deviation data from tensorboard logs"""
    best_params = {"5e6": None}
    for d in ["5e6"]:
        case = f'd{d}'
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
            case_folder = os.path.join(script_path, '..', 'log-sys_id', case+f'-res{res}')
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

                json.dump(data_dict, open(os.path.join(folder, 'raw_data.json'), 'w'))
                min_heightmap_id = np.argmin(data_dict['Validation Loss']['height_map_loss']) - 1
                json.dump({
                    'Step': float(min_heightmap_id),
                    'Loss': {
                        'emd_loss': data_dict['Loss']['emd_loss'][min_heightmap_id],
                        'height_map_loss': data_dict['Loss']['height_map_loss'][min_heightmap_id],
                    },
                    'Validation Loss': {
                        'emd_loss': data_dict['Validation Loss']['emd_loss'][min_heightmap_id],
                        'height_map_loss': data_dict['Validation Loss']['height_map_loss'][min_heightmap_id],
                    },
                    'Parameters': {
                        'E': data_dict['Parameters']['E'][min_heightmap_id],
                        'nu': data_dict['Parameters']['nu'][min_heightmap_id],
                        'rho': data_dict['Parameters']['rho'][min_heightmap_id],
                        'sand_angle': data_dict['Parameters']['sand_angle'][min_heightmap_id],
                    }},
                    open(os.path.join(folder, 'best_heightmap_loss.json'), 'w'))

                if data_dict['Validation Loss']['height_map_loss'][min_heightmap_id] < best_loss:
                    best_loss_info = {
                    'Step': float(min_heightmap_id),
                    'Loss': {
                        'emd_loss': data_dict['Loss']['emd_loss'][min_heightmap_id],
                        'height_map_loss': data_dict['Loss']['height_map_loss'][min_heightmap_id],
                    },
                    'Validation Loss': {
                        'emd_loss': data_dict['Validation Loss']['emd_loss'][min_heightmap_id],
                        'height_map_loss': data_dict['Validation Loss']['height_map_loss'][min_heightmap_id],
                    },
                    'Parameters': {
                        'E': data_dict['Parameters']['E'][min_heightmap_id],
                        'nu': data_dict['Parameters']['nu'][min_heightmap_id],
                        'rho': data_dict['Parameters']['rho'][min_heightmap_id],
                        'sand_angle': data_dict['Parameters']['sand_angle'][min_heightmap_id],
                    }}

            open(os.path.join(case_folder, 'best_params.json'), 'w').write(json.dumps(best_loss_info))


# find_best_parameters(grad_norm=True, line_search=True, resolutions=[40])
# find_best_parameters(grad_dy_scale=True, line_search=True, resolutions=[40])
# find_best_parameters(grad_clip=True, line_search=True, man_init=True, resolutions=[40])
# find_best_parameters(grad_clip=True, line_search=True, hm=True, man_init=True, resolutions=[40])
# exit()


def plot_scatter():
    plt.figure(figsize=(3.5, 4))
    cases = ['gclip', 'gclip-ls', 'gnorm-ls', 'gdys-ls', 'hm-gclip', 'hm-gclip-ls',
             'gclip-ls-man-init', 'hm-gclip-ls-man-init']
    casenames = ['ClipGrad', 'ClipGrad-LS', 'NormGrad-LS', 'DyScaleGrad\n-LS', 'ClipGrad-HM', 'ClipGrad\n-LS-HM',
                 'ClipGrad-LS\n-ManInit', 'ClipGrad-LS\n-HM-ManInit']
    for case_id in range(len(cases)):
        x = []
        x_std = []
        y = []
        y_std = []
        case = cases[case_id]
        for res in [40]:
            case_folder = os.path.join(script_path, '..', 'log-sys_id', f'd5e6-{case}-res{res}')
            best_loss = []
            earliest_epoch = []
            for seed in range(5):
                folder = os.path.join(case_folder, f'seed-{seed}')
                with open(os.path.join(folder, 'best_heightmap_loss.json')) as f:
                    data = json.load(f)
                    earliest_epoch.append(data['Step'])
                    if case == 'gclip' or case == 'hm-gclip':
                        best_loss.append(data['Validation Loss']['height_map_loss'] / (60*60) + data['Validation Loss']['emd_loss'] / (60*60))
                    else:
                        best_loss.append(data['Validation Loss']['height_map_loss'] / (40*40) + data['Validation Loss']['emd_loss'] / (40*40))
            y.append(np.mean(best_loss))
            y_std.append(np.std(best_loss))
            x.append(np.mean(earliest_epoch))
            x_std.append(np.std(earliest_epoch))
        plt.errorbar(y, len(cases)-case_id-1, label=casenames[case_id], color=colour_pool[case_id], xerr=y_std, fmt='h',
                     capsize=5, markersize=8, elinewidth=2, capthick=2)

    plt.grid(True, axis='x')
    casenames.reverse()
    plt.xticks([0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030], rotation=60)
    plt.yticks(np.arange(len(cases)), casenames)
    plt.xlabel('Best validation loss across 5 seeds')
    plt.savefig(os.path.join(script_path, '..', 'figs', 'sys_id.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.05)


plot_scatter()


def plot_loss_curve():
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(1, 8, figsize=(16, 2))
    plt.subplots_adjust(wspace=0, hspace=0.1)
    cases = ['gclip',
             'gclip-ls',
             'gnorm-ls',
             'gdys-ls',
             'hm-gclip', 'hm-gclip-ls',
             'gclip-ls-man-init', 'hm-gclip-ls-man-init'
             ]
    casenames = ['ClipGrad',
                 'ClipGrad-LS',
                 'NormGrad-LS',
                 'DyScaleGrad-LS',
                 'ClipGrad-HM', 'ClipGrad-LS-HM',
                 'ClipGrad-LS\n-ManInit', 'ClipGrad-LS-HM\n-ManInit'
                 ]
    window = 2
    for case_id in range(len(cases)):
        case = cases[case_id]
        for res in [40]:
            case_folder = os.path.join(script_path, '..', 'log-sys_id', f'd5e6-{case}-res{res}')
            losses = []
            for seed in range(5):
                folder = os.path.join(case_folder, f'seed-{seed}')
                with open(os.path.join(folder, 'raw_data.json')) as f:
                    data = json.load(f)
                    if case == 'gclip' or case == 'hm-gclip':
                        losses.append(np.asarray(data['Validation Loss']['height_map_loss']) / (60*60) +
                                      np.asarray(data['Validation Loss']['emd_loss']) / (60*60))
                    else:
                        losses.append(np.asarray(data['Validation Loss']['height_map_loss']) / (40*40) +
                                      np.asarray(data['Validation Loss']['emd_loss']) / (40*40))
            mean_loss = np.mean(losses, axis=0)
            running_avg = np.empty(mean_loss.shape[0])
            for n in range(mean_loss.shape[0]):
                running_avg[n] = np.mean(mean_loss[max(0, n - window):(n + 1)])
            xs = np.arange(len(running_avg))
            ax[case_id].plot(xs, running_avg, label=casenames[case_id], color=colour_pool[case_id], linewidth=2)
            for l in losses:
                running_avg_l = np.empty(l.shape[0])
                for n in range(l.shape[0]):
                    running_avg_l[n] = np.mean(l[max(0, n - window):(n + 1)])
                ax[case_id].plot(xs, running_avg_l, color=colour_pool[case_id], linestyle='--', linewidth=1)

            ax[case_id].set_ylim([0.0235, 0.033])
            ax[case_id].set_xlim([-2, 22])
            ax[case_id].set_xticks([0, 4, 9, 14, 19])
            ax[case_id].set_xticklabels(['1', '5', '10', '15', '20'])
            ax[case_id].grid(True)
            if case_id == 0:
                ax[case_id].set_ylabel('Validation loss')
                ax[case_id].spines[['right']].set_visible(False)
            elif case_id == len(cases) - 1:
                for tick in ax[case_id].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            else:
                ax[case_id].spines[['right']].set_visible(False)
                for tick in ax[case_id].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
            ax[case_id].set_xlabel('Epoch')

            handle = Line2D([0], [0], color=colour_pool[case_id], linewidth=5)
            if case_id < 6:
                ax[case_id].legend([handle], [casenames[case_id]], loc='upper left', fontsize=10, bbox_to_anchor=(-0.04, 1.23),
                                   handlelength=0.1, frameon=False)
            else:
                ax[case_id].legend([handle], [casenames[case_id]], loc='upper left', fontsize=10,
                                   bbox_to_anchor=(-0.04, 1.28),
                                   handlelength=0.1, frameon=False)

    plt.savefig(os.path.join(script_path, '..', 'figs', 'sys_id_loss_curve.pdf'),
                dpi=300, bbox_inches='tight', pad_inches=0.05)


plot_loss_curve()


def find_best_skill_parameters(ins=False, hm=False, main_folder_suffix='', lr='',
                               demo=False, batch=False, seeds=None, task_id=0):
    best_params = {"5e6": None,
                   "1e7": None}
    optimiser = 'rmsprop-b0.9'
    subfix = ''
    if ins:
        subfix += '-ins'
    if hm:
        subfix += '-hm'
    if demo:
        subfix += '-demo'
    if batch:
        subfix += '-batch'

    if lr is not '':
        subfix += f'-lr{lr}'

    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    for d in ["5e6"]:
        for seed in seeds:
            folder = os.path.join(script_path, '..', f'log-abs2-{optimiser}{main_folder_suffix}',
                                  f'd{d}-task-{task_id}{subfix}', f'seed-{seed}')
            data_dict = {
                'Loss': {
                    'emd_loss': [],
                    'height_map_loss': [],
                },
                'Parameters': {
                    'skill_params_0': [],
                    'skill_params_1': [],
                    'skill_params_2': [],
                    'skill_params_3': [],
                    'skill_params_4': [],
                }
            }
            for filename in os.listdir(folder):
                if filename[:5] == 'event':
                    for event in summary_iterator(os.path.join(folder, filename)):
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

            json.dump(data_dict, open(os.path.join(folder, 'raw_data.json'), 'w'))
            min_heightmap_id = np.argmin(data_dict['Loss']['height_map_loss'])
            json.dump({
                'Step': float(min_heightmap_id),
                'Loss': {
                    'emd_loss': data_dict['Loss']['emd_loss'][min_heightmap_id],
                    'height_map_loss': data_dict['Loss']['height_map_loss'][min_heightmap_id],
                },
                'Parameters': {
                    'skill_params_0': data_dict['Parameters']['skill_params_0'][min_heightmap_id],
                    'skill_params_1': data_dict['Parameters']['skill_params_1'][min_heightmap_id],
                    'skill_params_2': data_dict['Parameters']['skill_params_2'][min_heightmap_id],
                    'skill_params_3': data_dict['Parameters']['skill_params_3'][min_heightmap_id],
                    'skill_params_4': data_dict['Parameters']['skill_params_4'][min_heightmap_id],
                }},
                open(os.path.join(folder, 'best_heightmap_loss.json'), 'w'))
            min_emd_id = np.argmin(data_dict['Loss']['emd_loss'])
            json.dump({
                'Step': float(min_emd_id),
                'Loss': {
                    'emd_loss': data_dict['Loss']['emd_loss'][min_emd_id],
                    'height_map_loss': data_dict['Loss']['height_map_loss'][min_emd_id],
                },
                'Parameters': {
                    'skill_params_0': data_dict['Parameters']['skill_params_0'][min_emd_id],
                    'skill_params_1': data_dict['Parameters']['skill_params_1'][min_emd_id],
                    'skill_params_2': data_dict['Parameters']['skill_params_2'][min_emd_id],
                    'skill_params_3': data_dict['Parameters']['skill_params_3'][min_emd_id],
                    'skill_params_4': data_dict['Parameters']['skill_params_4'][min_emd_id],
                }},
                open(os.path.join(folder, 'best_emd_loss.json'), 'w'))

        best_heightmap_loss = 1000
        seed_0 = 0
        best_emd_loss = 1000
        seed_1 = 0
        for seed in seeds:
            with open(os.path.join(script_path, '..', f'log-abs2-{optimiser}{main_folder_suffix}', f'd{d}-task-{task_id}{subfix}', f'seed-{seed}',
                                   'best_heightmap_loss.json')) as f:
                data_0 = json.load(f)
                if data_0['Loss']['height_map_loss'] < best_heightmap_loss:
                    best_heightmap_loss = data_0['Loss']['height_map_loss']
                    seed_0 = seed

            with open(os.path.join(script_path, '..', f'log-abs2-{optimiser}{main_folder_suffix}', f'd{d}-task-{task_id}{subfix}', f'seed-{seed}',
                                   'best_emd_loss.json')) as f:
                data_1 = json.load(f)
                if data_1['Loss']['emd_loss'] < best_emd_loss:
                    best_emd_loss = data_1['Loss']['emd_loss']
                    seed_1 = seed

        with open(os.path.join(script_path, '..', f'log-abs2-{optimiser}{main_folder_suffix}', f'd{d}-task-{task_id}{subfix}', f'seed-{seed_0}',
                               'best_heightmap_loss.json')) as f:
            data_0 = json.load(f)
            print(f"Ptcl: {d}, best skill params with the lowest heightmap loss:")
            print(data_0['Loss']['height_map_loss'], '&', data_0['Loss']['emd_loss'], '&', data_0['Parameters']['skill_params_0'], '&',
                  data_0['Parameters']['skill_params_1'], '&', data_0['Parameters']['skill_params_2'], '&', data_0['Parameters']['skill_params_3'], '&', data_0['Parameters']['skill_params_4'])
            best_params[d] = data_0

    open(os.path.join(script_path, '..', f'log-abs2-{optimiser}{main_folder_suffix}', f'best_params-task-{task_id}{subfix}.json'), 'w').write(json.dumps(best_params))


# find_best_skill_parameters(hm=False, demo=True, ins=True,
#                            seeds=[0, 1, 2, 3, 4], task_id=0,
#                            main_folder_suffix='', lr='0.02')


def show_heightmaps(param='ER'):
    if 'skill' not in param:
        if param == 'ER':
            X = np.arange(2.5e5, 1e6, 1.5e4)
            Y = np.arange(1200, 2200, 20)
        else:
            X = np.arange(0.1, 0.4, (0.4-0.1)/50)
            Y = np.arange(10, 40, (40-10)/50)
        X, Y = np.meshgrid(X, Y)
        res = [10, 20, 30, 40, 50, 60]
        for i in range(6):
            loss_type = 'emd'
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.view_init(44, 58)
            ax.set_xlabel('Young\'s Modulus')
            ax.set_ylabel('Material Density')
            ax.set_zlabel(f'{loss_type.upper()} Loss')
            ax.set_title(f'{loss_type.upper()} loss landscape with height grid resolution {res[i]}')
            hm = np.load(os.path.join(script_path, '..', 'log-loss-analysis', 'd5e6',
                                      f'{loss_type}_losses-res{res[i]}-{param}.npy'))
            hm /= (res[i] ** 2)
            surf = ax.plot_surface(X, Y, hm, rstride=1, cstride=1,
                                   cmap='viridis', edgecolor='none')
            fig.colorbar(surf)
            plt.show()
            plt.close()

            loss_type = 'hm'
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.view_init(44, 58)
            ax.set_xlabel('Young\'s Modulus')
            ax.set_ylabel('Material Density')
            ax.set_zlabel(f'{loss_type.upper()} Loss')
            ax.set_title(f'{loss_type.upper()} loss landscape with height grid resolution {res[i]}')
            hm = np.load(os.path.join(script_path, '..', 'log-loss-analysis', 'd5e6',
                                      f'{loss_type}_losses-res{res[i]}-{param}.npy'))
            hm /= (res[i] ** 2)
            surf = ax.plot_surface(X, Y, hm, rstride=1, cstride=1,
                                   cmap='viridis', edgecolor='none')
            fig.colorbar(surf)
            plt.show()
            plt.close()
    else:
        res = [10, 20, 30, 40, 50, 60]
        for n in range(5):
            fig, ax = plt.subplots(4, 3, figsize=(18, 12))
            loss_type = 'emd'
            for i in range(6):
                loss = np.load(os.path.join(script_path, '..', 'log-loss-analysis', 'd5e6',
                                            f'{loss_type}_losses-res{res[i]}-skill-{n}.npy'))
                loss /= (res[i]**2)
                ax[i//3, i % 3].plot(loss)
                ax[i//3, i % 3].set_title(f'Resolution {res[i]} ({int(res[i]**2)} points)')
                ax[i//3, i % 3].set_ylabel(f'Avg. {loss_type.upper()} loss (m)')
                ax[i//3, i % 3].set_xticks([])
            loss_type = 'hm'
            for i in range(6, 12):
                loss = np.load(os.path.join(script_path, '..', 'log-loss-analysis', 'd5e6',
                                            f'{loss_type}_losses-res{res[i-6]}-skill-{n}.npy'))
                loss /= (res[i-6]**2)
                ax[i//3, i % 3].plot(loss)
                ax[i//3, i % 3].set_title(f'Resolution {res[i-6]}')
                ax[i//3, i % 3].set_ylabel(f'Avg. {loss_type.upper()} loss (m)')
                ax[i//3, i % 3].set_xticks([])
                if i >= 9:
                    ax[i//3, i % 3].set_xlabel('Skill parameter')
                    ax[i//3, i % 3].set_xticks([0, 50, 100, 150, 200], ['-1.0', '-0.5', '0.0', '0.5', '1.0'])
            plt.show()
            plt.close()


