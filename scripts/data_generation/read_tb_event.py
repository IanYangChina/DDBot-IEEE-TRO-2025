import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
colour_pool = ['#dbc6e0', '#d9e8f4', '#fee281', '#c8d9a6', '#dd6e60', '#80a5d0', '#f7e1bd']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"


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


find_best_parameters(grad_clip=True, line_search=True, resolutions=[40])
find_best_parameters(grad_clip=True, line_search=True, hm=True, resolutions=[40])
find_best_parameters(grad_clip=True, line_search=True, man_init=True, resolutions=[40])
find_best_parameters(grad_clip=True, line_search=True, hm=True, man_init=True, resolutions=[40])
exit()


def plot_scatter():
    x = np.arange(6*6)
    cases = ['gclip', 'gclip-ls', 'hm-gclip', 'hm-gclip-ls',
             'gclip-ls-man-init', 'hm-gclip-ls-man-init']
    x_labels = []
    for case_id in range(len(cases)):
        y = []
        y_std = []
        case = cases[case_id]
        for res in [10, 20, 30, 40, 50, 60]:
            case_folder = os.path.join(script_path, '..', 'log-sys_id', f'd5e6-{case}-res{res}')
            best_loss = []
            for seed in range(5):
                folder = os.path.join(case_folder, f'seed-{seed}')
                with open(os.path.join(folder, 'best_heightmap_loss.json')) as f:
                    data = json.load(f)
                    best_loss.append(data['Validation Loss']['height_map_loss'])
            y.append(np.mean(best_loss))
            y_std.append(np.std(best_loss))
            x_labels.append(f'{case}-{res}')
        plt.errorbar(x[case_id*6:(case_id+1)*6], y, label=case, color=colour_pool[case_id], yerr=y_std, fmt='o')
    plt.legend()
    plt.xticks(x, x_labels)
    plt.xticks(rotation=70)
    plt.xlabel('Case-Resolution')
    plt.ylabel('Best validation heightmap loss')
    plt.tight_layout()
    plt.show()


plot_scatter()


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


