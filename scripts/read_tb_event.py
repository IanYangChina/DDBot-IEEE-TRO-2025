import os
import json
import numpy as np
from drl_implementation.agent.utils import plot as plot
from tensorflow.python.summary.summary_iterator import summary_iterator
script_path = os.path.dirname(os.path.realpath(__file__))


def find_best_parameters():
    """generate mean and deviation data from tensorboard logs"""
    best_params = {"5e6": None,
                   "1e7": None,
                   "2e7": None,
                   "4e7": None,}
    for d in ["5e6", "1e7", "2e7", "4e7"]:
        case = f'd{d}_ns20'
        for seed in range(5):
            folder = os.path.join(script_path, '..', 'log-sys_id', case, f'seed-{seed}')
            data_dict = {
                'Loss': {
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
            min_heightmap_id = np.argmin(data_dict['Loss']['height_map_loss'])
            json.dump({
                'Step': float(min_heightmap_id),
                'Loss': {
                    'emd_loss': data_dict['Loss']['emd_loss'][min_heightmap_id],
                    'height_map_loss': data_dict['Loss']['height_map_loss'][min_heightmap_id],
                },
                'Parameters': {
                    'E': data_dict['Parameters']['E'][min_heightmap_id],
                    'nu': data_dict['Parameters']['nu'][min_heightmap_id],
                    'rho': data_dict['Parameters']['rho'][min_heightmap_id],
                    'sand_angle': data_dict['Parameters']['sand_angle'][min_heightmap_id],
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
                    'E': data_dict['Parameters']['E'][min_emd_id],
                    'nu': data_dict['Parameters']['nu'][min_emd_id],
                    'rho': data_dict['Parameters']['rho'][min_emd_id],
                    'sand_angle': data_dict['Parameters']['sand_angle'][min_emd_id],
                }},
                open(os.path.join(folder, 'best_emd_loss.json'), 'w'))

        best_heightmap_loss = 1000
        seed_0 = 0
        best_emd_loss = 1000
        seed_1 = 0
        for seed in range(5):
            with open(os.path.join(script_path, '..', 'log-sys_id', f'd{d}_ns20', f'seed-{seed}',
                                   'best_heightmap_loss.json')) as f:
                data_0 = json.load(f)
                if data_0['Loss']['height_map_loss'] < best_heightmap_loss:
                    best_heightmap_loss = data_0['Loss']['height_map_loss']
                    seed_0 = seed

            with open(os.path.join(script_path, '..', 'log-sys_id', f'd{d}_ns20', f'seed-{seed}',
                                   'best_emd_loss.json')) as f:
                data_1 = json.load(f)
                if data_1['Loss']['emd_loss'] < best_emd_loss:
                    best_emd_loss = data_1['Loss']['emd_loss']
                    seed_1 = seed

        with open(os.path.join(script_path, '..', 'log-sys_id', f'd{d}_ns20', f'seed-{seed_0}',
                               'best_heightmap_loss.json')) as f:
            data_0 = json.load(f)
            print(data_0['Loss']['height_map_loss'],
                  '&', data_0['Loss']['emd_loss'],
                  '&', data_0['Parameters']['E'],
                  '&', data_0['Parameters']['nu'],
                  '&', data_0['Parameters']['rho'],
                  '&', data_0['Parameters']['sand_angle'])
            best_params[d] = data_0

        open(os.path.join(script_path, '..', 'log-sys_id', 'best_params.json'), 'w').write(json.dumps(best_params))


def find_best_skill_parameters(demo=False, demo_batch=False, seeds=None, task_id=0):
    best_params = {"5e6": None,
                   "1e7": None}
    if demo:
        subfix = '-demo'
    elif demo_batch:
        subfix = '-demo-batch'
    else:
        subfix = ''
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    for d in ["5e6"]:
        for seed in seeds:
            folder = os.path.join(script_path, '..', 'log-abs2-adam',
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
            with open(os.path.join(script_path, '..', 'log-abs2-adam', f'd{d}-task-{task_id}{subfix}', f'seed-{seed}',
                                   'best_heightmap_loss.json')) as f:
                data_0 = json.load(f)
                if data_0['Loss']['height_map_loss'] < best_heightmap_loss:
                    best_heightmap_loss = data_0['Loss']['height_map_loss']
                    seed_0 = seed

            with open(os.path.join(script_path, '..', 'log-abs2-adam', f'd{d}-task-{task_id}{subfix}', f'seed-{seed}',
                                   'best_emd_loss.json')) as f:
                data_1 = json.load(f)
                if data_1['Loss']['emd_loss'] < best_emd_loss:
                    best_emd_loss = data_1['Loss']['emd_loss']
                    seed_1 = seed

        with open(os.path.join(script_path, '..', 'log-abs2-adam', f'd{d}-task-{task_id}{subfix}', f'seed-{seed_0}',
                               'best_heightmap_loss.json')) as f:
            data_0 = json.load(f)
            print(f"Ptcl: {d}, best skill params with the lowest heightmap loss:")
            print(data_0['Loss']['height_map_loss'], '&', data_0['Loss']['emd_loss'], '&', data_0['Parameters']['skill_params_0'], '&',
                  data_0['Parameters']['skill_params_1'], '&', data_0['Parameters']['skill_params_2'], '&', data_0['Parameters']['skill_params_3'], '&', data_0['Parameters']['skill_params_4'])
            best_params[d] = data_0

    open(os.path.join(script_path, '..', 'log-abs2-adam', f'best_params-task-{task_id}{subfix}.json'), 'w').write(json.dumps(best_params))


find_best_skill_parameters(demo=False, seeds=[0, 1, 2, 3, 4], task_id=0)
