import os
import json
import numpy as np
from drl_implementation.agent.utils import plot as plot
from tensorflow.python.summary.summary_iterator import summary_iterator
script_path = os.path.dirname(os.path.realpath(__file__))


def find_best_parameters(ptcl_d="1e7", ns="20"):
    """generate mean and deviation data from tensorboard logs"""
    case = f'd{ptcl_d}_ns{ns}'
    for seed in [0, 1, 2, 4, 5]:
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


best_params = {"5e6": None,
               "1e7": None,
               "2e7": None}
for d in ["5e6", "1e7", "2e7"]:
    best_heightmap_loss = 1000
    seed_0 = 0
    best_emd_loss = 1000
    seed_1 = 0
    for seed in ["0", "1", "2", "4", "5"]:
        print(f"Parameters with best heightmap loss for particle density {d}, seed {seed}:")
        with open(os.path.join(script_path, '..', 'log-sys_id', f'd{d}_ns20', f'seed-{seed}',
                               'best_heightmap_loss.json')) as f:
            data_0 = json.load(f)
            if data_0['Loss']['height_map_loss'] < best_heightmap_loss:
                best_heightmap_loss = data_0['Loss']['height_map_loss']
                seed_0 = seed

        print(f"Parameters with best emd loss for particle density {d}, seed {seed}:")
        with open(os.path.join(script_path, '..', 'log-sys_id', f'd{d}_ns20', f'seed-{seed}',
                               'best_emd_loss.json')) as f:
            data_1 = json.load(f)
            if data_1['Loss']['emd_loss'] < best_emd_loss:
                best_emd_loss = data_1['Loss']['emd_loss']
                seed_1 = seed

    with open(os.path.join(script_path, '..', 'log-sys_id', f'd{d}_ns20', f'seed-{seed_0}', 'best_heightmap_loss.json')) as f:
        data_0 = json.load(f)
        print(data_0['Loss']['height_map_loss'], '&', data_0['Loss']['emd_loss'], '&', data_0['Parameters']['E'], '&',
              data_0['Parameters']['nu'], '&', data_0['Parameters']['rho'], '&', data_0['Parameters']['sand_angle'])
        best_params[d] = data_0

open(os.path.join(script_path, '..', 'log-sys_id', 'best_params.json'), 'w').write(json.dumps(best_params))
