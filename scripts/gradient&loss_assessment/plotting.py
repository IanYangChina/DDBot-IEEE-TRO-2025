import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.python.summary.summary_iterator import summary_iterator
from drl_implementation.agent.utils import plot as plot
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
fig_path = os.path.join(script_path, '..', 'figs')
os.makedirs(fig_path, exist_ok=True)

colour_pool = ['#dbc6e0', '#d9e8f4', '#fee281', '#c8d9a6', '#dd6e60', '#80a5d0', '#f7e1bd']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"


def read_and_save_data(substep=False):
    d = "5e6"
    for res in ['10', '20', '30', '40', '50', '60']:
        for grad_op in ['gnone', 'gclip', 'gdys', 'gnorm']:
            if substep:
                case_folder = os.path.join(script_path, '..', 'log-substep-grad-analysis', f'd{d}-{grad_op}-res{res}')
            else:
                case_folder = os.path.join(script_path, '..', 'log-grad-analysis', f'd{d}-{grad_op}-res{res}')
            for seed in range(5):
                seed_folder = os.path.join(case_folder, f'seed-{seed}')
                data_dict = {
                    'grid': {
                        'mass_max': [],
                        'mass_min': [],
                        'v_in_max': [],
                        'v_in_min': [],
                        'v_out_max': [],
                        'v_out_min': [],
                    },
                    'p': {
                        'F_max': [],
                        'F_min': [],
                        'v_max': [],
                        'v_min': [],
                        'x_max': [],
                        'x_min': [],
                    },
                    'action':{
                        '0_max': [],
                        '0_min': [],
                        '1_max': [],
                        '1_min': [],
                        '2_max': [],
                        '2_min': [],
                        '3_max': [],
                        '3_min': [],
                        '4_max': [],
                        '4_min': [],
                        '5_max': [],
                        '5_min': [],
                    },
                    'params': {
                        'E': [],
                        'nu': [],
                        'rho': [],
                        'sand_angle': [],
                    },
                    'skill': []
                }

                for filename in os.listdir(seed_folder):
                    if filename[:5] == 'event':
                        for event in summary_iterator(os.path.join(seed_folder, filename)):
                            for v in event.summary.value:
                                if v.tag[:10] == 'Grad-grid/':
                                    if v.tag[10:] == 'mass_max':
                                        data_dict['grid']['mass_max'].append(v.simple_value)
                                    elif v.tag[10:] == 'mass_min':
                                        data_dict['grid']['mass_min'].append(v.simple_value)
                                    elif v.tag[10:] == 'v_in_max':
                                        data_dict['grid']['v_in_max'].append(v.simple_value)
                                    elif v.tag[10:] == 'v_in_min':
                                        data_dict['grid']['v_in_min'].append(v.simple_value)
                                    elif v.tag[10:] == 'v_out_max':
                                        data_dict['grid']['v_out_max'].append(v.simple_value)
                                    elif v.tag[10:] == 'v_out_min':
                                        data_dict['grid']['v_out_min'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:7] == 'Grad-p/':
                                    if v.tag[7:] == 'F_max':
                                        data_dict['p']['F_max'].append(v.simple_value)
                                    elif v.tag[7:] == 'F_min':
                                        data_dict['p']['F_min'].append(v.simple_value)
                                    elif v.tag[7:] == 'v_max':
                                        data_dict['p']['v_max'].append(v.simple_value)
                                    elif v.tag[7:] == 'v_min':
                                        data_dict['p']['v_min'].append(v.simple_value)
                                    elif v.tag[7:] == 'x_max':
                                        data_dict['p']['x_max'].append(v.simple_value)
                                    elif v.tag[7:] == 'x_min':
                                        data_dict['p']['x_min'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:12] == 'Grad_action/':
                                    if v.tag[12:] == '0_max':
                                        data_dict['action']['0_max'].append(v.simple_value)
                                    elif v.tag[12:] == '0_min':
                                        data_dict['action']['0_min'].append(v.simple_value)
                                    elif v.tag[12:] == '1_max':
                                        data_dict['action']['1_max'].append(v.simple_value)
                                    elif v.tag[12:] == '1_min':
                                        data_dict['action']['1_min'].append(v.simple_value)
                                    elif v.tag[12:] == '2_max':
                                        data_dict['action']['2_max'].append(v.simple_value)
                                    elif v.tag[12:] == '2_min':
                                        data_dict['action']['2_min'].append(v.simple_value)
                                    elif v.tag[12:] == '3_max':
                                        data_dict['action']['3_max'].append(v.simple_value)
                                    elif v.tag[12:] == '3_min':
                                        data_dict['action']['3_min'].append(v.simple_value)
                                    elif v.tag[12:] == '4_max':
                                        data_dict['action']['4_max'].append(v.simple_value)
                                    elif v.tag[12:] == '4_min':
                                        data_dict['action']['4_min'].append(v.simple_value)
                                    elif v.tag[12:] == '5_max':
                                        data_dict['action']['5_max'].append(v.simple_value)
                                    elif v.tag[12:] == '5_min':
                                        data_dict['action']['5_min'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:11] == 'Grad_param/':
                                    if v.tag[11:] == 'E':
                                        data_dict['params']['E'].append(v.simple_value)
                                    elif v.tag[11:] == 'nu':
                                        data_dict['params']['nu'].append(v.simple_value)
                                    elif v.tag[11:] == 'rho':
                                        data_dict['params']['rho'].append(v.simple_value)
                                    elif v.tag[11:] == 'sand_angle':
                                        data_dict['params']['sand_angle'].append(v.simple_value)
                                    else:
                                        pass
                                elif v.tag[:11] == 'Grad_skill/':
                                    data_dict['skill'].append(v.simple_value)
                                else:
                                    pass

                json.dump(data_dict, open(os.path.join(seed_folder, 'raw_data.json'), 'w'))


def plot_grads(data=('action', '0_max')):
    save_dir = os.path.join(fig_path, 'grads')
    os.makedirs(save_dir, exist_ok=True)
    d = "5e6"
    for res in ['10', '20', '30', '40', '50', '60']:
        # one fig each res
        grad_datas = []
        for grad_op in ['gnone', 'gclip', 'gdys', 'gnorm']:
            case_folder = os.path.join(script_path, '..', 'log-grad-analysis', f'd{d}-{grad_op}-res{res}')
            grad_data = []
            for seed in range(5):
                seed_folder = os.path.join(case_folder, f'seed-{seed}')
                data_dict = json.load(open(os.path.join(seed_folder, 'raw_data.json'), 'r'))
                grad_data.append(data_dict[data[0]][data[1]])
            grad_data = np.log(np.abs(np.mean(grad_data, axis=0)))
            grad_data = np.nan_to_num(grad_data, nan=100, neginf=-100, posinf=100)
            grad_datas.append(grad_data)

        file_name = os.path.join(save_dir, f'{data[0]}-{data[1]}-res{res}.pdf')
        plot.smoothed_plot_multi_line(file_name, grad_datas, colour_pool[:4],
                                      legend=['None', 'Clip', 'D-scale', 'Norm'],
                                      x_label='Global timestep (reverse)', y_label='Gradient (log-scale)')


# read_and_save_data()
plot_grads(('action', '1_min'))
