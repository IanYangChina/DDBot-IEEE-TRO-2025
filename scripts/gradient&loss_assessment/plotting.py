import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from tensorflow.python.summary.summary_iterator import summary_iterator
from drl_implementation.agent.utils import plot as plot
script_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.join(script_path, '..')
fig_path = os.path.join(script_path, '..', 'figs')
os.makedirs(fig_path, exist_ok=True)

colour_pool = ['#a42423', '#ff8a00', '#003153', '#436850', '#7cc2c0', '#b7bc56', '#7a81fc', '#7f4a88']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams["font.weight"] = "normal"
plt.rcParams.update({'font.size': 10})
mpl.use('Agg')


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


def plot_grads():
    plt.rcParams.update({'font.size': 8})
    save_dir = os.path.join(fig_path, 'grads')
    os.makedirs(save_dir, exist_ok=True)
    d = "5e6"
    res = '40'
    params = ['E', 'nu', 'rho', 'sand_angle']
    params_labels = [r'$\mathbf{\nabla E}$',
                     r'$\mathbf{\nabla \nu}$',
                     r'$\mathbf{\nabla \rho}$',
                     r'$\mathbf{\nabla\phi_f}$']
    p_vars = ['F', 'v', 'x']
    p_vars_labels = [r'$\mathbf{\nabla F_p}$',
                     r'$\mathbf{\nabla v_p}$',
                     r'$\mathbf{\nabla x_p}$']
    grid_vars = ['mass', 'v_in', 'v_out']
    grid_vars_labels = [r'$\mathbf{\nabla m_{grid}}$',
                        r'$\mathbf{\nabla v_{grid}}$',
                        r'$\mathbf{\nabla v^{\prime}_{grid}}$']
    action_vars = ['0', '1', '2', '3', '4', '5']
    action_vars_labels = [r'$\mathbf{\nabla a[0]}$',
                          r'$\mathbf{\nabla a[1]}$',
                          r'$\mathbf{\nabla a[2]}$',
                          r'$\mathbf{\nabla a[3]}$',
                          r'$\mathbf{\nabla a[4]}$',
                          r'$\mathbf{\nabla a[5]}$']

    subfig_height = 1.1

    for var_name in ['grid', 'p', 'action', 'params']:
        y_lim_margin = 5
        n_global_steps = 200
        x = [i for i in range(n_global_steps+1)]
        tick_interval = 50
        x_ticks = np.arange(0, n_global_steps+1, tick_interval)
        x_tick_labels = [str(i) for i in x_ticks]
        x_tick_labels.reverse()
        y_ticks = [-25, -10, 0, 10, 25]
        y_tick_labels = ['-Inf', '-5', '0', '5', 'NaN/Inf']
        ylims = [y_ticks[0] - y_lim_margin, y_ticks[-1] + y_lim_margin]
        if var_name == 'grid':
            vars_to_plot = grid_vars
            var_names = grid_vars_labels
        elif var_name == 'p':
            vars_to_plot = p_vars
            var_names = p_vars_labels
            y_ticks = [0, 5, 20, 25]
            y_tick_labels = ['0', '5', '20', 'NaN/Inf']
            ylims = [y_ticks[0] - y_lim_margin, y_ticks[-1] + y_lim_margin]
        elif var_name == 'params':
            vars_to_plot = params
            var_names = params_labels
            y_ticks = [-5, 0, 10, 25]
            y_tick_labels = ['-5', '0', '5', 'NaN/Inf']
            ylims = [y_ticks[0] - y_lim_margin, y_ticks[-1] + y_lim_margin]
        else:
            vars_to_plot = action_vars
            var_names = action_vars_labels

        grad_ops = ['gnone', 'gclip', 'gdys', 'gnorm']
        fig, ax = plt.subplots(len(vars_to_plot), 2, figsize=(3, subfig_height*len(vars_to_plot)))
        plt.subplots_adjust(wspace=0.27, hspace=0)
        for i in range(len(vars_to_plot)):
            for j in range(len(grad_ops)):
                grad_op = grad_ops[j]
                case_folder = os.path.join(script_path, '..', 'log-grad-analysis', f'd{d}-{grad_op}-res{res}')
                grad_data = []
                grad_data_ = []
                for seed in range(5):
                    seed_folder = os.path.join(case_folder, f'seed-{seed}')
                    data_dict = json.load(open(os.path.join(seed_folder, 'raw_data.json'), 'r'))
                    if var_name == 'params':
                        grad_data.append(data_dict[var_name][vars_to_plot[i]][:n_global_steps+1])
                    else:
                        grad_data.append(data_dict[var_name][vars_to_plot[i]+'_max'][:n_global_steps+1])
                        grad_data_.append(data_dict[var_name][vars_to_plot[i]+'_min'][:n_global_steps+1])
                grad_data = np.log10(np.abs(np.asarray(grad_data)))
                grad_data = np.mean(grad_data, axis=0)
                grad_data = np.nan_to_num(grad_data, nan=25, neginf=-25, posinf=25)
                ax[i, 0].plot(x, grad_data[:n_global_steps+1], colour_pool[j], linewidth=1)
                if var_name != 'params':
                    grad_data_ = np.log10(np.abs(np.asarray(grad_data_)))
                    grad_data_ = np.mean(grad_data_, axis=0)
                    grad_data_ = np.nan_to_num(grad_data_, nan=25, neginf=-25, posinf=25)
                    ax[i, 0].plot(x, grad_data_[:n_global_steps+1], colour_pool[j], linewidth=1, linestyle='--')
                ax[i, 0].set_ylabel(var_names[i], rotation=0, loc='bottom', fontsize=9)
                ax[i, 0].yaxis.set_label_coords(-0.5, 0.45, transform=None)
                ax[i, 0].grid(True)
                ax[i, 0].set_xticks(x_ticks)
                ax[i, 0].set_yticks(y_ticks)
                ax[i, 0].set_yticklabels(y_tick_labels)
                ax[i, 0].set_ylim(ylims[0], ylims[1])
                ax[i, 0].set_xlim(-5, n_global_steps+5)
                if vars_to_plot[i] == 'v_out':
                    ax[i, 0].set_yticks([-25, -10, -5])
                    ax[i, 0].set_yticklabels(['       -Inf', '-10', '-5'])
                    ax[i, 0].set_ylim(-27, -2)
                if i == len(vars_to_plot) - 1:
                    ax[i, 0].set_xlabel('Global timestep (reverse)')
                    ax[i, 0].set_xticklabels(x_tick_labels)
                else:
                    ax[i, 0].spines[['bottom']].set_visible(False)
                    for tick in ax[i, 0].xaxis.get_major_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)

        n_global_steps = 98
        x = [i for i in range(n_global_steps)]
        tick_interval = 25
        x_ticks = np.arange(0, 101, tick_interval)
        x_tick_labels = [str(i) for i in x_ticks]
        x_tick_labels.reverse()
        if var_name == 'params':
            y_ticks = [-5, 0, 5, 15]
            y_tick_labels = ['-5', '0', '5', '15']
            ylims = [y_ticks[0] - y_lim_margin, y_ticks[-1] + y_lim_margin]
        elif var_name == 'p':
            y_ticks = [0, 5, 20]
            y_tick_labels = ['0', '5', '20']
            ylims = [y_ticks[0] - y_lim_margin, y_ticks[-1] + y_lim_margin]
        elif var_name == 'grid':
            y_ticks = [0, 5, 20]
            y_tick_labels = ['0', '5', '20']
            ylims = [y_ticks[0] - y_lim_margin, y_ticks[-1] + y_lim_margin]
        else:
            y_ticks = [-25, -5, 0, 10]
            y_tick_labels = ['-Inf', '-5', '0', '10']
            ylims = [y_ticks[0] - y_lim_margin, y_ticks[-1] + y_lim_margin]
        for i in range(len(vars_to_plot)):
            for j in range(len(grad_ops)):
                grad_op = grad_ops[j]
                case_folder = os.path.join(script_path, '..', 'log-substep-grad-analysis', f'd{d}-{grad_op}-res{res}')
                grad_data = []
                grad_data_ = []
                for seed in range(5):
                    seed_folder = os.path.join(case_folder, f'seed-{seed}')
                    data_dict = json.load(open(os.path.join(seed_folder, 'raw_data.json'), 'r'))
                    if var_name == 'params':
                        grad_data.append(data_dict[var_name][vars_to_plot[i]][:n_global_steps+1])
                    else:
                        grad_data.append(data_dict[var_name][vars_to_plot[i]+'_max'][:n_global_steps])
                        grad_data_.append(data_dict[var_name][vars_to_plot[i]+'_min'][:n_global_steps])
                grad_data = np.log10(np.abs(np.asarray(grad_data)))
                grad_data = np.mean(grad_data, axis=0)
                grad_data = np.nan_to_num(grad_data, nan=25, neginf=-25, posinf=25)
                ax[i, 1].plot(x, grad_data[:n_global_steps], colour_pool[j], linewidth=1)
                if var_name != 'params':
                    grad_data_ = np.log10(np.abs(np.asarray(grad_data_)))
                    grad_data_ = np.mean(grad_data_, axis=0)
                    grad_data_ = np.nan_to_num(grad_data_, nan=25, neginf=-25, posinf=25)
                    ax[i, 1].plot(x, grad_data_[:n_global_steps], colour_pool[j], linewidth=1, linestyle='--')
                ax[i, 1].grid(True)
                ax[i, 1].set_xticks(x_ticks)
                ax[i, 1].set_yticks(y_ticks)
                ax[i, 1].set_yticklabels(y_tick_labels)
                ax[i, 1].set_ylim(ylims[0], ylims[1])
                ax[i, 1].set_xlim(-5, n_global_steps+5)
                if vars_to_plot[i] == 'v_out':
                    ax[i, 1].set_yticks([-25, -5, 0, 5])
                    ax[i, 1].set_yticklabels(['-Inf', '-5', '0', '5'])
                    ax[i, 1].set_ylim(-27, 7)
                if i == len(vars_to_plot) - 1:
                    ax[i, 1].set_xlabel('Last 100 substeps (reverse)')
                    ax[i, 1].set_xticklabels(x_tick_labels)
                else:
                    ax[i, 1].spines[['bottom']].set_visible(False)
                    for tick in ax[i, 1].xaxis.get_major_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)

        file_name = os.path.join(save_dir, f'{var_name}-res{res}.pdf')
        if var_name == 'params':
            legends = ['No operation', 'Clipping', 'Dynamic scaling', 'Normalization']
            handles = [Line2D([0], [0], color=colour_pool[i], linewidth=4) for i in range(len(legends))]
            plt.legend(handles, legends, handlelength=2, fontsize=10,
                                     title=None, loc="upper left", labelspacing=0.2,
                                     bbox_to_anchor=(-1.35, 5), ncol=1, frameon=False)
        plt.savefig(file_name, format='pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)


def plot_legends():
    legends = ['No operation', 'Clipping', 'Dynamic scaling', 'Normalization']
    plt.rcParams.update({'font.size': 40})
    handles = [Line2D([0], [0], color=colour_pool[i], linewidth=15) for i in range(len(legends))]
    legend_plot = plt.legend(handles, legends, handlelength=1,
                                 title=None, loc="upper right", labelspacing=0.15,
                                 bbox_to_anchor=(2, 2), ncol=1, frameon=False)
    fig = legend_plot.figure
    fig.canvas.draw()
    bbox = legend_plot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(fig_path, 'grads', 'legend.pdf'), dpi=500, bbox_inches=bbox)
    plt.close()


# read_and_save_data()
# plot_legends()
plot_grads()