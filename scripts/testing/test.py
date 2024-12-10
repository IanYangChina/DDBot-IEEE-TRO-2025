#from vedo import Points, show, Mesh
import os
# from doma.assets import asset_mesh_dir
#
# mesh_path = os.path.join(asset_mesh_dir, 'raw', 'ShovelEEF.obj')
# mesh = Mesh(mesh_path)
# show([mesh], __doc__, axes=True).close()

import json
script_path = os.path.dirname(os.path.realpath(__file__))
mat = ''
sys_id_folder = os.path.join(script_path, '..', '..', f'log-sys_id{mat}')

for case in [
    'd5e6-gclip-ls-res40',
    'd5e6-hm-gclip-ls-res40',
    'd5e6-gclip-ls-man-init-res40',
    'd5e6-hm-gclip-ls-man-init-res40'
]:
    with open(os.path.join(sys_id_folder, case, 'best_params.json'), 'r') as f:
        data = json.load(f)
    print(
        data['Validation Loss']['total_loss'], '&',
        data['Parameters']['E'], '&',
        data['Parameters']['nu'], '&',
        data['Parameters']['rho'], '&',
        data['Parameters']['sand_angle']
    )