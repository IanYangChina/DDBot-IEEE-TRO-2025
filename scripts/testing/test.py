from vedo import Points, show, Mesh
import os
from doma.assets import asset_mesh_dir

mesh_path = os.path.join(asset_mesh_dir, 'raw', 'SoilBox.obj')
mesh = Mesh(mesh_path)
show([mesh], __doc__, axes=True).close()