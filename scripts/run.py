import os
import yaml
import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
script_path = os.path.dirname(os.path.realpath(__file__))

ti.init(arch=ti.cuda, device_memory_GB=15, default_fp=ti.f32, fast_math=False)
from doma.envs import PlantingEnv
cam_cfg = {
    'pos': (-0.2, -0.2, 0.7),
    'lookat': (0.2, 0.2, 0.03),
    'fov': 30,
    'lights': [{'pos': (-1.2, 0.25, 0.2), 'color': (0.6, 0.6, 0.6)},
               {'pos': (-1.2, 0.5, 1.0), 'color': (0.6, 0.6, 0.6)},
               {'pos': (-1.2, 0.0, 1.0), 'color': (0.8, 0.8, 0.8)}],
    'particle_radius': 0.001,
    'res': (640, 640)
}


def get_skill_trajectory(skill, cur_pose, params=None):
    assert skill in ['insert', 'pullout', 'push-forward', 'gather', 'press']
    if params is None:
        with open(os.path.join(script_path, '..', 'data', 'skills.yaml'), 'r') as f:
            skill_params = yaml.safe_load(f)
        params = skill_params[skill]

    soil_height = 0.095

    if skill == 'insert':
        # set the initial pose of the end effector
        init_pos = np.concatenate((params['xy_location'], [soil_height]))
        init_euler = (0, 180+params['angle'], 90)
        # calculate trajectory points based on the given direction
        distance = params['distance']



if __name__ == '__main__':
    get_skill_trajectory('insert')