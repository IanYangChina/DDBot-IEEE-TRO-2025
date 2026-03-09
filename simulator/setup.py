from setuptools import setup, find_packages

packages = find_packages()
for p in packages:
    assert p == 'doma' or p.startswith('doma.')

setup(name='doma',
      version='1.0.0',
      description='deformable-object-manipulation-suite',
      url='https://github.com/IanYangChina/deformable-object-manipulation',
      author='Xintong Yang',
      author_email='YangX66@cardiff.ac.uk',
      packages=packages,
      package_dir={'deformable-object-manipulation': 'doma'},
      package_data={'doma': [
          'assets/meshes/processed/*.obj',
          'assets/meshes/processed/*.sdf',
          'assets/meshes/raw/*.obj',
          'assets/meshes/voxelized/*.vox',
          'engine/configs/manipulator_cfgs/*.yaml',
      ]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ]
      )
