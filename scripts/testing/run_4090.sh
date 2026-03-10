export CUDA_VISIBLE_DEVICES=1
python test_rendering.py --sac --task-id 0 --simg --shm --seed 0
python test_rendering.py --sac --task-id 0 --simg --shm --seed 1
python test_rendering.py --sac --task-id 0 --simg --shm --seed 2
python test_rendering.py --sac --task-id 0 --simg --shm --seed 3

python test_rendering.py --sac --task-id 1 --simg --shm --seed 0
python test_rendering.py --sac --task-id 1 --simg --shm --seed 1
python test_rendering.py --sac --task-id 1 --simg --shm --seed 2
python test_rendering.py --sac --task-id 1 --simg --shm --seed 3

python test_rendering.py --sac --task-id 2 --simg --shm --seed 0
python test_rendering.py --sac --task-id 2 --simg --shm --seed 1
python test_rendering.py --sac --task-id 2 --simg --shm --seed 2
python test_rendering.py --sac --task-id 2 --simg --shm --seed 3
