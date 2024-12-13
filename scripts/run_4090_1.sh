export CUDA_VISIBLE_DEVICES=1
python test_rendering --sac --task-id 0 --simg --shm --seed 0
python test_rendering --sac --task-id 0 --simg --shm --seed 1
python test_rendering --sac --task-id 0 --simg --shm --seed 2
python test_rendering --sac --task-id 0 --simg --shm --seed 3

python test_rendering --sac --task-id 1 --simg --shm --seed 0
python test_rendering --sac --task-id 1 --simg --shm --seed 1
python test_rendering --sac --task-id 1 --simg --shm --seed 2
python test_rendering --sac --task-id 1 --simg --shm --seed 3

python test_rendering --sac --task-id 2 --simg --shm --seed 0
python test_rendering --sac --task-id 2 --simg --shm --seed 1
python test_rendering --sac --task-id 2 --simg --shm --seed 2
python test_rendering --sac --task-id 2 --simg --shm --seed 3

rm -rf ../../log-abs2-sac/d5e6-task-0-her-demo/data
rm -rf ../../log-abs2-sac/d5e6-task-1-her-demo/data
rm -rf ../../log-abs2-sac/d5e6-task-2-her-demo/data
