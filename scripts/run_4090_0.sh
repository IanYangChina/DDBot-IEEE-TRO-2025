#python gradient_loss_assessment/assess_loss_si.py

#python run_so_abs2.py --ptcl_d 1e7 --cuda_GB 20 --task-id 0 --line-search --lr 0.03 --demon
#python run_so_abs2.py --ptcl_d 2e7 --cuda_GB 20 --task-id 0 --line-search --lr 0.03 --demon --hm

#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.005 --demon --sand
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.01 --demon --sand
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.02 --demon --sand
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.03 --demon --sand

python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 1 --line-search --lr 0.03 --demon --sand
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 2 --line-search --lr 0.03 --demon --sand
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.03 --demon --hm --sand
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 1 --line-search --lr 0.03 --demon --hm --sand
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 2 --line-search --lr 0.03 --demon --hm --sand

export CUDA_VISIBLE_DEVICES=1
python run_rl_abs2.py --t-cuda-id 0 --cuda_GB 12 --task-id 0 --demo --her -seed 0
#python run_rl_abs2.py --t-cuda-id 0 --cuda_GB 12 --task-id 0 --demo --her -seed 1
#python run_rl_abs2.py --t-cuda-id 0 --cuda_GB 12 --task-id 0 --demo --her -seed 2
#python run_rl_abs2.py --t-cuda-id 0 --cuda_GB 12 --task-id 0 --demo --her -seed 3
#python run_rl_abs2.py --t-cuda-id 0 --cuda_GB 12 --task-id 0 --demo --her -seed 4