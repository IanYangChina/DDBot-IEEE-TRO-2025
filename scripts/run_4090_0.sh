python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --init-search --lr 0.03 --task-id 0
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --init-search --lr 0.03 --task-id 1
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --init-search --lr 0.03 --task-id 2
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --init-search --lr 0.03 --hm --task-id 0
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --init-search --lr 0.03 --hm --task-id 1
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --init-search --lr 0.03 --hm --task-id 2

python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.03 --demon --hm --sand
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 1 --line-search --lr 0.03 --demon --hm --sand
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 2 --line-search --lr 0.03 --demon --hm --sand

python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.03 --sand --init-search
