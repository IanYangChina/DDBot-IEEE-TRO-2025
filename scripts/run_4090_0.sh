#python gradient_loss_assessment/assess_loss_si.py

python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --lr 0.03 --task-id 2 --seed 4
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --lr 0.03 --hm --task-id 2

python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --demon --init-search --lr 0.03 --task-id 0
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --demon --init-search --lr 0.03 --task-id 1
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --demon --init-search --lr 0.03 --task-id 2
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --demon --init-search --lr 0.03 --hm --task-id 0
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --demon --init-search --lr 0.03 --hm --task-id 1
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --line-search --demon --init-search --lr 0.03 --hm --task-id 2
