#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --grad-norm --res 40 --line-search --init-guess
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --grad-dy-scale --res 40 --line-search --init-guess
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --grad-norm --res 40 --line-search --init-guess --hm
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --grad-dy-scale --res 40 --line-search --init-guess --hm
#python run_si.py --ptcl_d 1e7 --cuda_GB 20 --grad-clip --res 40 --hm --line-search --init-guess
#python run_si.py --ptcl_d 2e7 --cuda_GB 20 --grad-clip --res 40 --hm --line-search --init-guess

python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.01 --demon
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.02 --demon
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --lr 0.01 --demon
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --lr 0.01
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.01
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.01 --zero-init

#python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --lr 0.001
#python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --lr 0.001 --demon
#python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.001
#python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.001 --demon
#python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.002 --demon
