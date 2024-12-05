# python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --lr 0.001
# python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --lr 0.001 --demon
# python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.001
python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.001 --demon
python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.002 --demon
python run_so_abs0.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.003 --demon

python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 1 --line-search --lr 0.03 --demon
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 2 --line-search --lr 0.03 --demon
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 3 --line-search --lr 0.03 --demon
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 0 --line-search --lr 0.03 --demon --hm
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 1 --line-search --lr 0.03 --demon --hm
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 2 --line-search --lr 0.03 --demon --hm
python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 12 --task-id 3 --line-search --lr 0.03 --demon --hm

python run_so_abs2.py --ptcl_d 1e7 --cuda_GB 20 --task-id 0 --line-search --lr 0.03 --demon
python run_so_abs2.py --ptcl_d 2e7 --cuda_GB 20 --task-id 0 --line-search --lr 0.03 --demon --hm

#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-clip
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-clip --line-search
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-clip --hm
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-clip --hm --line-search
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-clip --line-search --init-guess
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-clip --hm --line-search --init-guess
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-norm --line-search --init-guess
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-norm --hm --line-search --init-guess
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-dy-scale --line-search --init-guess
#python run_si.py --ptcl_d 5e6 --cuda_GB 12 --res 40 --sand --grad-dy-scale --hm --line-search --init-guess
