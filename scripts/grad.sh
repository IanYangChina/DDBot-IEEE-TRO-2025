#python compute_gradient_si.py --ptcl_d 5e6 --n_substep 20 --cuda_GB 20  # 1.5 hours
#python compute_gradient_si.py --ptcl_d 1e7 --n_substep 20 --cuda_GB 20  # 2 hours 10 minntes
#python compute_gradient_si.py --ptcl_d 2e7 --n_substep 20 --cuda_GB 20  # 4 hours

#python compute_gradient_si.py --ptcl_d 5e6 --n_substep 40 --cuda_GB 8
#python compute_gradient_si.py --ptcl_d 1e7 --n_substep 40 --cuda_GB 8
#python compute_gradient_si.py --ptcl_d 2e7 --n_substep 40 --cuda_GB 8

python compute_gradient_si.py --ptcl_d 4e7 --n_substep 40 --cuda_GB 12
python compute_gradient_si.py --ptcl_d 4e7 --n_substep 20 --cuda_GB 12
