# python compute_gradient_si.py --ptcl_d 5e6 --n_substep 10 --cuda_GB 20  # 1 hour on 4090
# python compute_gradient_si.py --ptcl_d 1e7 --n_substep 10 --cuda_GB 20  # 1.5 hour on 4090
python compute_gradient_si.py --ptcl_d 5e6 --n_substep 20 --cuda_GB 20
python compute_gradient_si.py --ptcl_d 1e7 --n_substep 20 --cuda_GB 20
python compute_gradient_si.py --ptcl_d 2e7 --n_substep 10 --cuda_GB 20
python compute_gradient_si.py --ptcl_d 2e7 --n_substep 20 --cuda_GB 20
