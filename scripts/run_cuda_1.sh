python assess_loss_si.py --ptcl_d 5e6 --cuda_GB 8 --res 10
python assess_loss_si.py --ptcl_d 1e7 --cuda_GB 8 --res 10
python assess_loss_si.py --ptcl_d 5e6 --cuda_GB 8 --res 20
python assess_loss_si.py --ptcl_d 1e7 --cuda_GB 8 --res 20
python assess_loss_si.py --ptcl_d 5e6 --cuda_GB 8 --res 30
python assess_loss_si.py --ptcl_d 1e7 --cuda_GB 8 --res 30

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 10 --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 20 --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 30 --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 40 --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 50 --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 60 --grad-clip

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 10 --grad-clip --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 20 --grad-clip --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 30 --grad-clip --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 40 --grad-clip --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 50 --grad-clip --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 60 --grad-clip --substep
