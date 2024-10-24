python assess_loss_skill.py --ptcl_d 5e6 --cuda_GB 5

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 10 --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 20 --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 30 --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 40 --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 50 --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 60 --grad-dy-scale

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 10 --substep --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 20 --substep --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 30 --substep --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 40 --substep --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 50 --substep --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 60 --substep --grad-dy-scale

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 10 --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 20 --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 30 --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 40 --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 50 --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 60 --grad-norm

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 10 --grad-norm --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 20 --grad-norm --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 30 --grad-norm --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 40 --grad-norm --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 50 --grad-norm --substep
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --res 60 --grad-norm --substep
