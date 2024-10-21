python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-dy-scale

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --soft-contact --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --soft-contact --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --soft-contact --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-clip --soft-contact
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-norm --soft-contact
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-dy-scale --soft-contact

python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --toi-contact --grad-clip
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --toi-contact --grad-norm
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --toi-contact --grad-dy-scale
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-clip --toi-contact
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-norm --toi-contact
python assess_grad.py --ptcl_d 5e6 --cuda_GB 5 --substep --grad-dy-scale --toi-contact