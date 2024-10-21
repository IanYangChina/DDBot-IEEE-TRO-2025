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

python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 0 --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 1 --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 2 --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 3 --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 4 --grad-dy-scale

python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 0 --soft-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 1 --soft-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 2 --soft-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 3 --soft-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 4 --soft-contact --grad-dy-scale

python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 0 --toi-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 1 --toi-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 2 --toi-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 3 --toi-contact --grad-dy-scale
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 4 --toi-contact --grad-dy-scale

python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 0 --grad-clip
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 1 --grad-clip
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 2 --grad-clip
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 3 --grad-clip
python run_si.py --ptcl_d 5e6 --cuda_GB 8 --seed 4 --grad-clip

#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 0 --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 1 --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 2 --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 3 --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 4 --lr 0.02
#
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 0 --demo --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 1 --demo --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 2 --demo --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 3 --demo --lr 0.02
#python run_so_abs2.py --ptcl_d 5e6 --cuda_GB 8 --task-id 0 --seed 4 --demo --lr 0.02