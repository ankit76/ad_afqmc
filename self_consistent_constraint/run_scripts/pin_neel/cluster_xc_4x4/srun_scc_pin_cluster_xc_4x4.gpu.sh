#!/bin/bash
#
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcdd-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --time=12:00:00
#SBATCH --mem=10g
#SBATCH --export=ALL
#SBATCH --no-requeue

# Email
#SBATCH --mail-user=su2254@columbia.edu
#SBATCH --mail-type=FAIL

# -----------------------------------------------------------------------------
# Prelims
# -----------------------------------------------------------------------------
# Activate Anaconda work environment                                               
source $HOME/.bashrc                                                               
conda deactivate                                                                   
conda deactivate                                                                   
conda activate ad_afqmc_gpu
                                                                                   
export OMP_NUM_THREADS=1

export PYTHONPATH=/projects/bcdd/shufay/pyscf
export PYTHONPATH=/projects/bcdd/shufay/ad_afqmc:$PYTHONPATH

echo "PYTHONPATH: $PYTHONPATH"
                                                                                   
# Print the essential SLURM job parameters
echo "SLURM job details:"                                                          
scontrol show job $SLURM_JOB_ID
echo 

# Inputs.
U=${1}
nup=${2}
ndown=${3}
nwalkers=${4}
run_cpmc=${5}
set_e_estimate=${6}
dt=${7}
n_eql=${8}
n_blocks=${9}
init_trial=${10}
Ueff=${11}
v=${12}
approx_dm_pure=${13}
tol_delta_min=${14}
verbose=${15}

python -u run_scc_pin_cluster_xc_4x4.gpu.py \
            --U ${U} \
            --nup ${nup} \
            --ndown ${ndown} \
            --nwalkers ${nwalkers} \
            --dt ${dt} \
            --n_eql ${n_eql} \
            --n_blocks ${n_blocks} \
            --run_cpmc ${run_cpmc} \
            --set_e_estimate ${set_e_estimate} \
            --init_trial ${init_trial} \
            --Ueff ${Ueff} \
            --v ${v} \
            --approx_dm_pure ${approx_dm_pure} \
            --tol_delta_min ${tol_delta_min} \
            --verbose ${verbose}

# Mark the time job finishes
date
