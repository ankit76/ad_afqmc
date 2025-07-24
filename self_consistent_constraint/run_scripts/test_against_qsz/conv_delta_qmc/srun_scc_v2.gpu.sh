#!/bin/bash
#
#SBATCH --partition=gpuA100x4
##SBATCH --partition=gpuA100x4-preempt
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
nx=${4}
ny=${5}
nwalkers=${6}
bc=${7}
run_cpmc=${8}
set_e_estimate=${9}
dt=${10}
n_eql=${11}
n_blocks=${12}
init_trial=${13}
Ueff=${14}
pin_type=${15}
v=${16}
proj_trs=${17}
approx_dm_pure=${18}
tol_delta_qmc=${19}
verbose=${20}

python -u run_scc_v2.gpu.py \
            --U ${U} \
            --nup ${nup} \
            --ndown ${ndown} \
            --nx ${nx} \
            --ny ${ny} \
            --nwalkers ${nwalkers} \
            --dt ${dt} \
            --n_eql ${n_eql} \
            --n_blocks ${n_blocks} \
            --bc ${bc} \
            --run_cpmc ${run_cpmc} \
            --set_e_estimate ${set_e_estimate} \
            --init_trial ${init_trial} \
            --Ueff ${Ueff} \
            --pin_type ${pin_type} \
            --v ${v} \
            --proj_trs ${proj_trs} \
            --approx_dm_pure ${approx_dm_pure} \
            --tol_delta_qmc ${tol_delta_qmc} \
            --verbose ${verbose}

# Mark the time job finishes
date
