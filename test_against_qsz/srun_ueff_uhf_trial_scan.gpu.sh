#!/bin/bash
#
#SBATCH --partition=gpuA100x4
#SBATCH --account=bcdd-delta-gpu
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=4   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --time=06:00:00
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
#conda activate ad_afqmc3
                                                                                   
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
Ueff=${2}
nup=${3}
ndown=${4}
nx=${5}
ny=${6}
nwalkers=${7}
bc=${8}
run_cpmc=${9}
set_e_estimate=${10}
pin_type=${11}
v=${12}
dt=${13}
n_eql=${14}
n_blocks=${15}
verbose=${16}

export JAX_TRACEBACK_FILTERING=off

python -u run_ueff_uhf_trial.gpu.py \
                                    --U ${U} \
                                    --Ueff ${Ueff} \
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
                                    --pin_type ${pin_type} \
                                    --v ${v} \
                                    --verbose ${verbose}

# Mark the time job finishes
date
