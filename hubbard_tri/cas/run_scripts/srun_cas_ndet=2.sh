#!/bin/bash
#
#SBATCH --partition=cpu
#SBATCH --account=bcdd-delta-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=40g
#SBATCH --time=1:00:00
#SBATCH --export=ALL

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
conda activate ad_afqmc
                                                                                   
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK                                        
                                                                                   
# Print the essential SLURM job parameters
echo "SLURM job details:"                                                          
scontrol show job $SLURM_JOB_ID
echo 

# Inputs.
U=${1}
nup=${2}
ndown=${3}
nup_cas=${4}
ndown_cas=${5}
nx=${6}
ny=${7}
ncas=${8}
nwalkers=${9}
bc=${10}
run_cpmc=${11}
det_tol=${12}
verbose=${13}

mpirun -np $SLURM_NTASKS python -u run_cas_ndet=2.py \
                                    --U ${U} \
                                    --nup ${nup} \
                                    --ndown ${ndown} \
                                    --nup_cas ${nup_cas} \
                                    --ndown_cas ${ndown_cas} \
                                    --nx ${nx} \
                                    --ny ${ny} \
                                    --ncas ${ncas} \
                                    --nwalkers ${nwalkers} \
                                    --bc ${bc} \
                                    --run_cpmc ${run_cpmc} \
                                    --det_tol ${det_tol} \
                                    --verbose ${verbose}

# Mark the time job finishes
date
