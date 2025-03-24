#!/bin/bash
#
# Replace ACCOUNT with your account name before submitting.
#
#SBATCH --account=ccce           # Replace ACCOUNT with your group account name
#SBATCH -N 1                     # The number of nodes to request                      
#SBATCH --ntasks=20              # Total number of tasks                               
#SBATCH --ntasks-per-node=20                                                       
#SBATCH --cpus-per-task=1        # The number of cpu cores to use (up to 112 cores per server)
#SBATCH --mem=40G               # The memory the job will use per cpu core            
#SBATCH --time=0-12:00:00        # The time the job will take to run in D-HH:MM

# Email
#SBATCH --mail-user=su2254@columbia.edu
#SBATCH --mail-type=FAIL

# -----------------------------------------------------------------------------
# Prelims
# -----------------------------------------------------------------------------
export SCRATCHDIR="/burg/ccce/users/su2254"
export HOMEDIR="/burg/home/su2254"
export TMPDIR=${SCRATCHDIR}

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

# Echo
echo "SCRATCHDIR=${SCRATCHDIR}"
echo "HOMEDIR=${HOMEDIR}"
echo "TMPDIR=${TMPDIR}"
echo
echo "========================================================================"
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
verbose=${9}

mpirun -np $SLURM_NTASKS python -u run_uhf_trs.py \
                                    --U ${U} \
                                    --nup ${nup} \
                                    --ndown ${ndown} \
                                    --nx ${nx} \
                                    --ny ${ny} \
                                    --nwalkers ${nwalkers} \
                                    --bc ${bc} \
                                    --run_cpmc ${run_cpmc} \
                                    --verbose ${verbose}

# Mark the time job finishes
date
