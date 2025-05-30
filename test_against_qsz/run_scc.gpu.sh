#!/bin/bash

nx=16
ny=4

declare -a nup_arr=(
    #[0]=14
    [1]=28
)

declare -a U_arr=(
    #[0]=6.0
    [1]=8.0
    #[2]=12.0
)

bc='open_x'
run_cpmc=1
set_e_estimate=1
init_trial='uhf'
Ueff=5
pin_type='afm'
v=0.25
proj_trs=0
approx_dm_pure=1
tol_delta_min=1e-4

dt=0.005
n_eql=500
n_blocks=400
nwalkers=1000
verbose=3

trs_tag=""
if [[ "$proj_trs" = "1" ]]; then
    trs_tag="trs"
fi

dm_tag="dm_mixed"
if [[ "$approx_dm_pure" = "1" ]]; then
    dm_tag="approx_dm_pure"
fi

outdir="/projects/bcdd/shufay/hubbard_tri/self_consistent_constraint/test_against_qsz/${trs_tag}/${dm_tag}/${nx}x${ny}/${bc}"

echo "nx, ny = $nx, $ny"

for i in "${!nup_arr[@]}"; do
    nup=${nup_arr[$i]}
    ndown=${nup}
    echo "---------------------------------------------------------------------"
    echo "nup, ndown = $nup, $ndown"
    echo

    for U in ${U_arr[@]}; do
        echo "---------------------------------------------------------------------"
        echo "U = $U"

        mkdir -p "${outdir}/U=${U}/pin=${pin_type}/${init_trial}_trial"

        jobname="afqmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"

        if [[ $run_cpmc == "1" ]]; then
            jobname="cpmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"
        fi

        sbatch -J ${jobname} \
               -o "${outdir}/U=${U}/pin=${pin_type}/${init_trial}_trial/${jobname}.%j.out" \
               -e "${outdir}/U=${U}/pin=${pin_type}/${init_trial}_trial/${jobname}.%j.err" \
               srun_scc.gpu.sh \
                ${U} ${nup} ${ndown} ${nx} ${ny} ${nwalkers} ${bc} \
                ${run_cpmc} ${set_e_estimate} ${dt} ${n_eql} ${n_blocks} \
                ${init_trial} ${Ueff} ${pin_type} ${v} ${proj_trs} ${approx_dm_pure} \
                ${tol_delta_min} ${verbose}
        echo
    done
done
