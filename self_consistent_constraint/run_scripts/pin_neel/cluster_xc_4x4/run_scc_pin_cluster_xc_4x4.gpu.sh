#!/bin/bash

nx=4
ny=4

declare -a nup_arr=(
    [8]=8
)

declare -a U_arr=(
    #[0]=4.0
    #[0]=6.0
    #[1]=8.0
    [2]=12.0
)

bc="xc"
run_cpmc=1
set_e_estimate=1
init_trial="ghf"
Ueff=5
v=0.25
approx_dm_pure=0
tol_delta_min=1e-4

dt=0.005
n_eql=200
n_blocks=400
nwalkers=800
verbose=3

dm_tag="dm_mixed"
if [[ "$approx_dm_pure" = "1" ]]; then
    dm_tag="approx_dm_pure"
fi

outdir="/projects/bcdd/shufay/hubbard_tri/self_consistent_constraint/pin_neel/cluster_${bc}_${nx}x${ny}/${dm_tag}"

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
        echo "v = $v"

        mkdir -p "${outdir}/U=${U}/${init_trial}_trial/v=${v}"

        jobname="afqmc_hubbard_cluster_${bc}_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}_v=${v}"

        if [[ $run_cpmc == "1" ]]; then
            jobname="cpmc_hubbard_cluster_${bc}_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}_v=${v}"
        fi

        sbatch -J ${jobname} \
               -o "${outdir}/U=${U}/${init_trial}_trial/v=${v}/${jobname}.%j.out" \
               -e "${outdir}/U=${U}/${init_trial}_trial/v=${v}/${jobname}.%j.err" \
               srun_scc_pin_cluster_xc_4x4.gpu.sh \
                ${U} ${nup} ${ndown} ${nwalkers} \
                ${run_cpmc} ${set_e_estimate} ${dt} ${n_eql} ${n_blocks} \
                ${init_trial} ${Ueff} ${v} ${approx_dm_pure} \
                ${tol_delta_min} ${verbose}
        echo
    done
done
