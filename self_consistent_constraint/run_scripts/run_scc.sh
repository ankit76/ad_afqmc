#!/bin/bash

nx=4
ny=4

declare -a nup_arr=(
    [8]=8
)

declare -a U_arr=(
    [0]=4.0
    #[0]=6.0
    [1]=8.0
    [2]=12.0
)

bc='open_x'
run_cpmc=1
proj_trs=0
approx_dm_pure=1
nwalkers=100
verbose=3

trs_tag=""
if [[ "$proj_trs" = "1" ]]; then
    trs_tag="trs"
fi

dm_tag="dm_mixed"
if [[ "$approx_dm_pure" = "1" ]]; then
    dm_tag="approx_dm_pure"
fi

outdir="/burg/ccce/users/su2254/ad_afqmc/hubbard_tri/self_consistent_constraint/${trs_tag}/${dm_tag}/${nx}x${ny}/${bc}"

echo "nx, ny = $nx, $ny"

for i in "${!nup_arr[@]}"; do
    nup=${nup_arr[$i]}
    ndown=${nup}
    echo "---------------------------------------------------------------------"
    echo "nup, ndown = $nup, $ndown"
    echo

    for U in ${U_arr[@]}; do
        echo "--------------------------------------------------------------------------------"
        echo "U = $U"

        mkdir -p "${outdir}/U=${U}"

        jobname="afqmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"

        if [[ $run_cpmc == "1" ]]; then
            jobname="cpmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"
        fi

        sbatch -J ${jobname} \
               -o "${outdir}/U=${U}/${jobname}.%j.out" \
               -e "${outdir}/U=${U}/${jobname}.%j.err" \
               srun_scc.sh ${U} ${nup} ${ndown} ${nx} ${ny} ${nwalkers} ${bc} \
                           ${run_cpmc} ${proj_trs} ${approx_dm_pure} ${verbose}
        echo
    done
done
