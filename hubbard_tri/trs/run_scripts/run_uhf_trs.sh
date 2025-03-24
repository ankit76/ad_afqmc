#!/bin/bash

nx=6
ny=6

declare -a nup_arr=(
    [8]=18
)

declare -a U_arr=(
    [0]=6.0 
    #[1]=8.0 
    #[2]=12.0
)

bc='open_x'
run_cpmc=1
nwalkers=20
verbose=3
outdir="/burg/ccce/users/su2254/ad_afqmc/hubbard_tri/trs/${nx}x${ny}/${bc}"

echo "nx, ny = $nx, $ny"

for i in "${!nup_arr[@]}"; do
    nup=${nup_arr[$i]}
    ndown=${nup}
    echo "---------------------------------------------------------------------"
    echo "nup, ndown = $nup, $ndown"
    echo

    for U in ${U_arr[@]}; do
        echo "U = $U"
        mkdir -p "${outdir}/U=${U}"
        jobname="afqmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"
        
        if [[ $run_cpmc == "1" ]]; then
            jobname="cpmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"
        fi

        sbatch -J ${jobname} \
               -o "${outdir}/${jobname}.%j.out" \
               -e "${outdir}/${jobname}.%j.err" \
               srun_uhf_trs.sh ${U} ${nup} ${ndown} ${nx} ${ny} ${nwalkers} ${bc} ${run_cpmc} ${verbose}
        echo
    done
done
