#!/bin/bash

nx=4
ny=4

declare -a nup_arr=(
    [8]=8
)

declare -a nup_cas_arr=(
    [8]=1
)

declare -a ncas_arr=(
    [8]=2
)

declare -a U_arr=(
    #[0]=4.0 
    #[1]=8.0 
    [2]=12.0
)

bc='open_x'
run_cpmc=1
nwalkers=20
det_tol=0.5
verbose=3
outdir="/burg/ccce/users/su2254/ad_afqmc/hubbard_tri/cas"

echo "nx, ny = $nx, $ny"

for i in "${!nup_arr[@]}"; do
    nup=${nup_arr[$i]}
    ndown=${nup}
    nup_cas=${nup_cas_arr[$i]}
    ndown_cas=${nup_cas}
    ncas=${ncas_arr[$i]}
    echo "---------------------------------------------------------------------"
    echo "nup, ndown = $nup, $ndown"
    echo "nup_cas, ndown_cas = $nup_cas, $ndown_cas"
    echo "ncas = $ncas"
    echo

    for U in ${U_arr[@]}; do
        echo "U = $U"
        mkdir -p "${outdir}/${nx}x${ny}/${bc}/cas_ndet=2/U=${U}"
        jobname="afqmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"
        
        if [[ $run_cpmc == "1" ]]; then
            jobname="cpmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}"
        fi

        sbatch -J ${jobname} \
               -o "U=${U}/${jobname}.%j.out" \
               -e "U=${U}/${jobname}.%j.err" \
               srun_cas_ndet=2.sh ${U} ${nup} ${ndown} ${nup_cas} ${ndown_cas} ${nx} ${ny} ${ncas} ${nwalkers} ${bc} ${run_cpmc} ${det_tol} ${verbose}
        echo
    done
done
