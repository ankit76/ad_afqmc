#!/bin/bash

nx=16
ny=4

declare -a nup_arr=(
    [8]=28
)

declare -a U_arr=(
    [0]=8.0
)

declare -a Ueff_arr=(
    [0]=0.001
    #[1]=2.7
    #[2]=3.11708
    #[3]=8.0

)

bc='open_x'
run_cpmc=1
set_e_estimate=1
pin_type='fm'
v=0.25

dt=0.005
n_eql=500
n_blocks=400
nwalkers=1000

verbose=3
outdir="/projects/bcdd/shufay/hubbard_tri/bcs/test_against_qsz/uhf/${nx}x${ny}/${bc}/dt=${dt}"

echo "nx, ny = $nx, $ny"

for i in "${!nup_arr[@]}"; do
    nup=${nup_arr[$i]}
    ndown=${nup}
    echo "---------------------------------------------------------------------"
    echo "nup, ndown = $nup, $ndown"
    echo

    for U in ${U_arr[@]}; do
        for Ueff in ${Ueff_arr[@]}; do
            echo "---------------------------------------------------------------------"
            echo "U = $U"
            echo "Ueff = $Ueff"

            if (( $(echo "$Ueff > $U" | bc -l) )); then
                break
            fi

            mkdir -p "${outdir}/U=${U}/Ueff=${Ueff}"

            jobname="afqmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}_Ueff=${Ueff}"

            if [[ $run_cpmc == "1" ]]; then
                jobname="cpmc_hubbard_${nx}x${ny}_nelec=(${nup},${ndown})_U=${U}_Ueff=${Ueff}"
            fi

            sbatch -J ${jobname} \
                   -o "${outdir}/U=${U}/Ueff=${Ueff}/pin=${pin_type}/${jobname}.%j.out" \
                   -e "${outdir}/U=${U}/Ueff=${Ueff}/pin=${pin_type}/${jobname}.%j.err" \
                   srun_ueff_uhf_trial_scan.gpu.sh ${U} ${Ueff} ${nup} ${ndown} ${nx} ${ny} ${nwalkers} ${bc} ${run_cpmc} ${set_e_estimate} ${pin_type} ${v} ${dt} ${n_eql} ${n_blocks} ${verbose}
            echo
        done
    done
done
