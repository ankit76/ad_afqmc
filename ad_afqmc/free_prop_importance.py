from ad_afqmc import mpi_jax, wavefunctions
from jax import random, lax
import jax.numpy as jnp
import jax, time, copy
import numpy as np

def makeHFdata(norb, nelec_sp, ham_data, options):
    hf = np.zeros((norb, nelec_sp[0]))
    hf[:nelec_sp[0], :nelec_sp[0]] = np.eye(nelec_sp[0])
    hf = jnp.asarray(hf)

    HFtrial = wavefunctions.rhf(norb, nelec_sp, n_batch=options["n_batch"])
    HFwave_data = {}
    HFwave_data["mo_coeff"] = hf[:, : nelec_sp[0]]
    HFham_data = HFtrial._build_measurement_intermediates(ham_data, HFwave_data)
    #HFham_data = ham.build_measurement_intermediates(ham_data, wave_data)
    return HFtrial, HFwave_data, HFham_data

def run():
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, _ = (
        mpi_jax._prep_afqmc()
    )
    initHam_data = copy.deepcopy(ham_data)

    assert trial is not None
    init = time.time()
    e_afqmc, err_afqmc = 0.0, 0.0


    ham_data = ham.build_measurement_intermediates(ham_data, trial, wave_data)
    ham_data = ham.build_propagation_intermediates(
        ham_data, prop, trial, wave_data
    )

    ImpSample = not options["free_projection"]

    HFtrial, HFwave_data, HFham_data = makeHFdata(trial.norb, trial.nelec, initHam_data, options)

    print(sampler.n_sr_blocks, sampler.n_ene_blocks)

    N = 10  ##run the same calcualtion 30 times
    energy_samplesN, HFenergy_samplesN= [], []
    for I in range(N):

        prop_data = prop.init_prop_data(trial, wave_data, ham_data, None)
        prop_data["key"] = random.PRNGKey(options["seed"]+I)#int(init*10.))
        prop_data["pop_control_ene_shift"] = prop_data["e_estimate"]


        HFTotaldenominator, HFTotalnumerator = 0., 0.
        total_weight, total_numerator = 0., 0.
        block_samples_e = []
        HFblock_samples_e = []

        for blocks in range(sampler.n_blocks):

            if (blocks == options["n_eql"]):
                HFTotaldenominator, HFTotalnumerator = 0., 0.
                total_weight, total_numerator = 0., 0.

            for sr_blocks in range(sampler.n_sr_blocks):

                for  ene_blocks in range(sampler.n_ene_blocks):

                    prop_data["key"], subkey = random.split(prop_data["key"])
                    fields = random.normal(subkey, shape=(sampler.n_prop_steps, prop.n_walkers, ham_data["chol"].shape[0],),)
                    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)


                    if (ImpSample): 
                        _propogate_block = lambda x, y: (prop.propagate(trial, ham_data, x, y, wave_data), y)
                    else:
                        _propogate_block = lambda x, y: (prop.propagate_free(trial, ham_data, x, y, wave_data), y)
                        
                    prop_data, _ = lax.scan(_propogate_block, prop_data, fields)


                    prop_data, norm = prop._orthonormalize_walkers(prop_data)
                    norm = norm**2
                    #norm = 1.
                    prop_data["overlaps"] = trial.calc_overlap(prop_data["walkers"], wave_data)
                    energy_samples = trial.calc_energy(prop_data["walkers"], ham_data, wave_data)

                    if (ImpSample):
                        block_weight     = jnp.sum(prop_data["weights"])
                        block_numerator  = jnp.sum(energy_samples * prop_data["weights"])
                        total_weight    += block_weight
                        total_numerator += block_numerator
                        block_energy    =  block_numerator / block_weight
                        total_energy    =  total_numerator / total_weight

                        HFOvlp          =   HFtrial.calc_overlap(prop_data["walkers"], HFwave_data)
                        HFdenominator   =  jnp.sum(HFOvlp * prop_data["weights"]/prop_data["overlaps"])
                        HFnumerator     =  jnp.sum(HFtrial.calc_energy(prop_data["walkers"], HFham_data, HFwave_data) * HFOvlp * prop_data["weights"] / prop_data["overlaps"])
                        HFTotaldenominator += HFdenominator
                        HFTotalnumerator   += HFnumerator
                        HFblockEnergy   =   HFnumerator / HFdenominator
                        HFTotalEnergy   =   HFTotalnumerator / HFTotaldenominator
                    else:
                        prop_data["weights"] *= norm
                        #prop_data["overlaps"] *= prop_data["weights"]
                        block_weight = jnp.sum(prop_data["overlaps"]*prop_data["weights"])
                        block_energy = jnp.sum(energy_samples * prop_data["overlaps"]*prop_data["weights"])/block_weight

                        HFovlp = HFtrial.calc_overlap(prop_data["walkers"], HFwave_data)
                        HFenergy = HFtrial.calc_energy(prop_data["walkers"], HFham_data, HFwave_data)
                        HFblockEnergy = jnp.sum(HFenergy * HFovlp * prop_data["weights"])/jnp.sum(HFovlp * prop_data["weights"])

                        HFTotalEnergy = HFblockEnergy
                        total_energy = block_energy

                    # prop_data["pop_control_ene_shift"] = (
                    #     0.9 * prop_data["pop_control_ene_shift"] + 0.1 * block_energy.real
                    # )


                    #if (blocks >= sampler.n_ene_blocks-1):
                    if (blocks >= options["n_eql"]):
                        block_samples_e.append(block_energy)
                        HFblock_samples_e.append(HFblockEnergy)
                        print(f"{blocks:5d} {ene_blocks:5d} {block_energy:.9e}, {HFblockEnergy:.9e}")

                #print()
                prop_data = prop.stochastic_reconfiguration_local(prop_data)
                #prop_data["e_estimate"] = 0.9 * prop_data["e_estimate"] + 0.1 * block_energy
        print(total_energy.real, jnp.mean(jnp.asarray(block_samples_e)).real, jnp.std(jnp.asarray(block_samples_e))/(jnp.asarray(block_samples_e).shape[0]-1)**0.5)
        print(HFTotalEnergy.real, jnp.mean(jnp.asarray(HFblock_samples_e)).real, jnp.std(jnp.asarray(HFblock_samples_e))/(jnp.asarray(HFblock_samples_e).shape[0]-1)**0.5)
        energy_samplesN.append(total_energy.real)
        HFenergy_samplesN.append(HFTotalEnergy.real)

    HFenergy_samplesN = jnp.asarray(HFenergy_samplesN)
    print(f"{jnp.mean(HFenergy_samplesN):.9e}  {jnp.std(HFenergy_samplesN)/(N-1.)**0.5:.9e}")

    print("\n", HFenergy_samplesN)
if __name__ == "__main__":
    run()