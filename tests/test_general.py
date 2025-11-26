from pyscf import scf, gto, cc
from ad_afqmc import utils, afqmc, run_afqmc
import numpy as np
import os

tmpdir="tmp"
 
if not os.path.exists(tmpdir):
    os.makedirs(tmpdir)

options = {
"dt": 0.005,
"n_eql": 3,
"n_ene_blocks": 1,
"n_sr_blocks": 5,
"n_blocks": 20,
"n_prop_steps": 50,
"n_walkers": 5,
"seed": 8,
"trial": "",
"walker_type": "",
}

def check(obj, options, e, atol, mpi):
    # write_to_disk = True and run_afqmc
    # or
    # write_to_disk = False and run_afqmc_ph

    #utils.prep_afqmc(obj, tmpdir=tmpdir, chol_cut=1e-12, write_to_disk=True)
    pyscf_prep = utils.prep_afqmc(obj, tmpdir=tmpdir, chol_cut=1e-12, write_to_disk=False)

    if mpi:
        mpi_prefix = "mpirun "
        nproc = 2
    else:
        mpi_prefix = None
        nproc = None

    #ene, _ = run_afqmc.run_afqmc(
    #    options=options, mpi_prefix=mpi_prefix, nproc=nproc, tmpdir=tmpdir
    #)
    ene, _ = run_afqmc.run_afqmc_ph(
        pyscf_prep, options=options, mpi_prefix=mpi_prefix, nproc=nproc, tmpdir=tmpdir
    )

    assert np.isclose(ene, e, atol)
    return ene

# rhf
def check_rhf_restricted_w(mf, e, atol, mpi):
    options["trial"] = "rhf"
    options["walker_type"] = "restricted"
    return check(mf, options, e, atol, mpi)
    
def check_rhf_unrestricted_w(mf, e, atol, mpi):
    options["trial"] = "rhf"
    options["walker_type"] = "unrestricted"
    return check(mf, options, e, atol, mpi)

# uhf
def check_uhf_restricted_w(mf, e, atol, mpi):
    options["trial"] = "uhf"
    options["walker_type"] = "restricted"
    return check(mf, options, e, atol, mpi)

def check_uhf_unrestricted_w(mf, e, atol, mpi):
    options["trial"] = "uhf"
    options["walker_type"] = "unrestricted"
    return check(mf, options, e, atol, mpi)

# cisd
## rhf
def check_rcisd_restricted_w(mf, e, atol, mpi):
    mycc = cc.RCCSD(mf)
    mycc.kernel()
    options["trial"] = "cisd"
    options["walker_type"] = "restricted"
    return check(mycc, options, e, atol, mpi)
     
def check_rcisd_fc_restricted_w(mf, nfrozen, e, atol, mpi):
    mycc = cc.RCCSD(mf)
    mycc.frozen = nfrozen
    mycc.kernel()
    options["trial"] = "cisd"
    options["walker_type"] = "restricted"
    return check(mycc, options, e, atol, mpi)

## uhf
def check_ucisd_restricted_w(mf, e, atol, mpi):
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    options["trial"] = "ucisd"
    options["walker_type"] = "restricted"
    return check(mycc, options, e, atol, mpi)

def check_ucisd_unrestricted_w(mf, e, atol, mpi):
    mycc = cc.UCCSD(mf)
    mycc.kernel()
    options["trial"] = "ucisd"
    options["walker_type"] = "unrestricted"
    return check(mycc, options, e, atol, mpi)

def check_ucisd_fc_restricted_w(mf, nfrozen, e, atol, mpi):
    mycc = cc.UCCSD(mf)
    mycc.frozen = nfrozen
    mycc.kernel()
    options["trial"] = "ucisd"
    options["walker_type"] = "restricted"
    return check(mycc, options, e, atol, mpi)

def run(l_flag, l_fun):
    res = []
    for flag, (fun, args) in zip(l_flag, l_fun):
        print(flag)
        if flag:
            res.append(float(fun(*args)))

    return res

# H2O
def check_h2o(l_flag):
    mol = gto.M(
        atom = f'''
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        ''',
        basis = '6-31g',
        spin=0,
        verbose = 3)
    mf = scf.RHF(mol)
    mf.kernel()
    
    l_fun = [
        # No MPI
        (check_rhf_restricted_w,      (mf, -76.14187898749667, 1e-5, False)),
        (check_rhf_unrestricted_w,    (mf, -76.14187898749667, 1e-5, False)),
        (check_rcisd_restricted_w,    (mf, -76.12243596268871, 1e-5, False)),
        (check_rcisd_fc_restricted_w, (mf, 1, -76.1215017439916, 1e-5, False)),
        # MPI
        (check_rhf_restricted_w,      (mf, -76.13233451013845, 1e-5, True)),
        (check_rhf_unrestricted_w,    (mf, -76.13233451013845, 1e-5, True)),
        (check_rcisd_restricted_w,    (mf, -76.12240255883655, 1e-5, True)),
        (check_rcisd_fc_restricted_w, (mf, 1, -76.12149845139608, 1e-5, True))
    ]

    res = run(l_flag, l_fun)
    print("H2O:")
    print(res)

# NH2 
def check_nh2(l_flag):
    mol = gto.M(atom='''
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        ''',
        basis='6-31g',
        spin=1,
        verbose = 3)
    mf = scf.UHF(mol)
    mf.kernel()

    l_fun = [
        # No MPI
        (check_uhf_restricted_w, (mf, -55.65599052681822, 1e-5, False)),
        (check_uhf_unrestricted_w, (mf, -55.655991946870884, 1e-5, False)),
        (check_ucisd_restricted_w, (mf, -55.636250653696585, 1e-5, False)),
        (check_ucisd_unrestricted_w, (mf, -55.636251418009444, 1e-5, False)),
        (check_ucisd_fc_restricted_w, (mf, 1, -55.63515276697131, 1e-5, False)),
        ## MPI
        (check_uhf_restricted_w, (mf, -55.648220092580814, 1e-5, True)),
        (check_uhf_unrestricted_w, (mf, -55.648219500346904, 1e-5, True)),
        (check_ucisd_restricted_w, (mf, -55.636109651463116, 1e-5, True)),
        (check_ucisd_unrestricted_w, (mf, -55.63613901998812, 1e-5, True)),
        (check_ucisd_fc_restricted_w, (mf, 1, -55.63511844686504, 1e-5, True))
    ]

    res = run(l_flag, l_fun)
    print("NH2:")
    print(res)

# Tests
def test_handler(pytestconfig):
    l_flag_cs = [False for i in range(8)] # Closed Shell
    l_flag_os = [False for i in range(10)] # Open Shell

    # MPI
    if pytestconfig.getoption("mpi"):
        l_flag_cs[4:8] = [True]*4
        l_flag_os[5:10] = [True]*5
    # No MPI
    else:
        l_flag_cs[0:4] = [True]*4
        l_flag_os[0:5] = [True]*5

    # MPI + no MPI
    if pytestconfig.getoption("all"):
        l_flag_cs = [True for i in range(8)]
        l_flag_os = [True for i in range(10)]

    check_h2o(l_flag_cs)
    check_nh2(l_flag_os)
