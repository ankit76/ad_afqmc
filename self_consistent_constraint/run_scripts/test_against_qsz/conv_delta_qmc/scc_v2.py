"""
v2
- Convergence criterion is based on the RMS error delta between the RDM1s of
successive iterations.
"""

import os
import sys
import h5py
import scipy
import numpy
from scipy.interpolate import interp1d
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo

from ad_afqmc import (
    driver,
    pyscf_interface,
    mpi_jax,
    linalg_utils,
    spin_utils,
    lattices,
    propagation,
    wavefunctions,
    hamiltonian,
)

# -----------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rcParams
import matplotlib.colors as mcolors
plt.rc('font',family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
plt.rc('axes',labelsize=18)
plt.rc('axes',titlesize=18)
plt.rc('legend',fontsize=14)
plt.rc('lines', linewidth=2)
plt.rc('savefig', dpi=300)

plt.rcParams['figure.autolayout'] =  True
plt.rcParams["font.family"] = "Serif"
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.figsize'] = [10, 8]
colors = list(mcolors.TABLEAU_COLORS.values())

# -----------------------------------------------------------------------------
# I/O functions.
def load_dm_qmc(iter, tmpdir='./'):
    npz = numpy.load(f'{tmpdir}rdm1_afqmc.npz')
    
    old_path = f'{tmpdir}rdm1_afqmc.npz'
    new_path = f'{tmpdir}rdm1_afqmc_{iter}.npz'
    try: os.rename(old_path, new_path)
    except FileNotFoundError: print(f"Error: '{old_path}' does not exist.")
    except FileExistsError: print(f"Error: '{new_path}' already exists.")
    except PermissionError: print("Error: Permission denied.")

    return npz['rdm1_avg'], npz['rdm1_noise']

def create_datasets(
        filename, dset_names, shape=(0,), maxshape=(None,), append=False, dtype='f8'):
    if append:
        with h5py.File(f'{filename}.h5', 'a') as f:
            for name in dset_names:
                if name not in f:
                    f.create_dataset(
                        name, shape=shape, maxshape=maxshape, chunks=(1,), dtype=dtype)

    else:
        with h5py.File(f'{filename}.h5', 'w') as f:
            name = dset_names[0]
            f.create_dataset(
                name, shape=shape, maxshape=maxshape, chunks=(1,), dtype=dtype)

        with h5py.File(f'{filename}.h5', 'a') as f:
            for name in dset_names[1:]:
                f.create_dataset(
                    name, shape=shape, maxshape=maxshape, chunks=(1,), dtype=dtype)
    
def push(filename, data, name):
    with h5py.File(f'{filename}.h5', 'a') as f:
        dset = f[name]
        old = dset.shape[0]
        new = old + 1
        dset.resize(new, axis=0)
        dset[old:new] = data

# -----------------------------------------------------------------------------
# Calculation helper functions.
def build_mol(n_elec, verbose=0):
    mol = gto.Mole()
    mol.nelectron = sum(n_elec)
    mol.incore_anyway = True
    mol.spin = abs(n_elec[0] - n_elec[1])
    mol.verbose = verbose
    mol.build()
    return mol

def get_integrals(U, v, lattice, pin_type='fm'):
    n_sites = lattice.n_sites

    # Integrals in the Hilbert space of the site basis.
    integrals = {}
    integrals["u"] = U
    integrals["h0"] = 0.0
    
    # Nearest-neighbour hopping.
    h1 = -1.0 * lattice.create_adjacency_matrix()
    
    # Pinning field.
    v1a = numpy.zeros(n_sites)
    v1b = numpy.zeros(n_sites)
    
    # Assuming the y-direction is the short one with PBC.
    for iy in range(lattice.l_y):
        val = (-1)**iy * v
        site_num_left = iy * lattice.l_x + 0
        site_num_right = iy * lattice.l_x + (lattice.l_x - 1)
        v1a[site_num_left] = val
        v1b[site_num_left] = -val

        if pin_type == 'fm':
            v1a[site_num_right] = val
            v1b[site_num_right] = -val

        elif pin_type == 'afm':
            v1a[site_num_right] = -val
            v1b[site_num_right] = val

    integrals["h1"] = numpy.array([h1 + numpy.diag(v1a), h1 + numpy.diag(v1b)])

    h2 = numpy.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites): h2[i, i, i, i] = U
    integrals["h2"] = ao2mo.restore(8, h2, n_sites)
    return integrals

def get_e_estimate(mol, lattice, integrals):
    n_sites = lattice.n_sites
    dm_init = numpy.zeros((2, n_sites, n_sites))

    for iy in range(lattice.l_y):
        for ix in range(lattice.l_x):
            site_num = iy * lattice.l_x + ix
            if (iy + ix) % 2 == 0: dm_init[0, site_num, site_num] = 1.0
            else: dm_init[1, site_num, site_num] = 1.0

    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: scipy.linalg.block_diag(*integrals["h1"])
    gmf.get_ovlp = lambda *args: numpy.eye(2 * n_sites)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    dm_init = scipy.linalg.block_diag(dm_init[0], dm_init[1])
    gmf.kernel(dm_init)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    return gmf.e_tot

def plot_density(lattice, density, figname=None, save=False):
    coords = numpy.array(lattice.sites)
    x = coords[:, 1]
    y = coords[:, 0]

    density_tot = density[0] + density[1]
    density_hole = 1 - density_tot
    density_spin = density[0] - density[1]
    density_stag_spin = numpy.zeros_like(density_spin)

    for ix in range(lattice.l_x):
        for iy in range(lattice.l_y):
            site_num = iy * lattice.l_x + ix
            density_stag_spin[site_num] = (-1)**(ix+iy) * density_spin[site_num]

    dat = [*density, density_tot, density_hole, density_spin, density_stag_spin]

    fig, axs = plt.subplots(6, 1, figsize=(8, 10), sharex=True)

    vmin_tot = numpy.amin(dat[:-2])
    vmax_tot = numpy.amax(dat[:-2])
    vmin_spin = numpy.amin(dat[-2])
    vmax_spin = numpy.amax(dat[-2])

    titles = [
          r'$\langle n_{i \uparrow} \rangle$',
          r'$\langle n_{i \downarrow} \rangle$',
          r'$\langle n_\text{tot} \rangle$',
          r'$\langle h \rangle$',
          r'$\langle n_\text{spin} \rangle$',
          r'$\langle n_\text{stag. spin} \rangle$'
         ]

    for i, ax in enumerate(axs):
        z = dat[i]

        if i < 3:
            cmap = 'bone_r'
            im = ax.scatter(x, y, c=z, cmap=cmap, vmin=vmin_tot, vmax=vmax_tot)

        elif i == 3:
            cmap = 'bone_r'
            im = ax.scatter(x, y, c=z, cmap=cmap)

        else:
            cmap = 'coolwarm'
            im = ax.scatter(x, y, c=z, cmap=cmap, vmin=vmin_spin, vmax=vmax_spin)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        ax.grid()
        ax.set_title(titles[i])
        ax.set_aspect("equal")

    plt.tight_layout()
    
    if save:
        if figname is None: figname = 'density.png'
        plt.savefig(figname, format='png')

def plot_density_slice(lattice, density, iy=1, figname=None, save=False):
    coords = numpy.array(lattice.sites)

    density_tot = density[0] + density[1]
    density_hole = 1 - density_tot
    density_spin = density[0] - density[1]
    density_stag_spin = numpy.zeros_like(density_spin)

    for i in range(lattice.l_x):
        for j in range(lattice.l_y):
            site_num = j * lattice.l_x + i
            density_stag_spin[site_num] = (-1)**(i+j) * density_spin[site_num]

    # Get slice along iy.
    site_nums = [iy * lattice.l_x + ix for ix in range(lattice.l_x)]
    slice_iy = coords[site_nums]
    x = slice_iy[:, 1]
    numpy.testing.assert_allclose(slice_iy[:, 0], iy)

    density_a_iy = density[0][site_nums]
    density_b_iy = density[1][site_nums]
    density_tot_iy = density_tot[site_nums]
    density_hole_iy = density_hole[site_nums]
    density_spin_iy = density_spin[site_nums]
    density_stag_spin_iy = density_stag_spin[site_nums]

    dat = [
            density_a_iy, 
            density_b_iy, 
            density_tot_iy, 
            density_hole_iy, 
            density_spin_iy, 
            density_stag_spin_iy
        ]

    fig, axs = plt.subplots(5, 1, figsize=(6, 10), sharex=True)

    titles = [
          r'$\langle n_{i \uparrow} \rangle$',
          r'$\langle n_{i \downarrow} \rangle$',
          r'$\langle n_\text{tot} \rangle$',
          r'$\langle h \rangle$',
          r'$\langle n_\text{spin} \rangle$',
          r'$\langle n_\text{stag. spin} \rangle$'
         ]

    for i, ax in enumerate(axs):
        if i == 0:
            z1 = dat[i]
            z2 = dat[i+1]
            ax.plot(x, z1, marker='o', ls='--', color='r', label=f'{titles[i]}')
            ax.plot(x, z2, marker='o', ls='--', color='b', label=f'{titles[i+1]}')

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        else:
            z = dat[i+1]
            ax.plot(x, z, marker='o', ls='--')
            ax.set_title(titles[i+1])

        ax.grid()
    
    plt.tight_layout()
    
    if save:
        if figname is None: figname = f'density_slice_iy={iy}.png'
        plt.savefig(figname, format='png')

# -----------------------------------------------------------------------------
# Build trials.
def build_fe_single_occ_trial(mol, lattice, integrals, verbose=0):
    nup, ndown = mol.nelec
    n_sites = lattice.n_sites
    evals_h1a, evecs_h1a = numpy.linalg.eigh(integrals["h1"][0])
    evals_h1b, evecs_h1b = numpy.linalg.eigh(integrals["h1"][1])

    umf = scf.UHF(mol)
    umf.get_hcore = lambda *args: integrals["h1"]
    umf.get_ovlp = lambda *args: numpy.eye(n_sites)
    umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    umf.mo_coeff = numpy.array([evecs_h1a, evecs_h1b])
    umf.mo_coeff[1][:, ndown-1] = evecs_h1b[:, ndown].copy()
    umf.mo_coeff[1][:, ndown] = evecs_h1b[:, ndown-1].copy()
    mo_occa = numpy.zeros(n_sites, dtype=int)
    mo_occb = numpy.zeros(n_sites, dtype=int)
    mo_occa[:nup] = 1
    mo_occb[:ndown] = 1
    umf.mo_occ = [mo_occa, mo_occb]
    return umf

def build_fe_double_occ_trial(mol, lattice, integrals, verbose=0):
    nup, ndown = mol.nelec
    n_sites = lattice.n_sites
    evals_h1a, evecs_h1a = numpy.linalg.eigh(integrals["h1"][0])
    evals_h1b, evecs_h1b = numpy.linalg.eigh(integrals["h1"][1])

    umf = scf.UHF(mol)
    umf.get_hcore = lambda *args: integrals["h1"]
    umf.get_ovlp = lambda *args: numpy.eye(n_sites)
    umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    umf.mo_coeff = numpy.array([evecs_h1a, evecs_h1b])
    mo_occa = numpy.zeros(n_sites, dtype=int)
    mo_occb = numpy.zeros(n_sites, dtype=int)
    mo_occa[:nup] = 1
    mo_occb[:ndown] = 1
    umf.mo_occ = [mo_occa, mo_occb]
    return umf

def build_uhf_trial(mol, lattice, integrals, seed=1, verbose=0):
    n_sites = lattice.n_sites
    numpy.random.seed(seed)

    umf = scf.UHF(mol)
    umf.get_hcore = lambda *args: integrals["h1"]
    umf.get_ovlp = lambda *args: numpy.eye(n_sites)
    umf._eri = ao2mo.restore(8, integrals["h2"], n_sites)

    # dm_init = 1.0 * umf.init_guess_by_1e()
    # dm_init += 1.0 * numpy.random.randn(*dm_init.shape)

    dm_init = 0.0 * umf.init_guess_by_1e()
    for iy in range(lattice.l_y):
        for ix in range(lattice.l_x):
            site_num = iy * lattice.l_x + ix
            if (ix + iy) % 2 == 0: dm_init[0, site_num, site_num] = 1.0
            else: dm_init[1, site_num, site_num] = 1.0

    umf.kernel(dm_init)
    mo1 = umf.stability()[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    mo1 = umf.stability()[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    mo1 = umf.stability()[0]
    umf = umf.newton().run(mo1, umf.mo_occ)
    mo1 = umf.stability()
    return umf

def project_trs_trial(umf, wave_data):
    # Prep free electron trial with singly-occupied orbitals at the Fermi level 
    # and obeying time-reversal symmetry.
    n_sites = umf.mo_coeff[0].shape[0]
    n_elec = umf.nelec

    trial_1 = wavefunctions.uhf_cpmc(n_sites, n_elec)
    trial_2 = wavefunctions.uhf_cpmc(n_sites, n_elec)
    trial = wavefunctions.sum_state_cpmc(n_sites, n_elec, (trial_1, trial_2))

    wave_data_0 = wave_data.copy()
    wave_data_0["mo_coeff"] = [
        umf.mo_coeff[0][:, : n_elec[0]],
        umf.mo_coeff[1][:, : n_elec[1]],
    ]

    wave_data_1 = wave_data.copy()
    wave_data_1["mo_coeff"] = [
        umf.mo_coeff[1][:, : n_elec[1]],
        umf.mo_coeff[0][:, : n_elec[0]],
    ]
    
    # CI coefficients.
    wave_data["coeffs"] = jnp.array([1 / numpy.sqrt(2.), 1 / numpy.sqrt(2.)])
    wave_data["0"] = wave_data_0
    wave_data["1"] = wave_data_1

    return trial, wave_data

# -----------------------------------------------------------------------------
# QMC.
def run_qmc(comm, umf, integrals, options, run_cpmc=True, proj_trs=False, 
            e_estimate=None, tmpdir='./', verbose=False):
    n_sites = umf.mo_coeff[0].shape[0]
    n_elec = umf.nelec

    if comm.rank == 0:
        pyscf_interface.prep_afqmc(
            umf, basis_coeff=numpy.eye(n_sites), integrals=integrals, tmpdir=tmpdir)
    
    comm.Barrier()
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
        mpi_jax._prep_afqmc(options, tmpdir=tmpdir)
    )

    if run_cpmc:
        if verbose: print(f'\n# Using CPMC propagator...')
        prop = propagation.propagator_cpmc(
            dt=options["dt"],
            n_walkers=options["n_walkers"],
        )

        if proj_trs:
            prop = propagation.propagator_cpmc_slow(
                dt=options["dt"],
                n_walkers=options["n_walkers"],
            )
    
    wave_data["mo_coeff"] = [umf.mo_coeff[0, :, :n_elec[0]], 
                             umf.mo_coeff[1, :, :n_elec[1]]]
    #wave_data["rdm1"] = umf.make_rdm1()
    ham_data["u"] = integrals["u"] # Needed for CPMC.
    if e_estimate is not None: ham_data["e_estimate"] = jnp.float64(e_estimate)

    if proj_trs: 
        if verbose: print(f'\n# Applying TRS projection for UHF trials...')
        trial, wave_data = project_trs_trial(umf, wave_data)

    else: trial = wavefunctions.uhf_cpmc(n_sites, n_elec)

    e_qmc, err_qmc = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,
        tmpdir=tmpdir
    )

    return e_qmc, err_qmc

# -----------------------------------------------------------------------------
# Scanning Ueff.
def get_bounds(x, a, b, npoint, alpha=2, thresh=0.1):
    interval = b - a
    dist_a = abs(x - a) / interval
    dist_b = abs(x - b) / interval
    stepsize = interval / (npoint - 1)

    # If x is close to the boundary, we'll use the same stepsize as the previous
    # iteration.
    if (dist_a < thresh) or (dist_b < thresh): d = (npoint - 1) // 2 * stepsize
    else: d = alpha * stepsize
    xmin = max(1e-3, x-d)
    xmax = x+d
    return [xmin, xmax]

def scan_ueff(iter, mol, lattice, dm_qmc, U=None, Ueff_prev=None, 
              bounds_prev=None, npoint=10, alpha=2, thresh=0.1, interp=False, 
              pin_type='fm', v=0.25, verbose=False):
    n_sites = lattice.n_sites
    dm_avg_qmc, dm_noise_qmc = dm_qmc

    # Rough scan for the first iteration.
    if iter == 0: 
        assert (U is not None)
        bounds = [1, U+4]
        Ueffs = numpy.arange(*bounds)
        stepsize = 1

    else:
        assert (Ueff_prev is not None)

        # Note that we'll only have (npoint - 1) points. This is because we 
        # obtain the Ueffs interval centered at `Ueff_prev` by concatenating 2 
        # arrays:
        #   [ bounds[0], Uprev ] -> npoint//2 elements
        #   ( Uprev, bounds[1] ] -> npoint//2 - 1 elements.
        bounds = get_bounds(
                Ueff_prev, *bounds_prev, npoint - 1, alpha=alpha, thresh=thresh)
        Ueffs_l = numpy.linspace(bounds[0], Ueff_prev, npoint//2)
        Ueffs_r = numpy.linspace(Ueff_prev, bounds[1], npoint//2)[1:]
        Ueffs = numpy.hstack((Ueffs_l, Ueffs_r))
        stepsize = Ueffs[1] - Ueffs[0]

    umfs = []
    deltas = []
    err_deltas = []

    if verbose: print(f'\n# Scanning over {bounds} with stepsize {stepsize}...')

    for Ueff in Ueffs:
        if verbose: print(f'\n# Ueff = {Ueff}')
        integrals_mf = get_integrals(Ueff, v, lattice, pin_type=pin_type)

        umf = scf.UHF(mol)
        umf.get_hcore = lambda *args: integrals_mf["h1"]
        umf.get_ovlp = lambda *args: numpy.eye(n_sites)
        umf._eri = ao2mo.restore(8, integrals_mf["h2"], n_sites)
        
        umf.kernel(dm_avg_qmc)
        mo1 = umf.stability()[0]
        umf = umf.newton().run(mo1, umf.mo_occ)
        mo1 = umf.stability()[0]
        umf = umf.newton().run(mo1, umf.mo_occ)
        mo1 = umf.stability()[0]
        umf = umf.newton().run(mo1, umf.mo_occ)
        mo1 = umf.stability()

        dm = umf.make_rdm1()
        density = numpy.array([numpy.diag(dm_s) for dm_s in dm])
        density_qmc = numpy.array([numpy.diag(dm_s) for dm_s in dm_avg_qmc])

        # Metric for determining convergence.
        delta = numpy.sqrt(numpy.sum( (density - density_qmc)**2 )) / n_sites
        err_delta = dm_noise_qmc / (2.*n_sites)

        umfs.append(umf)
        deltas.append(delta)
        err_deltas.append(err_delta)

    deltas = numpy.array(deltas)
    err_deltas = numpy.array(err_deltas)
    cost_func = None
    if interp: cost_func = interp1d(Ueffs, deltas)
    return Ueffs, deltas, err_deltas, umfs, cost_func

def get_ueff_opt(Ueffs, deltas, err_deltas, umfs):
    imin = numpy.argmin(deltas)
    delta_min = numpy.amin(deltas)
    err_delta_min = err_deltas[imin]
    Ueff_opt = Ueffs[imin]
    umf_opt = umfs[imin]
    return Ueff_opt, umf_opt, delta_min, err_delta_min

# -----------------------------------------------------------------------------
# Self-consistent procedure.
def check_convergence_qmc(delta_qmc, tol_delta_qmc=1e-4):
    delta_qmc_conv = abs(delta_qmc) < tol_delta_qmc
    if delta_qmc_conv: return True
    else: return False

def check_convergence(dUeff, dE, ddelta_min, err_E, err_delta_min, 
                      tol_ueff=1e-3, tol_energy=1e-8, tol_delta_min=1e-4):
    abs_dE = abs(dE)
    abs_ddelta_min = abs(ddelta_min)

    Ueff_conv = abs(dUeff) < tol_ueff
    energy_conv = (abs_dE < tol_energy) or (err_E > abs_dE)
    #delta_min_conv = (abs_ddelta_min < tol_delta_min) or (err_delta_min > abs_ddelta_min)
    delta_min_conv = abs_ddelta_min < tol_delta_min
    
    #if Ueff_conv and energy_conv and delta_min_conv: return True
    #if energy_conv and delta_min_conv: return True
    if delta_min_conv: return True
    else: return False

def run_scc(comm, U, nx, ny, n_elec, options, Ueff=None, pin_type='fm', v=0.25, 
            bc='open_x', run_cpmc=True, set_e_estimate=False, 
            init_trial='fe_single_occ', proj_trs=False, npoint=10, max_iter=20, 
            tol_delta_qmc=1e-4, tol_delta_min=1e-4,
            approx_dm_pure=False, do_plot_density=False, save_dm_hf=True, 
            tmpdir='./', filename='scc.out', verbose=0):
    if bc == 'open_x': lattice = lattices.two_dimensional_grid(nx, ny, open_x=True)
    else: lattice = lattices.two_dimensional_grid(nx, ny)
    n_sites = lattice.n_sites
    filling = sum(n_elec) / (2*n_sites)
    density = sum(n_elec) / n_sites

    if verbose: 
        print(f'\n# Filling factor = {filling}')
        print(f'# Density = {density}')
        print(f'\n# Pinning field = {pin_type}')
        print(f'# Pinning field strength = {v}')
    
    mol = build_mol(n_elec, verbose=verbose)
    integrals = get_integrals(U, v, lattice, pin_type=pin_type)

    # GHF energy for `e_estimate`.
    e_estimate = None

    if set_e_estimate: 
        e_estimate = get_e_estimate(mol, lattice, integrals)
        if verbose: print(f'\n# Setting `e_estimate` = {e_estimate}')

    if init_trial == 'fe_double_occ':
        umf_opt = build_fe_double_occ_trial(mol, lattice, integrals)
    
    elif init_trial == 'fe_single_occ':
        umf_opt = build_fe_single_occ_trial(mol, lattice, integrals)
    
    elif init_trial == 'uhf':
        umf_opt = build_uhf_trial(mol, lattice, integrals)

    elif init_trial == 'uhf_eff':
        if Ueff is None: Ueff = U/2.
        if verbose: print(f'# Ueff = {Ueff}')
        integrals_hf = get_integrals(Ueff, v, lattice, pin_type=pin_type)
        umf_opt = build_uhf_trial(mol, lattice, integrals_hf)
    
    # Initial values.
    conv = False
    iter = 0
    Ueff_prev = U
    bounds_prev = None
    e_qmc_prev = umf_opt.e_tot
    err_qmc_prev = 0.
    delta_min_prev = 0.
    err_delta_min_prev = 0.
    err_delta_qmc_prev = 0.
    dm_avg_qmc_prev = numpy.zeros((2, n_sites, n_sites))
    dm_noise_qmc_prev = 0.

    # For saving.
    scalar_names = [
        'e_qmc', 'err_qmc', 'Ueff_opt', 'delta_min', 'err_delta_min', 
        'delta_qmc', 'err_delta_qmc']
    arr_names = ['Ueffs', 'deltas', 'err_deltas']
    vlen_dtype = h5py.vlen_dtype(numpy.float64)

    if comm.rank == 0:
        h5_filename = f'{tmpdir}{filename}'
        create_datasets(
                h5_filename, scalar_names, shape=(0,), maxshape=(None,))
        create_datasets(
                h5_filename, arr_names, shape=(0,), maxshape=(None,), 
                dtype=vlen_dtype, append=True)

    while (not conv) and (iter < max_iter):
        # Run CPMC. The mixed estimator for rdm1 is saved by ad_afqmc.
        e_qmc, err_qmc = run_qmc(
                comm, umf_opt, integrals, options, run_cpmc=run_cpmc, 
                proj_trs=proj_trs, e_estimate=e_estimate, tmpdir=tmpdir, verbose=verbose)
        
        # Get rdm1.
        if comm.rank == 0: 
            dm_avg_qmc, dm_noise_qmc = load_dm_qmc(iter, tmpdir=tmpdir)

            if approx_dm_pure:
                if verbose: print(f'\n# Approximating the pure rdm1...')
                dm_hf = numpy.array(umf_opt.make_rdm1())
                dm_avg_qmc = 2 * dm_avg_qmc - dm_hf
                dm_noise_qmc = 2 * dm_noise_qmc

                # Save HF solution.
                numpy.savez(tmpdir + f'/rdm1_uhf_{iter}.npz', rdm1_uhf=dm_hf)
            
            density_qmc = numpy.array([numpy.diag(dm_s) for dm_s in dm_avg_qmc])
            density_qmc_prev = numpy.array([numpy.diag(dm_s) for dm_s in dm_avg_qmc_prev])

            # Convergence metric.
            delta_qmc = numpy.sqrt(numpy.sum( (density_qmc - density_qmc_prev)**2 )) / n_sites
            if iter == 0: err_delta_qmc = dm_noise_qmc / (2*n_sites)
            else: err_delta_qmc = numpy.sqrt(dm_noise_qmc**2 + dm_noise_qmc_prev**2) / (4*n_sites)

        else: 
            dm_avg_qmc, dm_noise_qmc = None, None
            delta_qmc, err_delta_qmc = None, None

        dm_avg_qmc = comm.bcast(dm_avg_qmc, root=0)
        dm_noise_qmc = comm.bcast(dm_noise_qmc, root=0)
        delta_qmc = comm.bcast(delta_qmc, root=0)
        err_delta_qmc = comm.bcast(err_delta_qmc, root=0)
        dm_qmc = (dm_avg_qmc, dm_noise_qmc)
        
        if do_plot_density and comm.rank == 0:
            iy = 1
            plot_density(
                lattice, density_qmc, f'{tmpdir}cpmc_density_iter={iter}.png', 
                save=True)
            plot_density_slice(
                lattice, density_qmc, iy=iy, 
                figname=f'{tmpdir}cpmc_density_slice_iy={iy}_iter={iter}.png', 
                save=True)
        
        # Find the optimal Ueff that minimizes the distance between the UHF and 
        # QMC site densities.
        Ueffs, deltas, err_deltas, umfs, cost_func = scan_ueff(
            iter, mol, lattice, dm_qmc, U=U, Ueff_prev=Ueff_prev, 
            bounds_prev=bounds_prev, npoint=npoint, pin_type=pin_type,
            v=v, verbose=verbose)

        # Update `Ueff_opt` and store the associated `umf_opt`.
        Ueff_opt, umf_opt, delta_min, err_delta_min = get_ueff_opt(Ueffs, deltas, err_deltas, umfs)

        # Check convergence.
        dUeff = Ueff_opt - Ueff_prev
        dE = e_qmc - e_qmc_prev
        ddelta_min = delta_min - delta_min_prev
        #err_dE = numpy.sqrt(err_qmc**2 + err_qmc_prev**2)
        #conv = check_convergence(
        #        dUeff, dE, ddelta_min, err_qmc_prev, err_delta_min_prev,
        #        tol_delta_min=tol_delta_min)

        conv = check_convergence_qmc(delta_qmc, tol_delta_qmc=tol_delta_qmc)
        
        if verbose:
            print(f'\n# Iter {iter}: Ueff_opt = {Ueff_opt}')
            print(f'# dUeff = {dUeff}, dE = {dE:.5f}, ddelta_min = {ddelta_min:.5e}, delta_qmc = {delta_qmc:.5e}')
            print(f'# err_qmc_prev = {err_qmc_prev:.3e}, err_delta_min_prev: {err_delta_min_prev:.3e}, err_delta_qmc_prev: {err_delta_qmc_prev:.3e}')
            if conv: print(f'\n# Self-consistency converged')

        # Write to hdf5.
        if comm.rank == 0:
            scalars = [e_qmc, err_qmc, Ueff_opt, delta_min, err_delta_min, 
                       delta_qmc, err_delta_qmc]
            arrays = [Ueffs, deltas, err_deltas]
            for i, scalar in enumerate(scalars): push(h5_filename, scalar, scalar_names[i])
            for i, arr in enumerate(arrays): push(h5_filename, arr, arr_names[i])
        
            if conv and save_dm_hf:
                numpy.savez(tmpdir + '/rdm1_uhf.npz', rdm1_uhf=umf_opt.make_rdm1())

        # Update values.
        bounds_prev = [Ueffs[0], Ueffs[-1]]
        Ueff_prev = Ueff_opt
        umf_prev = umf_opt
        e_qmc_prev = e_qmc
        err_qmc_prev = err_qmc
        delta_min_prev = delta_min
        err_delta_min_prev = err_delta_min
        err_delta_qmc_prev = err_delta_qmc
        dm_avg_qmc_prev = dm_avg_qmc
        dm_noise_qmc_prev = dm_noise_qmc
        iter += 1
    
    return conv
