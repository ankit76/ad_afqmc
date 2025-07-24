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

import ad_afqmc
print(ad_afqmc.__file__)

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
    
    # Rename file.
    old_path = f'{tmpdir}rdm1_afqmc.npz'
    new_path = f'{tmpdir}rdm1_afqmc_{iter}.npz'
    try: os.rename(old_path, new_path)
    except FileNotFoundError: print(f"Error: '{old_path}' does not exist.")
    except FileExistsError: print(f"Error: '{new_path}' already exists.")
    except PermissionError: print("Error: Permission denied.")
    return npz['rdm1_avg'], npz['rdm1_noise']

def create_datasets(
        filename, dset_names, shape=(0,), maxshape=(None,), append=False, dtype='f8'):
    """
    Create hdf5 datasets.
    """
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
    """
    Push data to hdf5 file after each iteration.
    """
    with h5py.File(f'{filename}.h5', 'a') as f:
        dset = f[name]
        old = dset.shape[0]
        new = old + 1
        dset.resize(new, axis=0)
        dset[old:new] = data

# -----------------------------------------------------------------------------
# Calculation helper functions.
def build_mol(n_elec, verbose=0):
    """
    Build PySCF `mol` object.
    """
    mol = gto.Mole()
    mol.nelectron = sum(n_elec)
    mol.incore_anyway = True
    mol.spin = abs(n_elec[0] - n_elec[1])
    mol.verbose = verbose
    mol.build()
    return mol

def get_integrals(U, v, lattice, pin_sites):
    """
    Build integrals in the site basis for CPMC.
    """
    n_sites = lattice.n_sites
    sites_1, sites_2, sites_3 = pin_sites

    # Integrals.
    integrals = {}
    integrals["u"] = U
    integrals["h0"] = 0.0
    adjacency = lattice.create_adjacency_matrix()

    # Nearest-neighbour hopping.
    h1 = -1.0 * adjacency
    h1 = scipy.linalg.block_diag(h1, h1)

    # Pinning field.
    v *= -1.

    # A sites.
    vaa_1 = numpy.zeros((n_sites, n_sites))
    vbb_1 = numpy.zeros((n_sites, n_sites))

    for idx in sites_1:
        cz = v
        vaa_1[idx, idx] = cz
        vbb_1[idx, idx] = -cz

    v1 = scipy.linalg.block_diag(vaa_1, vbb_1)

    # B sites.
    vaa_2 = numpy.zeros((n_sites, n_sites))
    vab_2 = numpy.zeros((n_sites, n_sites))
    vba_2 = numpy.zeros((n_sites, n_sites))
    vbb_2 = numpy.zeros((n_sites, n_sites))

    theta = numpy.pi/6.
    for idx in sites_2:
        cx = -v * numpy.cos(theta)
        cz = -v * numpy.sin(theta)
        vaa_2[idx, idx] = cz
        vbb_2[idx, idx] = -cz
        vab_2[idx, idx] = cx
        vba_2[idx, idx] = cx

    v2 = numpy.block([[vaa_2, vab_2], [vba_2, vbb_2]])

    # C sites.
    vaa_3 = numpy.zeros((n_sites, n_sites))
    vab_3 = numpy.zeros((n_sites, n_sites))
    vba_3 = numpy.zeros((n_sites, n_sites))
    vbb_3 = numpy.zeros((n_sites, n_sites))

    theta = numpy.pi/6.
    for idx in sites_3:
        cx = v * numpy.cos(theta)
        cz = -v * numpy.sin(theta)
        vaa_3[idx, idx] = cz
        vbb_3[idx, idx] = -cz
        vab_3[idx, idx] = cx
        vba_3[idx, idx] = cx

    v3 = numpy.block([[vaa_3, vab_3], [vba_3, vbb_3]])

    integrals["h1"] = h1 + v1 + v2 + v3

    h2 = numpy.zeros((n_sites, n_sites, n_sites, n_sites))
    for i in range(n_sites): h2[i, i, i, i] = U
    integrals["h2"] = ao2mo.restore(8, h2, n_sites)
    return integrals

def get_e_estimate(mol, lattice, integrals, idx_up):
    """
    Set the initial energy for CPMC.
    """
    n_sites = lattice.n_sites
    dma_init = numpy.zeros(n_sites)
    dmb_init = numpy.zeros(n_sites)
    dma_init[idx_up] = 1
    dmb_init[numpy.where(dma_init == 0)[0]] = 0.5
    dm_init = scipy.linalg.block_diag(numpy.diag(dma_init), numpy.diag(dmb_init))

    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: integrals["h1"]
    gmf.get_ovlp = lambda *args: numpy.eye(2 * n_sites)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
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
    coords = numpy.array([lattice.get_site_coordinate(site) for site in lattice.sites])
    x = coords[:, 0]
    y = coords[:, 1]
    density_tot = density[0] + density[1]
    density_hole = 1 - density_tot
    density_spin = density[0] - density[1]
    density_stag_spin = numpy.zeros_like(density_spin)

    for ix in range(lattice.l_x):
        for iy in range(lattice.l_y):
            site_num = iy * lattice.l_y + ix
            density_stag_spin[site_num] = (-1)**(ix+iy) * density_spin[site_num]

    dat = [*density, density_tot, density_hole, density_spin, density_stag_spin]

    fig, axs = plt.subplots(6, 1, figsize=(6, 10), sharex=True)

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
    """
    Plot densities as a function of x, keeping y fixed.
    """
    coords = numpy.array([lattice.get_site_coordinate(site) for site in lattice.sites])

    density_tot = density[0] + density[1]
    density_hole = 1 - density_tot
    density_spin = density[0] - density[1]
    density_stag_spin = numpy.zeros_like(density_spin)
    
    # NOTE Use `i, j` rather than `ix, iy` since `iy` is a function argument!
    for i in range(lattice.l_x):
        for j in range(lattice.l_y):
            site_num = j * lattice.l_y + i
            density_stag_spin[site_num] = (-1)**(i+j) * density_spin[site_num]

    # Get slice along iy.
    site_nums = [iy * lattice.l_y + ix for ix in range(lattice.l_x)]
    slice_iy = coords[site_nums]
    x = slice_iy[:, 0]
    numpy.testing.assert_allclose(slice_iy[:, 1], iy * numpy.sin(numpy.pi/3.))

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
def build_fe_double_occ_trial(mol, lattice, integrals, verbose=0):
    nup, ndown = mol.nelec
    nocc = nup + ndown
    n_sites = lattice.n_sites
    evals_h1, evecs_h1 = scipy.linalg.eigh(integrals["h1"])

    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: integrals["h1"]
    gmf.get_ovlp = lambda *args: numpy.eye(2 * n_sites)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)
    gmf.mo_coeff = evecs_h1
    mo_occ = numpy.zeros(2 * n_sites, dtype=int)
    mo_occ[:nocc] = 1
    gmf.mo_occ = mo_occ
    return gmf

def build_ghf_trial(mol, lattice, integrals, idx_up, seed=1, verbose=0):
    n_sites = lattice.n_sites
    numpy.random.seed(seed)

    gmf = scf.GHF(mol)
    gmf.get_hcore = lambda *args: integrals["h1"]
    gmf.get_ovlp = lambda *args: numpy.eye(2 * n_sites)
    gmf._eri = ao2mo.restore(8, integrals["h2"], n_sites)

    dma_init = numpy.zeros(n_sites)
    dmb_init = numpy.zeros(n_sites)
    dma_init[idx_up] = 1
    dmb_init[numpy.where(dma_init == 0)[0]] = 0.5
    dm_init = scipy.linalg.block_diag(numpy.diag(dma_init), numpy.diag(dmb_init))

    #dm_init = numpy.random.random((2*n_sites, 2*n_sites))

    gmf.kernel(dm_init)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    gmf = gmf.newton().run(mo1, gmf.mo_occ)
    mo1 = gmf.stability(external=True)
    return gmf

# -----------------------------------------------------------------------------
# QMC.
def run_qmc(comm, gmf, integrals, options, run_cpmc=True, e_estimate=None, 
            tmpdir='./', verbose=False):
    n_sites = gmf.mo_coeff[0].shape[0] // 2
    n_elec = gmf.mol.nelec
    nocc = sum(n_elec)

    if comm.rank == 0:
        pyscf_interface.prep_afqmc_ghf(
            gmf, basis_coeff=numpy.eye(2*n_sites), integrals=integrals, tmpdir=tmpdir)
    
    comm.Barrier()
    ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI = (
        mpi_jax._prep_afqmc(options, tmpdir=tmpdir)
    )

    if run_cpmc:
        if verbose: print(f'\n# Using CPMC propagator...')
        prop = propagation.propagator_cpmc_generalized(
            dt=options["dt"],
            n_walkers=options["n_walkers"],
        )
    
    trial = wavefunctions.ghf_cpmc(n_sites, n_elec)
    wave_data["mo_coeff"] = gmf.mo_coeff[:, :nocc]
    wave_data["rdm1"] = wave_data["mo_coeff"] @ wave_data["mo_coeff"].T
    ham_data["u"] = integrals["u"] # Needed for CPMC.
    if e_estimate is not None: ham_data["e_estimate"] = jnp.float64(e_estimate)

    e_qmc, err_qmc = driver.afqmc(
        ham_data, ham, prop, trial, wave_data, sampler, observable, options, MPI,
        tmpdir=tmpdir
    )

    return e_qmc, err_qmc

# -----------------------------------------------------------------------------
# Scanning Ueff.
def get_bounds(x, a, b, npoint, alpha=2, thresh=0.1):
    """
    Get bounds around `x`, the optimal Ueff value determined in the previous
    iteration, within which to scan Ueff values. If `x` is close to the previous
    bounds `a, b`, the new bounds span an interval of the same width as the
    previous iteration. Otherwise, the new bounds span an interval of width
    2 * alpha * stepsize, where stepsize is defined from the previous bounds.
    """
    interval = b - a
    dist_a = abs(x - a) / interval
    dist_b = abs(x - b) / interval
    stepsize = interval / (npoint - 1)

    # If x is close to the boundary, we'll use the same interval as the previous
    # iteration.
    if (dist_a < thresh) or (dist_b < thresh): d = (npoint - 1) // 2 * stepsize
    else: d = alpha * stepsize
    xmin = max(1e-3, x-d)
    xmax = x+d
    return [xmin, xmax]

def scan_ueff(iter, mol, lattice, dm_qmc, pin_sites, U=None, Ueff_prev=None, 
              bounds_prev=None, npoint=10, alpha=3, thresh=0.1, interp=False, 
              v=0.25, verbose=False):
    """
    Scan Ueff values and determine the optimal Ueff that produces a GHF solution
    with the minimum RMS error (delta) in the rdm1 relative to the CPMC rdm1.
    """
    n_sites = lattice.n_sites
    dm_avg_qmc, dm_noise_qmc = dm_qmc

    # Rough scan for the first iteration.
    if iter == 1: 
        assert (U is not None)
        bounds = [1, U+4]
        Ueffs = numpy.arange(*bounds)
        stepsize = 1

    else:
        assert (Ueff_prev is not None)

        # Note that we'll only have (npoint - 1) points. This is because we 
        # obtain the Ueff interval centered at `Ueff_prev` by concatenating 2 
        # arrays:
        #   [ bounds[0], Uprev ] -> npoint//2 elements
        #   ( Uprev, bounds[1] ] -> npoint//2 - 1 elements.
        bounds = get_bounds(
                Ueff_prev, *bounds_prev, npoint - 1, alpha=alpha, thresh=thresh)
        Ueffs_l = numpy.linspace(bounds[0], Ueff_prev, npoint//2)
        Ueffs_r = numpy.linspace(Ueff_prev, bounds[1], npoint//2)[1:]
        Ueffs = numpy.hstack((Ueffs_l, Ueffs_r))
        stepsize = Ueffs[1] - Ueffs[0]

    # NOTE These arrays are arranged in the order of `Ueffs`.
    gmfs = []
    deltas = []
    err_deltas = []

    if verbose: print(f'\n# Scanning over {bounds} with stepsize {stepsize}...')

    for Ueff in Ueffs:
        if verbose: print(f'\n# Ueff = {Ueff}')
        integrals_mf = get_integrals(Ueff, v, lattice, pin_sites)

        gmf = scf.GHF(mol)
        gmf.get_hcore = lambda *args: integrals_mf["h1"]
        gmf.get_ovlp = lambda *args: numpy.eye(2 * n_sites)
        gmf._eri = ao2mo.restore(8, integrals_mf["h2"], n_sites)
        
        gmf.kernel(dm_avg_qmc)
        mo1 = gmf.stability(external=True)
        gmf = gmf.newton().run(mo1, gmf.mo_occ)
        mo1 = gmf.stability(external=True)
        gmf = gmf.newton().run(mo1, gmf.mo_occ)
        mo1 = gmf.stability(external=True)
        gmf = gmf.newton().run(mo1, gmf.mo_occ)
        mo1 = gmf.stability(external=True)

        dm = gmf.make_rdm1()
        dma = dm[:n_sites, :n_sites]
        dmb = dm[n_sites:, n_sites:]
        dma_avg_qmc = dm_avg_qmc[:n_sites, :n_sites]
        dmb_avg_qmc = dm_avg_qmc[n_sites:, n_sites:]
        density = numpy.array([numpy.diag(dma), numpy.diag(dmb)])
        density_qmc = numpy.array([numpy.diag(dma_avg_qmc), numpy.diag(dmb_avg_qmc)])

        # RMS error between the GHF and CPMC rdm1s.
        delta = numpy.sqrt(numpy.sum( (density - density_qmc)**2 )) / n_sites
        err_delta = dm_noise_qmc / (2.*n_sites)
        
        gmfs.append(gmf)
        deltas.append(delta)
        err_deltas.append(err_delta)

    deltas = numpy.array(deltas)
    err_deltas = numpy.array(err_deltas)
    cost_func = None
    if interp: cost_func = interp1d(Ueffs, deltas)
    return Ueffs, deltas, err_deltas, gmfs, cost_func

def get_ueff_opt(Ueffs, deltas, err_deltas, gmfs):
    """
    Get the optimal Ueff that produces a minimum `delta` value.
    """
    imin = numpy.argmin(deltas)
    delta_min = numpy.amin(deltas)
    err_delta_min = err_deltas[imin]
    Ueff_opt = Ueffs[imin]
    gmf_opt = gmfs[imin]
    return Ueff_opt, gmf_opt, delta_min, err_delta_min

# -----------------------------------------------------------------------------
# Self-consistent procedure.
def check_convergence(dUeff, dE, ddelta_min, err_E, err_delta_min, 
                      tol_ueff=1e-3, tol_energy=1e-8, tol_delta_min=1e-3):
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

def run_scc(
    comm, U, nx, ny, n_elec, pin_sites, idx_up, options, Ueff=None, v=0.5, 
    bc='xc', run_cpmc=True, set_e_estimate=False, init_trial='ghf', 
    npoint=10, max_iter=20, tol_delta_min=1e-4,
    approx_dm_pure=False, do_plot_density=False, save_dm_hf=True, 
    tmpdir='./', filename='scc.out', verbose=0):
    """
    Run the self-consistent procedure.
    """
    lattice = lattices.triangular_grid(nx, ny, boundary=bc)
    n_sites = lattice.n_sites
    filling = sum(n_elec) / (2*n_sites)
    density = sum(n_elec) / n_sites

    if verbose: 
        print(f'\n# Boundary condition: {bc}')
        print(f'\n# Filling factor = {filling}')
        print(f'# Density = {density}')
        print(f'\n# Pinning field strength = {v}')
    
    mol = build_mol(n_elec, verbose=verbose)
    integrals = get_integrals(U, v, lattice, pin_sites)

    # GHF energy for `e_estimate`.
    e_estimate = None

    if set_e_estimate: 
        e_estimate = get_e_estimate(mol, lattice, integrals, idx_up)
        if verbose: print(f'\n# Setting `e_estimate` = {e_estimate}')

    if init_trial == 'fe_double_occ':
        gmf_opt = build_fe_double_occ_trial(mol, lattice, integrals)
    
    elif init_trial == 'ghf':
        gmf_opt = build_ghf_trial(mol, lattice, integrals, idx_up)

    elif init_trial == 'ghf_eff':
        if Ueff is None: Ueff = U/2.
        if verbose: print(f'# Ueff = {Ueff}')
        integrals_mf = get_integrals(Ueff, v, lattice, pin_sites)
        gmf_opt = build_ghf_trial(mol, lattice, integrals_mf, idx_up)
    
    # Initial values.
    conv = False
    iter = 0
    Ueff_prev = U
    bounds_prev = None
    e_qmc_prev = gmf_opt.e_tot
    err_qmc_prev = 0.
    delta_min_prev = 0.
    err_delta_min_prev = 0.

    # For saving.
    scalar_names = ['e_qmc', 'err_qmc', 'Ueff_opt', 'delta_min', 'err_delta_min']
    arr_names = ['Ueffs', 'deltas', 'err_deltas']
    vlen_dtype = h5py.vlen_dtype(numpy.float64)

    if comm.rank == 0:
        h5_filename = f'{tmpdir}{filename}'
        create_datasets(
                h5_filename, scalar_names, shape=(0,), maxshape=(None,))
        create_datasets(
                h5_filename, arr_names, shape=(0,), maxshape=(None,), 
                dtype=vlen_dtype, append=True)
    
    # Self-consistency loop.
    while (not conv) and (iter < max_iter):
        # Run CPMC. The mixed estimator for rdm1 is saved by ad_afqmc.
        e_qmc, err_qmc = run_qmc(
            comm, gmf_opt, integrals, options, run_cpmc=run_cpmc, 
            e_estimate=e_estimate, tmpdir=tmpdir, verbose=verbose)
        
        # Load CPMC rdm1.
        if comm.rank == 0: 
            dm_avg_qmc, dm_noise_qmc = load_dm_qmc(iter, tmpdir=tmpdir)

            if approx_dm_pure:
                # Approximate the pure estimator using
                #   2 * mixed_rdm1 - ghf_rdm1
                if verbose: print(f'\n# Approximating the pure rdm1...')
                dm_hf = numpy.array(gmf_opt.make_rdm1())
                dm_avg_qmc = 2 * dm_avg_qmc - dm_hf
                dm_noise_qmc = 2 * dm_noise_qmc

                # Save HF solution.
                numpy.savez(tmpdir + f'/rdm1_ghf_{iter}.npz', rdm1_ghf=dm_hf)
            
            # Only available at rank 0.
            dma_avg_qmc = dm_avg_qmc[:n_sites, :n_sites]
            dmb_avg_qmc = dm_avg_qmc[n_sites:, n_sites:]
            density_qmc = numpy.array([numpy.diag(dma_avg_qmc), numpy.diag(dmb_avg_qmc)])

        else: dm_avg_qmc, dm_noise_qmc = None, None
        dm_avg_qmc = comm.bcast(dm_avg_qmc, root=0)
        dm_noise_qmc = comm.bcast(dm_noise_qmc, root=0)
        dm_qmc = (dm_avg_qmc, dm_noise_qmc)
        
        # Save densities.
        if do_plot_density and comm.rank == 0:
            iy = 1
            plot_density(
                lattice, density_qmc, f'{tmpdir}cpmc_density_iter={iter}.png', 
                save=True)
            plot_density_slice(
                lattice, density_qmc, iy=iy, 
                figname=f'{tmpdir}cpmc_density_slice_iy={iy}_iter={iter}.png', 
                save=True)
        
        # Find the optimal Ueff that minimizes the distance between the GHF and 
        # CPMC site densities.
        iter += 1 # Update iter before each Ueff scan.
        Ueffs, deltas, err_deltas, gmfs, cost_func = scan_ueff(
            iter, mol, lattice, dm_qmc, pin_sites, U=U, Ueff_prev=Ueff_prev, 
            bounds_prev=bounds_prev, npoint=npoint, v=v, verbose=verbose)

        # Update `Ueff_opt` and store the associated `gmf_opt`.
        Ueff_opt, gmf_opt, delta_min, err_delta_min = get_ueff_opt(Ueffs, deltas, err_deltas, gmfs)

        # Check convergence.
        dUeff = Ueff_opt - Ueff_prev
        dE = e_qmc - e_qmc_prev
        ddelta_min = delta_min - delta_min_prev
        #err_dE = numpy.sqrt(err_qmc**2 + err_qmc_prev**2)
        conv = check_convergence(
                dUeff, dE, ddelta_min, err_qmc_prev, err_delta_min_prev,
                tol_delta_min=tol_delta_min)
        
        if verbose:
            print(f'\n# Iter {iter}: Ueff_opt = {Ueff_opt}')
            print(f'# dUeff = {dUeff}, dE = {dE:.5f}, ddelta_min = {ddelta_min:.5e}')
            print(f'# err_qmc_prev = {err_qmc_prev:.3e}, err_delta_min_prev: {err_delta_min_prev:.3e}')
            if conv: print(f'\n# Self-consistency converged')

        # Write to hdf5.
        if comm.rank == 0:
            scalars = [e_qmc, err_qmc, Ueff_opt, delta_min, err_delta_min]
            arrays = [Ueffs, deltas, err_deltas]
            for i, scalar in enumerate(scalars): push(h5_filename, scalar, scalar_names[i])
            for i, arr in enumerate(arrays): push(h5_filename, arr, arr_names[i])
        
            if conv and save_dm_hf:
                numpy.savez(tmpdir + '/rdm1_ghf.npz', rdm1_ghf=gmf_opt.make_rdm1())

        # Update values.
        bounds_prev = [Ueffs[0], Ueffs[-1]]
        Ueff_prev = Ueff_opt
        gmf_prev = gmf_opt
        e_qmc_prev = e_qmc
        err_qmc_prev = err_qmc
        delta_min_prev = delta_min
        err_delta_min_prev = err_delta_min
    
    return conv
