import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Custom modules.
from ad_afqmc import spin_utils

def check_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

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

def plot_ghf_spin(gmf, coords, adjacency, save=False, figname=None):
    n_sites = gmf.mo_coeff.shape[0] // 2
    dm_gmf = gmf.make_rdm1()
    dmaa_gmf = dm_gmf[:n_sites, :n_sites]
    dmbb_gmf = dm_gmf[n_sites:, n_sites:]
    dmab_gmf = dm_gmf[:n_sites, n_sites:]
    dmba_gmf = dm_gmf[n_sites:, :n_sites]

    sz_gmf = (dmaa_gmf - dmbb_gmf).diagonal() / 2
    charge_gmf = (dmaa_gmf + dmbb_gmf).diagonal()
    sx_gmf = (dmab_gmf + dmba_gmf).diagonal() / 2

    ao_ovlp = np.eye(n_sites)
    spin_utils.spin_collinearity_test(gmf.mo_coeff, ao_ovlp, verbose=1)

    # draw the triangular lattice with arrows showing the spin
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    arrow_size = 0.7

    for i in range(n_sites):
        coord = coords[i]
        x, y = coord

        # GHF.
        ax.plot(x, y, "ko")
        ax.arrow(
            x,
            y,
            arrow_size * sx_gmf.ravel()[i],
            arrow_size * sz_gmf.ravel()[i],
            head_width=0.1,
        )

        # draw lines to nearest neighbors
        nn = np.nonzero(adjacency[i])[0]

        for n in nn:
            site_n = coords[n]
            x_n, y_n = site_n
            ax.plot([x, x_n], [y, y_n], "k-", lw=0.1)

    ax.set_aspect("equal")

    if save:
        if figname is None: figname = 'ghf_spin.png'
        plt.savefig(figname, format='png')

