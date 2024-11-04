import numpy as np

def get_neel_sites(lattice):
    sites = lattice.sites
    sites_0 = []
    sites_120 = []
    sites_240 = []

    for site in sites:
        x, y = site
        i = lattice.get_site_num(site)
        if y % 2 == 0: x += 1
        if x % 3 == 0: sites_0.append(i)
        elif x % 3 == 1: sites_120.append(i)
        elif x % 3 == 2: sites_240.append(i)

    return np.array([sites_0, sites_120, sites_240])

def get_ghf_neel_guess(lattice):
    """
    Only for half-filling. Places 1 electron per site with Neel spin order.
    """
    n_sites = lattice.n_sites
    sites_0, sites_120, sites_240 = get_neel_sites(lattice)
    init_psi = np.zeros((2*n_sites, 2*n_sites))
    
    # Spin rotation matrices.
    theta = np.pi / 3. # 120/2 deg
    rot_120 = np.array([[np.cos(theta),  -np.sin(theta)],
                        [-np.sin(theta), -np.cos(theta)]])
    rot_240 = np.array([[np.cos(theta),  np.sin(theta)],
                        [np.sin(theta), -np.cos(theta)]])

    U_120 = np.kron(rot_120, np.eye(n_sites))
    U_240 = np.kron(rot_240, np.eye(n_sites))

    for i in sites_0:
        init_psi[i, i] = 1

    for i in sites_120:
        init_psi[i, i] = 1
        init_psi[:, i] = U_120 @ init_psi[:, i]

    for i in sites_240:
        init_psi[i, i] = 1
        init_psi[:, i] = U_240 @ init_psi[:, i]

    return init_psi

def get_afm_stripe_guess(lattice):
    """
    Only for half-filling. Places 1 electron per site with alternating spins
    between stripes.
    """
    n_sites = lattice.n_sites
    init_psi = np.zeros((2*n_sites, 2*n_sites))

    for i in range(n_sites):
        x = i // lattice.l_y
        y = i % lattice.l_y

        if x % 2 == 0: init_psi[i, i] = 1 # Spin up.
        else: init_psi[n_sites+i, i] = 1 # Spin down.

    return init_psi
