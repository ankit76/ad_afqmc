import numpy as np

def get_neel_sites(lattice):
    # TODO: Only works for XC lattice.
    sites = lattice.sites
    neel_sites = {f'site_{i}': [] for i in range(3)}

    for site in sites:
        y, x = site
        i = lattice.get_site_num(site)
        if y % 2 == 0: x += 1
        if x % 3 == 0: neel_sites['site_0'].append(i)
        elif x % 3 == 1: neel_sites['site_1'].append(i)
        elif x % 3 == 2: neel_sites['site_2'].append(i)

    return neel_sites

def get_ghf_neel_guess(lattice):
    # TODO: 
    #   Only works for XC lattice.
    #   Only for half-filling. Places 1 electron per site with Neel spins.
    n_sites = lattice.n_sites
    neel_sites = get_neel_sites(lattice)
    init_psi = np.zeros((2*n_sites, n_sites))
    
    # Spin rotation matrices.
    theta = np.pi / 3. # 120/2 deg
    rot_120 = np.array([[np.cos(theta),  -np.sin(theta)],
                        [-np.sin(theta), -np.cos(theta)]])
    rot_240 = np.array([[np.cos(theta),  np.sin(theta)],
                        [np.sin(theta), -np.cos(theta)]])

    U_120 = np.kron(rot_120, np.eye(n_sites))
    U_240 = np.kron(rot_240, np.eye(n_sites))

    for i in neel_sites['site_0']:
        init_psi[i, i] = 1

    for i in neel_sites['site_1']:
        init_psi[i, i] = 1
        init_psi[:, i] = U_120 @ init_psi[:, i]

    for i in neel_sites['site_2']:
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
