using ITensors, ITensorMPS
include("../lattices.jl")

U = parse(Float64, ARGS[1])
v = parse(Float64, ARGS[2])
nup = parse(Int, ARGS[3])
ndown = parse(Int, ARGS[4])
verbose = parse(Bool, ARGS[5])
nocc = nup + ndown

if verbose
    println("\nU = $U")
    println("v = $v")
    println("nelec = $nup, $ndown")
end

# Hubbard model on a triangular lattice.
let
    nx = 6
    ny = 6
    n_sites = nx * ny
    sites = siteinds("Electron", n_sites; conserve_nf=true)
    lattice = triangular_lattice_yc(nx, ny)

    # -------------------------------------------------------------------------
    # Input operator terms which define a Hamiltonian matrix, and convert
    # these terms to an MPO tensor network.
    os = OpSum()
    
    # Hopping.
    for b in lattice
        os += -1.0, "Cdagup", b.s1, "Cup", b.s2
        os += -1.0, "Cdagup", b.s2, "Cup", b.s1
        os += -1.0, "Cdagdn", b.s1, "Cdn", b.s2
        os += -1.0, "Cdagdn", b.s2, "Cdn", b.s1
    end

    sites_1 = [1, 4, 32, 35] # A sublattice.
    sites_2 = [2, 5, 33, 36] # B sublattice.
    sites_3 = [3, 6, 31, 34] # C sublattice.

    # -------------------------------------------------------------------------
    # Pinning fields.
    # A sublattice
    for j in sites_1
        os += -v * 2, "Sz", j
    end

    # B sublattice
    theta = pi/6
    for j in sites_2
        os += -v * (-cos(theta)) * 2, "Sx", j
        os += -v * (-sin(theta)) * 2, "Sz", j
    end

    # C sublattice
    for j in sites_3
        os += -v * cos(theta) * 2, "Sx", j
        os += -v * (-sin(theta)) * 2, "Sz", j
    end

    # -------------------------------------------------------------------------
    # Coulomb interaction.
    for j = 1:n_sites
        os += U, "Nup", j, "Ndn", j
    end

    H = MPO(os, sites)

    # -------------------------------------------------------------------------
    # Define the number operator.
    os = OpSum()

    for j = 1:n_sites
        os += "Ntot", j
    end

    num_op = MPO(os, sites)

    # -------------------------------------------------------------------------
    # Define the Sz operator.
    os = OpSum()

    for j = 1:n_sites
        os += "Sz", j
    end

    sz_op = MPO(os, sites)

    # -------------------------------------------------------------------------
    # Create an initial AFM matrix product state.
    psi0_af = [((div(i-1, ny) + mod(i-1, ny)) % 2 == 0) ? "Up" : "Dn" for i = 1:nocc]

    for i = 1:(n_sites - nocc)
        push!(psi0_af, "Emp")
    end

    # Create an initial random matrix product state
    psi0 = random_mps(sites, psi0_af; linkdims=10)
    # @show linkdims(psi0)
    num_obs = inner(psi0, num_op, psi0)
    sz_obs = inner(psi0, sz_op, psi0)
    println("\nInitial guess")
    println("<N> = ", num_obs)
    println("<Sz> = ", sz_obs)

    # Plan to do 20 passes or 'sweeps' of DMRG, setting maximum MPS internal 
    # dimensions for each sweep and maximum truncation cutoff used when adapting 
    # internal dimensions:
    nsweeps = 20

    # Gradually increas states kept.
    #maxdim = [10, 20, 100, 200, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500]
    maxdim = [10, 20, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000]
    noise = [1E-6, 1E-7, 1E-7, 1E-7, 1E-8, 1E-8, 1E-8, 1E-8, 0.0]
    cutoff = [1E-10] # desired truncation error

    # Run the DMRG algorithm, returning energy (dominant eigenvalue) and optimized MPS.
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, noise)

    # Compute the number of fermions in psi.
    num_obs = inner(psi, num_op, psi)
    sz_obs = inner(psi, sz_op, psi)
    println("<N> = ", num_obs)
    println("<Sz> = ", sz_obs)
    println("<E> = ", energy)
    @show linkdims(psi)
    return
end
