using ITensors, ITensorMPS

function triangular_lattice_xc(Nx::Int, Ny::Int)::Lattice
  N = Nx * Ny
  # Estimate the maximum possible number of bonds
  max_bonds = 3 * N  # East, South, and Diagonal bonds

  # Create a lattice to hold the bonds
  latt = Lattice(undef, max_bonds)
  b = 0  # Bond counter

  for x in 0:(Nx-1)
    for y in 0:(Ny-1)
      site = x * Ny + y + 1  # Convert to 1-indexed

      # East bond
      if x + 1 < Nx
        east = (x + 1) * Ny + y + 1
        latt[b += 1] = LatticeBond(site, east)
      end

      # South bond (with y-periodicity)
      south_y = mod(y + 1, Ny)
      south = x * Ny + south_y + 1
      latt[b += 1] = LatticeBond(site, south)

      # Diagonal bond (depends on row parity)
      if y % 2 == 0  # Even rows (counting from 0).
        # Southeast diagonal
        if x + 1 < Nx
          se_y = mod(y + 1, Ny)
          se = (x + 1) * Ny + se_y + 1
          latt[b += 1] = LatticeBond(site, se)
        end
      else  # Odd rows
        # Southwest diagonal
        if x - 1 >= 0
          sw_y = mod(y + 1, Ny)
          sw = (x - 1) * Ny + sw_y + 1
          latt[b += 1] = LatticeBond(site, sw)
        end
      end
    end
  end

  # Resize to actual number of bonds
  resize!(latt, b)
  return latt
end

function triangular_lattice_yc(Nx::Int, Ny::Int)::Lattice
  N = Nx * Ny
  # Estimate the maximum possible number of bonds
  max_bonds = 3 * N  # East, South, and Diagonal bonds

  # Create a lattice to hold the bonds
  latt = Lattice(undef, max_bonds)
  b = 0  # Bond counter

  for x in 0:(Nx-1)
    for y in 0:(Ny-1)
      site = x * Ny + y + 1  # Convert to 1-indexed

      # East bond
      if x + 1 < Nx
        east = (x + 1) * Ny + y + 1
        latt[b += 1] = LatticeBond(site, east)
      end

      # South bond (with y-periodicity)
      south_y = mod(y + 1, Ny)
      south = x * Ny + south_y + 1
      latt[b += 1] = LatticeBond(site, south)

      # Diagonal bond (depends on column parity)
      if x % 2 == 1  # Odd columns
        # Southeast diagonal
        if x + 1 < Nx
          se_y = mod(y + 1, Ny)
          se = (x + 1) * Ny + se_y + 1
          latt[b += 1] = LatticeBond(site, se)
        end
      else  # Even rows
        # Northeast diagonal
        if x + 1 < Nx
          ne_y = mod(y - 1, Ny)
          ne = (x + 1) * Ny + ne_y + 1
          latt[b += 1] = LatticeBond(site, ne)
        end
      end
    end
  end

  # Resize to actual number of bonds
  resize!(latt, b)
  return latt
end

function get_site_coordinates_xc(Nx::Int, Ny::Int)
    n_sites = Nx * Ny
    theta = pi / 3.
    L1 = [1., 0.]
    L2 = [cos(theta), sin(theta)]
    L3 = L2 - L1
    Ly = [L3, L2]
    coords = zeros(n_sites, 2)
    
    for x in 0:(Nx-1)
        for y in 0:(Ny-1)
            site = x * Ny + y + 1

            for i in 1:y
                coords[site, :] .+= Ly[((i-1) % 2) + 1]
            end

            coords[site, :] .+= x * L1
        end
    end

    return coords
end

function get_site_coordinates_yc(Nx::Int, Ny::Int)
    n_sites = Nx * Ny
    theta = pi / 3.
    L1 = [0., 1.]
    L2 = [sin(theta), cos(theta)]
    L3 = L2 - L1
    Lx = [L2, L3]
    coords = zeros(n_sites, 2)
    
    for x in 0:(Nx-1)
        for y in 0:(Ny-1)
            site = x * Ny + y + 1

            for i in 1:x
                coords[site, :] .+= Lx[((i-1) % 2) + 1]
            end

            coords[site, :] .+= y * L1
        end
    end

    return coords
end
