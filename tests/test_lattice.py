import numpy as np

from ad_afqmc import config

config.setup_jax()

from ad_afqmc.lattices import triangular_grid

lattice_4x4 = triangular_grid(4, 4, open_x=False)
lattice_6x6 = triangular_grid(6, 6, open_x=False)

def test_get_boundary_pairs():
    ref_4x4 = np.array([[0, 3], [4, 7], [8, 11], [12, 15],
                        [0, 12], [1, 13], [2, 14], [3, 15]])
    ref_6x6 = np.array([[0, 5], [6, 11], [12, 17], [18, 23], [24, 29], [30, 35],
                        [0, 30], [1, 31], [2, 32], [3, 33], [4, 34], [5, 35]])
    test_4x4 = lattice_4x4.get_boundary_pairs()
    test_6x6 = lattice_6x6.get_boundary_pairs()
    
    assert ref_4x4.shape == test_4x4.shape
    assert ref_6x6.shape == test_6x6.shape

    for pair in test_4x4:
        assert np.count_nonzero(np.all(np.isclose(pair, ref_4x4), axis=1)) == 1
    
    for pair in test_6x6:
        assert np.count_nonzero(np.all(np.isclose(pair, ref_6x6), axis=1)) == 1


if __name__ == "__main__":
    test_get_boundary_pairs()
