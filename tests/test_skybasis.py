import healpy as hp
import numpy as np

from lusee_faraday.skybasis import n_modes, spin2_basis


def test_n_modes_counts():
    # per l: E and B; m=0 (1 part) + m=1..l (2 parts each) = (1+2l) parts
    assert n_modes(2) == 2 * (1 + 4)  # l=2 only -> 10
    assert n_modes(3) == 2 * (1 + 4) + 2 * (1 + 6)  # +l=3 -> 24


def test_basis_shape_real_nonzero():
    nside = 8
    basis = spin2_basis(nside, lmax=2)
    assert len(basis) == n_modes(2)
    npix = hp.nside2npix(nside)
    for label, Q, U in basis:
        assert isinstance(label, str)
        assert Q.shape == (npix,) and U.shape == (npix,)
        assert np.isrealobj(Q) and np.isrealobj(U)
        assert np.any(Q != 0) or np.any(U != 0)
