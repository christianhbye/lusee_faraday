import healpy as hp
import numpy as np


class HealpixGrid:
    def __init__(self, nside, horizon=True):
        self.nside = nside
        self.theta, self.phi = hp.pix2ang(nside, np.arange(self.npix))
        mask = np.ones(self.npix, dtype=bool)
        if horizon:
            mask[self.theta > np.pi / 2] = False
        self.mask = mask
    
    @property
    def npix(self):
        return 12 * self.nside ** 2

    @property
    def pix_area(self):
        return 4 * np.pi / self.npix

    def interp_hp(self, arr, theta, phi):
        """
        Interpolate a theta/phi grid to a healpix.

        Parameters
        ----------
        arr : array-like
            The input array to interpolate. Has shape (..., n_theta, n_phi).
        theta : array-like
            The theta values in radians.
        phi : array-like
            The phi values in radians.

        Returns
        -------
        hp_map : ndarray
            The interpolated healpix map. Has shape (..., npix).

        """
        raise NotImplementedError
