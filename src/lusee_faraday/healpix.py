import healpy as hp
import numpy as np
from scipy.interpolate import RectBivariateSpline


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
        return 12 * self.nside**2

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
        theta = np.array(theta)
        phi = np.array(phi)
        pix_theta, pix_phi = hp.pix2ang(self.nside, np.arange(self.npix))
        original_shape = arr.shape
        arr = np.array(arr, copy=True).reshape(-1, theta.size, phi.size)

        # remove poles before interpolation
        pole_values = np.full((len(arr), 2), None)
        if np.isclose(theta[0], 0):  # north pole
            theta = theta[1:]
            pole_values[:, 0] = arr[:, 0, :].mean(axis=-1)
            arr = arr[:, 1:, :]
        if np.isclose(theta[-1], np.pi):  # south pole
            theta = theta[:-1]
            pole_values[:, 1] = arr[:, -1, :].mean(axis=-1)
            arr = arr[:, :-1, :]

        # interpolate
        interp_data = np.empty((len(arr), self.npix))
        for i, a in enumerate(arr):
            spline = RectBivariateSpline(
                theta, phi, a, pole_values=pole_values[i]
            )
            interp_data[i] = spline(pix_theta, pix_phi, grid=False)
        return interp_data.reshape(original_shape[:-2] + (self.npix,))
