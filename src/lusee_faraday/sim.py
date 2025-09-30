from dataclasses import dataclass
import numpy as np

from .healpix import HealpixGrid
from . import Beam, SkyModel


@dataclass
class SimConfig:
    freqs: np.ndarray
    times: np.ndarray
    sky: SkyModel
    beam: Beam
    nside: int = 128
    site: str = "lusee"
    faraday: bool = True


def beam_sky_multiply(beam_weights, sky_stokes):
    """
    Multiply beam weights with sky Stokes parameters.

    Parameters
    ----------
    beam_weights : dict
        Dictonary contating beam weights for 'x', 'y', and 'xy'
        polarizations. Keys are expected to be the same as in the Beam
        class. Values are arrays of length npix.
    sky_stokes : np.ndarray
        Array of shape (3, nfreq, npix) containing the Stokes I, Q, U
        parameters for each frequency and pixel.

    Returns
    -------
    Rxx, Ryy, Rxy : tuple of np.ndarray
        3 arrays of len nfreq containing the resulting visibilities
        for Rxx, Ryy, and Rxy. Note that the pixel area dOmega is not
        taken into account here and should be multiplied later.

    """
    # shape is (3, npix)
    wx = np.array(
        [beam_weights["wI_x"], beam_weights["wQ_x"], beam_weights["wU_x"]]
    )
    wy = np.array(
        [beam_weights["wI_y"], beam_weights["wQ_y"], beam_weights["wU_y"]]
    )
    wxy = np.array(
        [beam_weights["wI_xy"], beam_weights["wQ_xy"], beam_weights["wU_xy"]]
    )

    Rxx = np.sum(wx[:, None, :] * sky_stokes, axis=(0, 2))
    Ryy = np.sum(wy[:, None, :] * sky_stokes, axis=(0, 2))
    Rxy = np.sum(wxy[:, None, :] * sky_stokes, axis=(0, 2))

    return Rxx, Ryy, Rxy


class Simulator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.beam = cfg.beam
        self.sky = cfg.sky
        self.grid = HealpixGrid(nside=cfg.nside, horizon=True)

    def simulate_step(self, I_map, Q_map, U_map):
        """
        Simulate the visibilities given the beam, sky, and grid.

        Parameters
        ----------
        I_map : np.ndarray
            Stokes I map in topocentric coordinates. Shape is (nfreqs, npix).
        Q_map : np.ndarray
            Stokes Q map in topocentric coordinates. Shape is (nfreqs, npix).
        U_map : np.ndarray
            Stokes U map in topocentric coordinates. Shape is (nfreqs, npix).

        Returns
        -------
        np.ndarray
            Simulated visibilities for Rxx, Ryy, and Rxy.
            The shape is (3, nfreqs).

        """
        w = self.beam.weights
        m = self.grid.mask.astype(int)  # npix

        sky_stokes = np.array([I_map, Q_map, U_map])  #(3, nfreqs, npix)
        sky_stokes = sky_stokes * m[None, None, :]  # horizon mask
        Rxx, Ryy, Rxy = beam_sky_multiply(w, sky_stokes)

        # compute normalization: beam above horizon
        # factors of dOmega cancel out
        norm = np.sum(w["wI_x"] * m) + np.sum(w["wI_y"] * m)
        vis = np.array([Rxx, Ryy, Rxy]) / norm

        return vis

    @staticmethod
    def compute_stokes(Rmat):
        """
        Compute the Stokes parameters from the visibility matrix.

        Parameters
        ----------
        Rmat : np.ndarray
           Shape is (ntimes, 3, nfreqs) containing Rxx, Ryy, Rxy.

        Returns
        -------
        tuple of np.ndarray
            Stokes parameters I, Q, U. Each has shape (ntimes, nfreqs).

        """
        Rxx = Rmat[:, 0, :]
        Ryy = Rmat[:, 1, :]
        Rxy = Rmat[:, 2, :]

        I = Rxx + Ryy
        Q = Rxx - Ryy
        U = 2 * np.real(Rxy)

        return I, Q, U

    def simulate(self, in_topo=False):
        """
        Simulate the visibilities over all times and frequencies.

        Parameters
        ----------
        in_topo : bool, optional
            If True, the sky model is assumed to be already in
            topocentric coordinates. Otherwise, it is converted for
            each time step. This only makes sense for one time step,
            as we are not evolving the topocentric sky model over time.

        Returns
        -------
        np.ndarray
            Simulated visibilities for all times and frequencies. The
            resulting array has shape (times.size, 3, freqs.size).

        """
        if self.cfg.faraday:
            self.sky.apply_fd()
        if in_topo:
            if self.cfg.times.size != 1:
                raise ValueError(
                    "in_topo=True only makes sense for one time step."
                )
            return self.simulate_step(
                self.cfg.sky.I_map,
                self.cfg.sky.Q_map,
                self.cfg.sky.U_map,
            )[None, ...]
        res = np.empty((self.cfg.times.size, 3, self.cfg.freqs.size))
        for i, t in enumerate(self.cfg.times):
            print("Simulating time step", i + 1, "of", self.cfg.times.size)
            I_t, Q_t, U_t = self.sky.to_topocentric(t, loc=self.cfg.site)
            R_t = self.simulate_step(I_t, Q_t, U_t)
            res[i] = R_t
        return res
