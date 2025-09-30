from dataclasses import dataclass
import numpy as np

from .healpix import HealpixGrid


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
        self.beam = cfg.beam
        self.sky = cfg.sky
        self.grid = HealpixGrid(nside=cfg.nside, horizon=True)

    def simulate_step(self):
        """
        Simulate the visibilities given the beam, sky, and grid.

        Returns
        -------
        np.ndarray
            Simulated visibilities for Rxx, Ryy, and Rxy.
            The shape is (3, nfreqs).

        """
        w = self.beam.weights
        dOmega = self.grid.pix_area
        m = self.grid.mask

        sky_stokes = np.array(
            [self.sky.I_map[m], self.sky.Q_map[m], self.sky.U_map[m]]
        )
        Rxx, Ryy, Rxy = beam_sky_multiply(w, sky_stokes)

        return dOmega * np.array([Rxx.real, Ryy.real, Rxy])

    @staticmethod
    def compute_stokes(Rmat):
        """
        Compute the Stokes parameters from the visibility matrix.

        Parameters
        ----------
        Rmat : np.ndarray

        Returns
        -------
        np.ndarray
            Stokes parameters I, Q, U.

        """
        Rxx, Ryy, Rxy = Rmat

        I = 1 / 2 * (Rxx + Ryy)
        Q = 1 / 2 * (Rxx - Ryy)
        U = 2 * np.real(Rxy)

        return np.array([I, Q, U])

    def simulate(self):
        """
        Simulate the visibilities over all times and frequencies.

        Returns
        -------
        np.ndarray
            Simulated visibilities for all times and frequencies. The
            resulting array has shape (times.size, 3, freqs.size).

        """
        if self.cfg.faraday:
            self.sky.apply_fd()
        res = np.empty((self.cfg.times.size, 3, self.cfg.freqs.size))
        for i, t in enumerate(times):
            I_t, Q_t, U_t = sky.to_topocentric(t, loc=self.cfg.site)
            R_t = self.simulate_step(I_t, Q_t, U_t)
            res[i] = R_t
        return res
