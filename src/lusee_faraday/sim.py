from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    freqs: np.ndarray
    times: np.ndarray
    sky: SkyModel
    beam: Beam
    nside: int = 128
    site: str = "lusee"
    faraday: bool = True


class Simulator(self, beam, sky, grid):
    def __init__(self, beam, sky, grid):
        self.beam = beam
        self.sky = sky
        self.grid = grid

    def simulate_step(self, I_map, Q_map, U_map):
        """
        Simulate the visibilities given the beam, sky, and grid.

        Parameters
        ----------
        faraday : bool, optional
            If True, use the Faraday-rotated Q and U maps. Default is True.

        Returns
        -------
        np.ndarray
            Simulated visibilities for Rxx, Ryy, and Rxy.

        """
        w = self.beam.weights
        m = self.grid.mask
        dOmega = self.grid.pix_area

        sky_stokes = np.array([I_map[m], Q_map[m], U_map[m]])

        Rxx = np.dot(w["x"], sky_stokes)
        Ryy = np.dot(w["y"], sky_stokes)
        Rxy = np.dot(w["xy"], sky_stokes)

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

    def simulate(self, cfg):
        # XXX figure shapes
        res = np.empty((cfg.freqs.size, cfg.times.size, 3, 3))
        for t in times:
            I_t, Q_t, U_t = sky.to_topocentric(
                t, loc=cfg.site, faraday=cfg.faraday
            )
            R_t = self.simulate_step(I_t, Q_t, U_t)
            res[:, t, :, :] = R_t
        return res
