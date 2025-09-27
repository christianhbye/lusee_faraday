import numpy as np

class SkyModel:
    def __init__(self, I_map, Q_map, U_map, RM_map, freq=30, frame="galactic"):
        self.I_map = I_map
        self.Q_map = Q_map
        self.U_map = U_map
        self.RM_map = RM_map
        self.frame = frame
        self.freq = freq

    def to_topocentric(self, time, loc="lusee"):
        """
        Convert maps to topocentric frame.

        Parameters
        ----------
        time : astropy.time.Time
            Observation time.
        loc : str
            Location. Supported: "lusee" = LuSEE Night landing site.
        """
        raise NotImplementedError

    def apply_fd(self):
        """
        Apply Faraday rotation to Q and U maps.
        """
        lambda_sq = (3e8 / (self.freq * 1e6))**2
        arg = 2 * self.RM_map[None, :] * lambda_sq[:, None]
        c = np.cos(arg)
        s = np.sin(arg)
        Q_rot = self.Q_map[None, :] * c - self.U_map[None, :] * s
        U_rot = self.Q_map[None, :] * s + self.U_map[None, :] * c
        self.Q_rot = Q_rot
        self.U_rot = U_rot
