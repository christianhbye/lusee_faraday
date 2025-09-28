import numpy as np
from lunarsky import LunarTopo, MoonLocation

from .utils import gal2topo

LUSEE_LOC = MoonLocation(lat=-23.813, lon=182.258, height=0)


class SkyModel:
    def __init__(self, I_map, Q_map, U_map, RM_map, freq=30, frame="galactic"):
        """
        Initialize SkyModel. The Stokes maps I, Q, U must have shape
        (nfreq, npix) or (npix,). The RM map must have shape (npix,).

        Parameters
        ----------
        I_map : array_like
            Intensity map.
        Q_map : array_like
            Stokes Q map.
        U_map : array_like
            Stokes U map.
        RM_map : array_like
            Rotation Measure map.
        freq : float or array_like
            Frequency in MHz or array of frequencies.
        frame : str
            Coordinate frame of the maps.

        """
        self.I_map = np.atleast_2d(I_map)
        self.Q_map = np.atleast_2d(Q_map)
        self.U_map = np.atleast_2d(U_map)
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
        if loc == "lusee":
            location = LUSEE_LOC
        else:
            raise ValueError(f"Location {loc} not supported.")
        if self.frame != "galactic":
            raise ValueError(
                "Current frame must be 'galactic' to convert to topocentric."
            )
        topo = LunarTopo(location=location, obstime=time)
        Itopo, QTopo, Utopo = gal2topo(
            self.I_map, self.Q_map, self.U_map, topo, interp=False
        )
        self.I_map = Itopo
        self.Q_map = QTopo
        self.U_map = Utopo
        self.frame = f"topocentric_{loc}"

    def apply_fd(self):
        """
        Apply Faraday rotation to Q and U maps.
        """
        lambda_sq = (3e8 / (self.freq * 1e6)) ** 2
        arg = 2 * self.RM_map[None, :] * lambda_sq[:, None]
        c = np.cos(arg)
        s = np.sin(arg)
        Q_rot = self.Q_map * c - self.U_map * s
        U_rot = self.Q_map * s + self.U_map * c
        self.Q_map = Q_rot
        self.U_map = U_rot
