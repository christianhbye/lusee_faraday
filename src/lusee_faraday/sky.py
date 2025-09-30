from astropy.io import fits
import h5py
import healpy as hp
from lunarsky import LunarTopo, MoonLocation
import numpy as np
import scipy.constants as const

from .rotations import gal2topo

LUSEE_LOC = MoonLocation(lat=-23.813, lon=182.258, height=0)


def Tthermo2RJy(freq):
    """
    Conversion factor between thermodynamic temperature and antenna
    temperature in the Rayleigh-Jeans limit.

    Parameters
    ----------
    freq : float
        Frequency in MHz.

    Returns
    -------
    float
        Conversion factor in K_RJ / K_thermo.

    """
    Tcmb = 2.7255  # K
    f_hz = freq * 1e6
    x = const.h * f_hz / (const.k * Tcmb)
    f = x**2 * np.exp(x) / (np.exp(x) - 1) ** 2
    return f


def load_wmap(path, nside=128):
    nu_mhz = 23e3  # WMAP K-band is 23 GHz
    fconv = Tthermo2RJy(nu_mhz)
    with fits.open(path) as hdul:
        d = hdul["Stokes Maps"].data  # in mK, confirmed on LAMBDA website
        I_wmap = d["TEMPERATURE"] * 1e-3 * fconv
        Q_wmap = d["Q_POLARISATION"] * 1e-3 * fconv
        U_wmap = d["U_POLARISATION"] * 1e-3 * fconv
    I_wmap = hp.ud_grade(I_wmap, nside, order_in="NEST", order_out="RING")
    Q_wmap = hp.ud_grade(Q_wmap, nside, order_in="NEST", order_out="RING")
    U_wmap = hp.ud_grade(U_wmap, nside, order_in="NEST", order_out="RING")
    I_wmap = np.clip(I_wmap, 0, None)  # set negative pixels to zero
    maps = np.array([I_wmap, Q_wmap, U_wmap])
    return maps


def load_rm(path, nside=128):
    with h5py.File(path, "r") as f:
        RM_map = f["faraday_sky_mean"][:]
    RM_map = hp.ud_grade(RM_map, nside, order_in="RING", order_out="RING")
    return RM_map


class SkyModel:
    def __init__(
        self,
        I_map,
        Q_map,
        U_map,
        RM_map,
        freq=30,
        frame="galactic",
        faraday_rotated=False,
    ):
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
        faraday_rotated : bool

        """
        self.I_map = np.atleast_2d(I_map)
        self.Q_map = np.atleast_2d(Q_map)
        self.U_map = np.atleast_2d(U_map)
        self.RM_map = RM_map
        self.frame = frame
        self.freq = np.atleast_1d(freq)
        self.faraday_rotated = faraday_rotated

    def to_topocentric(self, time, loc="lusee"):
        """
        Convert maps to topocentric frame.

        Parameters
        ----------
        time : astropy.time.Time
            Observation time.
        loc : str
            Location. Supported: "lusee" = LuSEE Night landing site.

        Returns
        -------
        stokes_topo : ndarray
            Stokes maps in topocentric frame. Shape is (3, npix) or
            (3, nfreq, npix) depending on input shape, in the order
            (I, Q, U).

        """
        if loc == "lusee":
            loc = LUSEE_LOC
        else:
            raise ValueError(f"Location {loc} not supported.")
        if self.frame != "galactic":
            raise ValueError(
                "Current frame must be 'galactic' to convert to topocentric."
            )
        topo = LunarTopo(location=loc, obstime=time)
        Itopo, QTopo, Utopo = gal2topo(
            self.I_map, self.Q_map, self.U_map, topo_frame=topo
        )
        return np.array([Itopo, QTopo, Utopo])

    def apply_fd(self):
        """
        Apply Faraday rotation to Q and U maps.
        """
        if self.faraday_rotated:
            print("Faraday rotation already applied.")
            return
        lambda_sq = (3e8 / (self.freq * 1e6)) ** 2
        arg = 2 * self.RM_map[None, :] * lambda_sq[:, None]
        c = np.cos(arg)
        s = np.sin(arg)
        Q_rot = self.Q_map * c - self.U_map * s
        U_rot = self.Q_map * s + self.U_map * c
        self.Q_map = Q_rot
        self.U_map = U_rot
        self.faraday_rotated = True
