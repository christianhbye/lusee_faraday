from astropy.io import fits
import numpy as np

from . import HealpixGrid


def rotate_beam(jones, angle):
    """
    Rotate the beam by a given angle, counter-clockwise about the z-axis.
    The function assumes that the beam is defined on regular 1-degree
    intervals in theta and phi and that `angle` is an integer.

    Parameters
    ----------
    jones : np.ndarray
        The Jones matrix. Shape is (2, ntheta, nphi); first axis is
        (Eth, Eph).

    angle : int
        Angle to rotate the beam by in degrees, counter-clockwise.

    Returns
    -------
    np.ndarray
        The rotated Jones matrix.

    """
    Eth, Eph = jones
    Eth_roll = np.roll(Eth, angle, axis=-1)
    Eph_roll = np.roll(Eph, angle, axis=-1)
    jones_rot = np.array([Eth_roll, Eph_roll])
    return jones_rot


class Beam:
    def __init__(self, jones_x, jones_y):
        """
        Initialize the Beam object.

        Parameters
        ----------
        jones_x : np.ndarray
            The Jones matrix for the X polarization.
        jones_y : np.ndarray
            The Jones matrix for the Y polarization.

        Notes
        -----
        The Jones matrices should have shape (2, npix), where the first axis
        corresponds to (Eth, Eph) and npix is the number of HEALPix pixels.

        """
        self.jones_x = jones_x
        self.jones_y = jones_y
        self.weights = None

    @classmethod
    def from_file(cls, path, frequency=30, nside=128, orientation="y"):
        """
        Load a LuSEE beam from a FITS file.

        Parameters
        ----------
        path : str
            Path to the FITS file containing the beam data.
        frequency : float
            Frequency in MHz to extract the beam data for.
        nside : int
            HEALPix nside parameter for the output beam map.
        orientation : str
            Orientation of the beam, either "x" or "y".

        Returns
        -------
        Beam
            An instance of the Beam class with the loaded beam data.

        Raises
        ------
        ValueError
            If the orientation is not "x" or "y".

        Notes
        -----
        Horizon cut is applied in the simulator.

        """
        with fits.open(path) as f:
            # shape is freq, theta, phi
            Eth = f["Etheta_real"].data + 1j * f["Etheta_imag"].data
            Eph = f["Ephi_real"].data + 1j * f["Ephi_imag"].data
            freq_ix = np.argwhere(f["freq"].data == frequency)[0, 0]
            Eth = Eth[freq_ix]
            Eph = Eph[freq_ix]
            # remove phi = 360 deg duplicate
            Eth = Eth[..., :-1]
            Eph = Eph[..., :-1]

        jones = np.array([Eth, Eph])
        # add lower hempisphere but cut off one theta pixel
        lower_hemi = np.zeros_like(jones)[:, :-1, :]
        jones = np.concatenate([jones, lower_hemi], axis=1)

        # rotate the beam to get the other pseudo-dipole
        if orientation == "y":
            jones_x = rotate_beam(jones, 270)
            jones_y = jones
        elif orientation == "x":
            jones_x = jones
            jones_y = rotate_beam(jones, 90)
        else:
            raise ValueError("Orientation must be either 'x' or 'y'")

        grid = HealpixGrid(nside=nside, horizon=False)
        # the lusee beam is defined on 1deg resolution grid
        th = np.radians(np.linspace(0, 180, num=181))
        ph = np.radians(np.linspace(0, 359, num=360))
        jones_x_hp = grid.interp_hp(jones_x, th, ph)
        jones_y_hp = grid.interp_hp(jones_y, th, ph)

        return cls(jones_x_hp, jones_y_hp)

    @classmethod
    def short_dipole(cls, nside=128):
        """
        Create a short dipole beam.

        Parameters
        ----------
        nside : int
            HEALPix nside parameter for the output beam map.

        Returns
        -------
        Beam
            An instance of the Beam class with the short dipole beam
            data.

        Notes
        -----
        Horizon cut is applied in the simulator.

        """
        grid = HealpixGrid(nside=nside, horizon=False)
        # X beam
        Eth = -np.cos(grid.theta) * np.cos(grid.phi)
        Eph = np.sin(grid.phi)
        jones_x = np.array([Eth, Eph])
        # Y beam
        Eth = -np.cos(grid.theta) * np.sin(grid.phi)
        Eph = -np.cos(grid.phi)
        jones_y = np.array([Eth, Eph])

        return cls(jones_x, jones_y)

    def precompute_weights(self):
        """
        Precompute the weights needed for convolution with the sky and
        store them in the `weights` attribute, which is a dictionary
        containing the precomputed weights. The keys are 'wI_x',
        'wQ_x', 'wU_x', 'wI_y', 'wQ_y', 'wU_y', 'wI_xy', 'wQ_xy',
        'wU_xy'. Values are np.ndarrays of shape (npix,).

        Notes
        -----
        Weights get multiplied with the sky Stokes parameters I, Q, U.
        V is taken to be zero.

        """
        if self.weights is not None:
            return

        Ex_th = self.jones_x[0]
        Ex_ph = self.jones_x[1]
        Ey_th = self.jones_y[0]
        Ey_ph = self.jones_y[1]

        weights = {}
        weights["wI_x"] = 1 / 2 * (np.abs(Ex_th) ** 2 + np.abs(Ex_ph) ** 2)
        weights["wQ_x"] = 1 / 2 * (np.abs(Ex_th) ** 2 - np.abs(Ex_ph) ** 2)
        weights["wU_x"] = np.real(Ex_th * np.conj(Ex_ph))
        weights["wI_y"] = 1 / 2 * (np.abs(Ey_th) ** 2 + np.abs(Ey_ph) ** 2)
        weights["wQ_y"] = 1 / 2 * (np.abs(Ey_th) ** 2 - np.abs(Ey_ph) ** 2)
        weights["wU_y"] = np.real(Ey_th * np.conj(Ey_ph))
        weights["wI_xy"] = (
            1 / 2 * (Ex_th * np.conj(Ey_th) + Ex_ph * np.conj(Ey_ph))
        )
        weights["wQ_xy"] = (
            1 / 2 * (Ex_th * np.conj(Ey_th) - Ex_ph * np.conj(Ey_ph))
        )
        weights["wU_xy"] = (
            1 / 2 * (Ex_th * np.conj(Ey_ph) + Ex_ph * np.conj(Ey_th))
        )

        self.weights = weights
