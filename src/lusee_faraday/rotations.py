import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord


def get_rot_mat(topo_frame):
    """
    Compute the rotation matrix from Galactic to Topocentric frame.

    Parameters
    ----------
    topo_frame : astropy.coordinates.BaseCoordinateFrame
        The topocentric frame to convert to.

    Returns
    -------
    rmat : ndarray
        Rotation matrix from Galactic to Topocentric frame.

    """
    x, y, z = np.eye(3)
    sc = SkyCoord(
        x=x, y=y, z=z, representation_type="cartesian", frame=topo_frame
    )
    rmat = sc.transform_to("galactic").cartesian.xyz.value.T
    return rmat


def rotmat_to_eulerZYX(rmat):
    """
    Convert a rotation matrix to ZYX Euler angles. This is the default
    convention used in HEALPix.

    Parameters
    ----------
    rmat : ndarray
        Rotation matrix.

    Returns
    -------
    alpha : float
        Rotation angle around Z axis.
    beta : float
        Rotation angle around Y axis.
    gamma : float
        Rotation angle around X axis.

    """
    beta = -np.arcsin(rmat[0, 2])  # pitch
    cos_beta = np.cos(beta)
    if np.isclose(cos_beta, 0):  # gimbal lock
        gamma = 0
        alpha = np.arctan2(-rmat[1, 0], rmat[1, 1])
    else:
        gamma = np.arctan2(rmat[1, 2] / cos_beta, rmat[2, 2] / cos_beta)
        alpha = np.arctan2(rmat[0, 1] / cos_beta, rmat[0, 0] / cos_beta)
    return alpha, -beta, gamma  # return -beta to match HEALPix convention


def gal2topo(I_map, Q_map, U_map, topo_frame=None, euler=None):
    """
    Convert Stokes I, Q, U maps from Galactic to Topocentric frame. The
    input is one map or a list of maps, with shape (Npix,) or
    (Nmaps, Npix).

    Parameters
    ----------
    I_map : array_like
        Stokes I map in Galactic coordinates.
    Q_map : array_like
        Stokes Q map in Galactic coordinates.
    U_map : array_like
        Stokes U map in Galactic coordinates.
    topo_frame : astropy.coordinates.BaseCoordinateFrame
        The topocentric frame to convert to.
    euler : tuple
        Precomputed Euler angles (alpha, beta, gamma) for the rotation.
        If None, they will be computed from the topo_frame.

    Returns
    -------
    I_topo : ndarray
        Stokes I map in Topocentric coordinates.
    Q_topo : ndarray
        Stokes Q map in Topocentric coordinates.
    U_topo : ndarray
        Stokes U map in Topocentric coordinates.

    """
    if euler is None:
        if topo_frame is None:
            raise ValueError("Either topo_frame or euler must be provided.")
        rmat = get_rot_mat(topo_frame)
        euler = rotmat_to_eulerZYX(rmat)

    rot = hp.Rotator(
        rot=euler, coord=None, inv=False, deg=False, eulertype="ZYX"
    )
    I_map = np.atleast_2d(I_map)
    Q_map = np.atleast_2d(Q_map)
    U_map = np.atleast_2d(U_map)
    I_topo = np.empty_like(I_map)
    Q_topo = np.empty_like(Q_map)
    U_topo = np.empty_like(U_map)
    n_maps = I_map.shape[0]
    for i in range(n_maps):
        stokes = np.array([I_map[i], Q_map[i], U_map[i]])
        I_topo[i], Q_topo[i], U_topo[i] = rot.rotate_map_alms(stokes)
    return I_topo, Q_topo, U_topo
