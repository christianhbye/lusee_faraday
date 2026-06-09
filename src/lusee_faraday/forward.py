"""Reusable linear forward operator for the Fisher forecast.

compute_vis_fast is linear in the reference (Q, U) maps, and Faraday
rotation enters only through cos/sin(2*alpha*RM*lambda^2). pol_response
wraps it into a single complex polarized response P = pQ + i*pU as a
function of an intrinsic (I, Q, U) topocentric sky and a Faraday
amplitude alpha, so Fisher Jacobian columns can be built by applying it
to basis sky maps (sky derivatives) and by perturbing alpha.
"""

import healpy as hp
import numpy as np

from .fast_sim import compute_vis_fast, precompute_rotated_maps
from .sim import Simulator


def rotate_pol_maps(Q_ref, U_ref, rm_gal, times, nside, site_loc):
    """Rotate a polarized (Q, U) reference sky + RM to topocentric.

    Thin wrapper over precompute_rotated_maps with a zero I map, for
    building Jacobian columns from basis Q/U maps. Returns
    (Q_topo, U_topo, rm_topo), each shape (ntimes, npix).
    """
    zero = np.zeros(hp.nside2npix(nside))
    _, Q_topo, U_topo, rm_topo = precompute_rotated_maps(
        zero, Q_ref, U_ref, rm_gal, times, nside, site_loc
    )
    return Q_topo, U_topo, rm_topo


def pol_response(
    I_topo, Q_topo, U_topo, rm_topo, beam, mask, freqs, alpha=1.0, **kwargs
):
    """Complex polarized response P = pQ + i*pU at given frequencies.

    Linear in (Q_topo, U_topo). alpha scales the rotation measure
    (alpha=1 is the physical Faraday sky, alpha=0 unrotated). Extra
    kwargs forward to compute_vis_fast. Returns complex (ntimes, nfreq).
    """
    vis = compute_vis_fast(
        I_topo,
        Q_topo,
        U_topo,
        alpha * np.asarray(rm_topo),
        beam,
        freqs,
        mask,
        **kwargs,
    )
    _, Q, U = Simulator.compute_stokes(vis)
    return Q + 1j * U
