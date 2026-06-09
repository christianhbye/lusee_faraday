"""Full-band channelized Faraday simulation.

Runs the optimized visibility computation on a FrequencyPlan's
deduplicated raw grid and channelizes the result. The FR (Faraday)
Stokes are channelized to capture sub-channel bandwidth depolarization;
the no-FR Stokes are smooth in frequency and are evaluated directly at
the channel centers (exact to the smooth-spectrum limit, far cheaper).
"""

import numpy as np

from .fast_sim import compute_vis_fast_parallel
from .sim import Simulator


def simulate_channelized(
    plan, I_topo, Q_topo, U_topo, rm_topo, beam, mask,
    nproc=None, **kwargs
):
    """Channelized FR and no-FR Stokes for a FrequencyPlan.

    Returns (out, table) where out has keys pI_FR/pQ_FR/pU_FR and
    pI_noFR/pQ_noFR/pU_noFR each shape (ntimes, nchan), and table is
    the plan's channel table. Extra kwargs go to compute_vis_fast.
    """
    sim_freqs = plan.sim_freqs()
    table = plan.channel_table

    # FR: channelize the rippled spectrum (captures depolarization)
    vis_fr = compute_vis_fast_parallel(
        I_topo, Q_topo, U_topo, rm_topo, beam, sim_freqs, mask,
        nproc=nproc, **kwargs,
    )
    I_fr, Q_fr, U_fr = Simulator.compute_stokes(vis_fr)
    out = {
        "pI_FR": plan.channelize(I_fr),
        "pQ_FR": plan.channelize(Q_fr),
        "pU_FR": plan.channelize(U_fr),
    }

    # no-FR: smooth spectrum, evaluate at channel centers (rm = 0)
    zeros = np.zeros_like(rm_topo)
    vis_nf = compute_vis_fast_parallel(
        I_topo, Q_topo, U_topo, zeros, beam, table["nu"], mask,
        nproc=nproc, **kwargs,
    )
    I_nf, Q_nf, U_nf = Simulator.compute_stokes(vis_nf)
    out["pI_noFR"] = I_nf
    out["pQ_noFR"] = Q_nf
    out["pU_noFR"] = U_nf
    return out, table
