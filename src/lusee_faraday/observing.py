"""Explicit nightside exposure of preselected lunar sidereal phase bins."""

import numpy as np

from .config import N_TIMES, SIDEREAL_DAY_S


def phase_exposures(
    duration_s,
    sidereal_turns,
    solar_altitude_deg,
    duty_cycle,
    bin_seconds=SIDEREAL_DAY_S / N_TIMES,
    night_limit_deg=-1.0,
):
    """Count complete, entirely nightside visits in a calendar interval.

    solar_altitude_deg maps relative seconds to altitude at the actual
    observing site. Three samples (bin edges and center) conservatively
    select nightside bins, with a declared solar-altitude margin. Partial
    endpoint bins are excluded. Zero exposure is recorded, not replaced by
    a nominal fraction of a lunation or a zero measurement-noise level.
    """
    turns = np.atleast_1d(np.asarray(sidereal_turns, float))
    if (
        not np.isfinite(duration_s)
        or duration_s <= 0
        or not np.isfinite(bin_seconds)
        or not 0 < bin_seconds < SIDEREAL_DAY_S
        or not np.isfinite(duty_cycle)
        or not 0 <= duty_cycle <= 1
        or not np.isfinite(night_limit_deg)
        or not -90 <= night_limit_deg <= 0
        or turns.ndim != 1
        or not turns.size
        or not np.all(np.isfinite(turns) & (turns >= 0) & (turns < 1))
    ):
        raise ValueError("invalid calendar, phase bin or observing duty")
    rows = []
    for phase in turns:
        visits = (
            phase + np.arange(int(np.ceil(duration_s / SIDEREAL_DAY_S)) + 1)
        ) * SIDEREAL_DAY_S
        visits = visits[
            (visits - bin_seconds / 2 >= 0)
            & (visits + bin_seconds / 2 <= duration_s)
        ]
        probes = visits[:, None] + np.array([-0.5, 0, 0.5]) * bin_seconds
        altitude = np.asarray(
            solar_altitude_deg(probes.ravel()), float
        ).reshape(probes.shape)
        if not np.all(np.isfinite(altitude)):
            raise ValueError("nonfinite solar altitude")
        selected = np.all(altitude <= night_limit_deg, axis=1)
        seconds = float(selected.sum() * bin_seconds * duty_cycle)
        rows.append(
            dict(
                sidereal_turn=float(phase),
                available=seconds > 0,
                exposure_s=seconds,
                eligible_visits=int(selected.sum()),
                candidate_visits=int(len(visits)),
                visit_seconds=visits[selected].tolist(),
                solar_altitudes_deg=altitude.tolist(),
                night_limit_deg=night_limit_deg,
                duty_cycle=duty_cycle,
                bin_seconds=bin_seconds,
            )
        )
    return rows


def site_solar_altitudes(start_time, relative_seconds, location):
    """Astropy/lunarsky solar altitude at the configured lunar site.

    Like luseepy's LunarCalendar implementation, evaluate scalar times to
    avoid its unsupported vector transform. Its fixed equatorial location
    is replaced by the caller's actual MoonLocation. No calendar cache is
    written in an upstream checkout.
    """
    import astropy.units as u
    from astropy.coordinates import get_sun
    from lunarsky import LunarTopo
    from lusee.spice_utils import ensure_lunarsky_moon_frame

    ensure_lunarsky_moon_frame()
    relative = np.atleast_1d(np.asarray(relative_seconds, float))
    times = start_time + relative * u.s
    return np.array(
        [
            get_sun(t)
            .transform_to(LunarTopo(location=location, obstime=t))
            .alt.to_value(u.deg)
            for t in times
        ]
    )


def sidereal_bin_window(lmax, bin_seconds=SIDEREAL_DAY_S / N_TIMES):
    """Exact boxcar average of exp(-i m angle) over one phase bin."""
    if (
        not isinstance(lmax, int)
        or lmax < 0
        or not np.isfinite(bin_seconds)
        or not 0 < bin_seconds <= SIDEREAL_DAY_S
    ):
        raise ValueError("invalid sidereal averaging window")
    return np.sinc(np.arange(-lmax, lmax + 1) * bin_seconds / SIDEREAL_DAY_S)
