"""Frequency plan: per-parent choice of zoom or wide channelization.

A plan is a list of (center_mhz, mode) specs. It builds the minimal
deduplicated raw frequency grid to simulate (`sim_freqs`) and maps a
simulated raw spectrum onto the spectrometer channels (`channelize`),
reusing SpectrometerResponse.apply_wide / apply_narrow. All parent
centers and response offsets share a common integer-Hz lattice, so
overlapping windows are deduplicated exactly without interpolation.
"""

import numpy as np

from .utils import freqs_lusee
from .rmsynth import lambda2

BIN_WIDTH_HZ = 25000.0
N_ZOOM = 64


def _snap_to_lusee(center_mhz):
    f = freqs_lusee()
    return float(f[np.argmin(np.abs(f - center_mhz))])


class FrequencyPlan:
    def __init__(self, response, specs, decimation=1, support=1.0):
        """response: SpectrometerResponse. specs: list of
        (center_mhz, mode), mode in {"zoom", "wide"}.
        decimation: int applied to all specs, or a dict keyed by mode
        (must include every mode present in specs).
        support: if < 1, truncate the response to that weight fraction.
        """
        base = response.truncate(support) if support < 1.0 else response
        self.specs = [(_snap_to_lusee(c), m) for c, m in specs]
        self._resp = []
        self._off_hz = []
        abs_hz = []
        for c, mode in self.specs:
            dec = (
                decimation[mode]
                if isinstance(decimation, dict)
                else decimation
            )
            r = base.decimate(dec) if dec > 1 else base
            off = np.round(r.freq_offset_hz).astype(np.int64)
            self._resp.append(r)
            self._off_hz.append(off)
            abs_hz.append(
                np.round(c * 1e6).astype(np.int64) + off
            )
        self._grid_hz = np.unique(np.concatenate(abs_hz))
        self._idx = [
            np.searchsorted(self._grid_hz, a) for a in abs_hz
        ]

    def sim_freqs(self):
        """Sorted unique absolute frequencies to simulate (MHz)."""
        return self._grid_hz * 1e-6

    def channelize(self, raw):
        """Map a raw spectrum (..., nraw) aligned with sim_freqs() to
        the spectrometer channels (..., nchan)."""
        out = []
        for (_, mode), r, idx in zip(self.specs, self._resp, self._idx):
            window = raw[..., idx]
            if mode == "wide":
                out.append(r.apply_wide(window)[..., None])
            else:
                out.append(r.apply_narrow(window))
        return np.concatenate(out, axis=-1)

    @property
    def channel_table(self):
        """Per-channel nu (MHz), lambda2 (m^2), dnu (Hz). nu is the
        response-weighted effective frequency (correct for the
        non-monotonic zoom ordering)."""
        nu = self.channelize(self.sim_freqs())
        dnu = []
        for _, mode in self.specs:
            if mode == "wide":
                dnu.append(BIN_WIDTH_HZ)
            else:
                dnu.extend([BIN_WIDTH_HZ / N_ZOOM] * N_ZOOM)
        dnu = np.array(dnu)
        return {"nu": nu, "lambda2": lambda2(nu), "dnu": dnu}
