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
    def __init__(self, response, specs, decimation=1):
        """response: SpectrometerResponse. specs: list of
        (center_mhz, mode) with mode in {"zoom", "wide"}.
        decimation: subsample the raw response grid."""
        self.response = (
            response.decimate(decimation) if decimation > 1 else response
        )
        self.specs = [(_snap_to_lusee(c), m) for c, m in specs]
        self._off_hz = np.round(self.response.freq_offset_hz).astype(np.int64)
        abs_hz = [
            np.round(c * 1e6).astype(np.int64) + self._off_hz
            for c, _ in self.specs
        ]
        self._grid_hz = np.unique(np.concatenate(abs_hz))
        self._idx = [np.searchsorted(self._grid_hz, a) for a in abs_hz]

    def sim_freqs(self):
        """Sorted unique absolute frequencies to simulate (MHz)."""
        return self._grid_hz * 1e-6

    def channelize(self, raw):
        """Map a raw spectrum (..., nraw) aligned with sim_freqs() to
        the spectrometer channels (..., nchan)."""
        out = []
        for (_, mode), idx in zip(self.specs, self._idx):
            window = raw[..., idx]
            if mode == "wide":
                out.append(self.response.apply_wide(window)[..., None])
            else:
                out.append(self.response.apply_narrow(window))
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
