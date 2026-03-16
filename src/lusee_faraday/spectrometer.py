import numpy as np


class SpectrometerResponse:
    """
    Spectrometer bin response for LuSEE, supporting wide (parent) and
    narrow (zoom) bin modes.

    The response data defines how the spectrometer weights the sky signal
    across frequency. The wide bin is the standard 25 kHz channel. The
    narrow (zoom) bins split one parent bin into 64 sub-bins.

    Parameters
    ----------
    freq_offset_hz : np.ndarray
        Frequency offsets in Hz relative to bin center. Shape (npoints,).
    wide : np.ndarray
        Wide (parent) bin response at each offset. Shape (npoints,).
    narrow : np.ndarray
        Narrow (zoom) bin responses. Shape (npoints, 64).

    """

    N_ZOOM = 64
    BIN_WIDTH_HZ = 25000.0  # 51.2 MHz / 2048 channels

    def __init__(self, freq_offset_hz, wide, narrow):
        self.freq_offset_hz = freq_offset_hz
        self.wide = wide
        self.narrow = narrow
        self._wide_norm = wide / wide.sum()
        self._narrow_norm = narrow / narrow.sum(axis=0, keepdims=True)

    @classmethod
    def from_file(cls, path):
        """
        Load spectrometer response from the standard text file.

        Parameters
        ----------
        path : str or Path
            Path to spectrometer_bin_response.txt.

        Returns
        -------
        SpectrometerResponse

        """
        data = np.loadtxt(path)
        freq_offset_hz = data[:, 1]
        wide = data[:, 2]
        narrow = data[:, 3:]  # columns zm0 through zm63
        return cls(freq_offset_hz, wide, narrow)

    @property
    def freq_offset_mhz(self):
        """Frequency offsets in MHz."""
        return self.freq_offset_hz * 1e-6

    @property
    def df_hz(self):
        """Frequency step in Hz."""
        return self.freq_offset_hz[1] - self.freq_offset_hz[0]

    def freqs(self, center_mhz):
        """
        Absolute frequencies in MHz for simulation.

        Parameters
        ----------
        center_mhz : float
            Center frequency of the bin in MHz.

        Returns
        -------
        np.ndarray
            Frequencies in MHz at which to evaluate the sky signal.

        """
        return center_mhz + self.freq_offset_mhz

    def decimate(self, factor):
        """
        Return a new SpectrometerResponse with every `factor`-th sample.

        Parameters
        ----------
        factor : int
            Decimation factor.

        Returns
        -------
        SpectrometerResponse

        """
        return SpectrometerResponse(
            self.freq_offset_hz[::factor],
            self.wide[::factor],
            self.narrow[::factor],
        )

    def apply_wide(self, spectrum):
        """
        Convolve a spectrum with the wide (parent) bin response.

        Parameters
        ----------
        spectrum : np.ndarray
            Sky signal evaluated at the frequency offsets. Shape
            ``(..., npoints)`` where npoints matches the number of
            frequency offsets.

        Returns
        -------
        np.ndarray
            Weighted average over frequency. Shape ``(...,)``.

        """
        return np.tensordot(spectrum, self._wide_norm, axes=([-1], [0]))

    def apply_narrow(self, spectrum):
        """
        Convolve a spectrum with the 64 narrow (zoom) bin responses.

        Parameters
        ----------
        spectrum : np.ndarray
            Sky signal evaluated at the frequency offsets. Shape
            ``(..., npoints)``.

        Returns
        -------
        np.ndarray
            Weighted averages for each zoom bin. Shape ``(..., 64)``.

        """
        return np.tensordot(spectrum, self._narrow_norm, axes=([-1], [0]))
