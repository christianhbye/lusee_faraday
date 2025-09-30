import numpy as np

def freqs_lusee():
    """
    LuSEE has 2048 frequency channels spanning 0 to 51.2 MHz.
    """
    return np.linspace(0, 51.2, num=2048, endpoint=False)

def freqs_zoom(center=30):
    """
    Split a channel into 64 sub-channels.
    """
    f = freqs_lusee()
    idx = np.argmin(np.abs(f - center))
    fmin = f[idx]
    fmax = f[idx + 1]
    return np.linspace(fmin, fmax, num=64, endpoint=False)
