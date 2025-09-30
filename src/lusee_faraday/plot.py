import healpy as hp


def plot_beam(beam, vmin=None, vmax=None):
    hp.mollview(
        beam.jones_x[0], title="E_x^theta", cbar=True, min=vmin, max=vmax
    )
    hp.mollview(
        beam.jones_x[1], title="E_x^phi", cbar=True, min=vmin, max=vmax
    )
    hp.mollview(
        beam.jones_y[0], title="E_y^theta", cbar=True, min=vmin, max=vmax
    )
    hp.mollview(
        beam.jones_y[1], title="E_y^phi", cbar=True, min=vmin, max=vmax
    )


def plot_beam_power(gain_x, gain_y, vmin=None, vmax=None):
    hp.mollview(gain_x, cbar=True, title="X beam", min=vmin, max=vmax)
    hp.mollview(gain_y, cbar=True, title="Y beam", min=vmin, max=vmax)
