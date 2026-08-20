"""Confirm croissant's engine resolution at the resolutions we use.

The spec claims croissant 1c4d6c5 added a memory cap to
_low_pass_in_one_step, so PolarizedSky(nside=512).compute_alm(lmax=30)
takes the native transform and truncates instead of building an ~800 GB
dense operator.  Verify rather than trust.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import croissant as cro
    import healpy as hp

    nside = 512
    npix = hp.nside2npix(nside)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(1, 4, npix)) * 1e-3
    sky = cro.PolarizedSky(data, np.array([30.0]), sampling="healpix")
    print("resolved engines:", sky.engine)
    print("reasons:", sky.engine_reason)
    alm = np.asarray(sky.compute_alm(lmax=30))
    print("alm shape:", alm.shape)
    assert alm.shape == (1, 4, 31, 61), alm.shape
    assert "dense" not in set(sky.engine.values()), sky.engine
    print("OK: no dense engine at nside=512, lmax=30")


if __name__ == "__main__":
    main()
