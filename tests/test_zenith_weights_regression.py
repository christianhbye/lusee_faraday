import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RESPONSE = REPO / "data" / "BGL_v16" / "lusee_bgl_v16_response_v3.fits"
BASELINES = json.loads(
    (REPO / "tests" / "fixtures" / "regression_baselines.json").read_text()
)

pytestmark = pytest.mark.skipif(
    not RESPONSE.exists(), reason="BGL_v16 response artifact not present"
)


@pytest.mark.slow
@pytest.mark.parametrize("center", [30.0, 10.0, 50.0])
def test_ortho_weights_null_zenith_polarization(center):
    import jax

    jax.config.update("jax_enable_x64", True)
    from lusee.ReceiverImpedance import JFETReceiver

    from lusee_faraday import polarimeter as pol
    from lusee_faraday import response as rsp

    resp = rsp.load_response(RESPONSE)
    x, y, C0 = pol.zenith_vectors(resp, JFETReceiver(), center)
    stokes = pol.pseudo_stokes(C0, x, y)
    residual = np.abs(stokes[1:]).max() / stokes[0]
    assert residual < 10 * BASELINES["zenith_null_ortho_max"]["value"]

    # Pin against the published Table 1 (report.tex) values themselves,
    # not just the residual: a Loewdin G^{-1/2} transform nulls the
    # zenith leakage of *any* Hermitian positive-definite C0, so the
    # residual check above cannot detect a wrong response file, a wrong
    # receiver model, or mislabelled/sign-flipped ports.  This
    # comparison would catch all of those, plus a change in the
    # dominant-dipole phase convention Table 1's caption describes.
    table = BASELINES["zenith_ortho_vectors_table1"]
    band = table["bands"][f"{center:g}"]
    atol = table["atol"]
    expected_x = np.array([complex(re, im) for re, im in band["x_p"]])
    expected_y = np.array([complex(re, im) for re, im in band["y_p"]])
    np.testing.assert_allclose(x, expected_x, atol=atol)
    np.testing.assert_allclose(y, expected_y, atol=atol)
