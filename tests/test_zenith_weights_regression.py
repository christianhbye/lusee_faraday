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
