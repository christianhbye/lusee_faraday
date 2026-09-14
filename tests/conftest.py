"""Shared test configuration.

The x64 setdefault is a backstop: pytest imports ``conftest.py`` before
any test module, so it holds even for a module that forgets its own.
Each test module still sets it at the top too, because a module run
directly (``python tests/test_x.py``) never goes through conftest.

The former ``data_dir`` fixture was removed in the final fix round: no
test requested it (``grep -rn data_dir tests/`` returned only its own
definition), and the justification recorded for keeping it -- that
``test_response_two_port.py`` used it -- was not true; that module
builds synthetic dipoles.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
