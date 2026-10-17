"""drvec: VARMA-VEC cointegration models by exact maximum likelihood.

The Python port of the C program drvec (atsw-gui, engines/drvec). Its oracle
is drvec C 0.13 (atsw-gui ae95446; the tag drvec-v0.13 comes with the
release): every fit is homologated against that program's output (logL to
1e-6, estimates to 1e-4).
"""
try:
    from importlib.metadata import version as _version
    __version__ = _version("drvec")
except Exception:                                   # noqa: BLE001
    __version__ = "0.0.0.dev0"

from .api import FitResult, Param, fit, rank_test, rungs   # noqa: E402,F401
from .data import Data, read_pre                           # noqa: E402,F401
from .fit import LineSearchRoundoff                        # noqa: E402,F401
from .params import Spec                                   # noqa: E402,F401
