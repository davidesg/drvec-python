"""drvec: VARMA-VEC cointegration models by exact maximum likelihood.

The Python port of the C program drvec (atsw-gui, engines/drvec). Its oracle
is the tag drvec-v0.12: every fit is homologated against that program's
output (logL to 1e-6, estimates to 1e-4).
"""
try:
    from importlib.metadata import version as _version
    __version__ = _version("drvec")
except Exception:                                   # noqa: BLE001
    __version__ = "0.0.0.dev0"
