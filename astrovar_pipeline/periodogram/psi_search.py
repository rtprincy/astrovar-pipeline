from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple
import importlib
from astropy.timeseries import LombScargle
from lk_stat_package import lk_stat


def freq_grid(times, oversample_factor=10, f0=None, fn=None, verbose=False):
    times = np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if (f0 is None) | (int(f0) == 0):
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times))

    if verbose:
        print("Time: ", times)
        print("Minimum freq: ", f0)
        print("Maximum freq: ", fn)
        print("Freq resolution prior oversampling: ", df)
        print("Oversampling factor: ", oversample_factor)

    return np.arange(f0, fn, df / oversample_factor)


def psi_periodogram(
    time, mag, err, min_freq, max_freq, oversample=5
) -> Dict[str, np.ndarray]:
    """Compute Ψ-statistic = 2*LSP / theta, combining LS and LK.
    Returns dict with 'freq', 'lsp', 'theta', 'psi'.
    """

    frequencies = freq_grid(
        time, oversample_factor=oversample, f0=min_freq, fn=max_freq
    )
    periods = 1 / frequencies

    lsp = LombScargle(time, mag, err, nterms=1).power(
        frequencies, method="cython", normalization="psd"
    )
    theta = lk_stat(periods, mag, err, time)

    # psi = (2 * lsp) / theta

    return {
        "freq": frequencies.astype(np.float32),
        "lsp": lsp.astype(np.float32),
        "theta": theta.astype(np.float32),
    }
