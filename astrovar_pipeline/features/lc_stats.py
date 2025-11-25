
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class LCStats:
    time: np.ndarray
    mag: np.ndarray
    err: np.ndarray|None = None

    def ptp_amplitude(self):
        return float(np.nanmax(self.mag) - np.nanmin(self.mag))

    def mean(self):
        return float(np.nanmean(self.mag))

    def median(self):
        return float(np.nanmedian(self.mag))

    def std(self):
        return float(np.nanstd(self.mag))

    def iqr(self):
        q3, q1 = np.nanpercentile(self.mag, [75,25])
        return float(q3 - q1)

    def abbe(self):
        diffs = np.diff(self.mag)
        return float(np.nanmean(diffs**2) / (2*np.nanvar(self.mag)) ) if np.nanvar(self.mag)>0 else float("nan")

    def to_features(self, prefix=""):
        d = {
            prefix+"ptp_amp": self.ptp_amplitude(),
            prefix+"mean_mag": self.mean(),
            prefix+"median_mag": self.median(),
            prefix+"std_mag": self.std(),
            prefix+"iqr_mag": self.iqr(),
            prefix+"abbe": self.abbe(),
        }
        return d
