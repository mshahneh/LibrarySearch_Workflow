import numpy as np
from dataclasses import dataclass


@dataclass
class Spectrum:
    """Simple spectrum class that stores peaks as 2D numpy array"""
    file: str
    scan: int
    precursor_mz: float
    rt: float
    charge: int
    tic: float
    peaks: np.ndarray


    def __post_init__(self):
        """Validate peaks data"""
        assert isinstance(self.peaks, np.ndarray), "Peaks must be numpy array"
        assert self.peaks.ndim == 2, "Peaks must be 2D array"
        assert self.peaks.shape[1] == 2, "Peaks must have shape (n, 2) for (mz, intensity)"
        # Ensure peaks are sorted by m/z
        sort_idx = np.argsort(self.peaks[:, 0])
        self.peaks = self.peaks[sort_idx]
        # Ensure float32 type
        self.peaks = self.peaks.astype(np.float32)


@dataclass
class SpectrumMatch:
    """Simple class to store a matched spectrum"""
    file: str
    scan: int
    precursor_mz: float
    rt: float
    charge: int
    tic: float

    lib_id: str
    score: float
    matches: int
    mz_error_ppm: float
    mass_diff: float
