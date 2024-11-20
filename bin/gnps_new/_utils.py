import numpy as np
from numba import njit
from dataclasses import dataclass


@dataclass
class Spectrum:
    """
    A class to represent a mass spectrum with its associated metadata.
    """
    # file: str
    scan: int
    precursor_mz: float
    rt: float
    charge: int
    tic: float
    peaks: np.ndarray
    cleaned_peaks: np.ndarray = None


    def __post_init__(self):
        """Validate peaks data"""
        assert isinstance(self.peaks, np.ndarray), "Peaks must be numpy array"
        assert self.peaks.ndim == 2, "Peaks must be 2D array"
        assert self.peaks.shape[1] == 2, "Peaks must have shape (n, 2) for (mz, intensity)"

        # Ensure peaks are sorted by m/z
        sort_idx = np.argsort(self.peaks[:, 0])
        self.peaks = self.peaks[sort_idx]

        # do some roundings, reduce memory usage
        self.precursor_mz = round(self.precursor_mz, 4)
        self.rt = round(self.rt, 4)
        self.tic = round(self.tic)

        # Ensure float32 type
        self.peaks = self.peaks.astype(np.float32)


@njit
def clean_peaks(peaks, prec_mz,
                rel_int_threshold=0.01, prec_mz_removal_da=1.5, peak_transformation='sqrt'):
    """
    Clean MS/MS peaks
    """

    # Ensure peaks are 2D
    if peaks.ndim == 1:
        peaks = peaks.reshape(-1, 2)

    peaks = peaks[np.bitwise_and(peaks[:, 0] > 0, peaks[:, 1] > 0)]

    if peaks.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Remove low intensity peaks
    peaks = peaks[peaks[:, 1] > rel_int_threshold * np.max(peaks[:, 1])]

    if peaks.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Remove peaks with mz > prec_mz - prec_mz_removal_da
    max_allowed_mz = prec_mz - prec_mz_removal_da
    peaks = peaks[peaks[:, 0] < max_allowed_mz]

    if peaks.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # Transform peak intensities
    if peak_transformation is not None:
        if peak_transformation == 'sqrt':
            peaks[:, 1] = np.sqrt(peaks[:, 1])

    return peaks

