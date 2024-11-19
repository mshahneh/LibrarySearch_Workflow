"""
This file contains code modified from the matchms project
(https://github.com/matchms/matchms)
Copyright matchms Team 2020

Modified by Shipei Xing in 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import numba as nb
from typing import Tuple


@nb.njit
def find_matches(spec1_mz: np.ndarray, spec2_mz: np.ndarray,
                 tolerance: float, shift: float = 0.0) -> np.ndarray:
    """Find matching peaks between two spectra."""
    matches = []
    lowest_idx = 0

    for peak1_idx in range(len(spec1_mz)):
        mz = spec1_mz[peak1_idx]
        low_bound = mz - tolerance
        high_bound = mz + tolerance

        for peak2_idx in range(lowest_idx, len(spec2_mz)):
            mz2 = spec2_mz[peak2_idx] + shift
            if mz2 > high_bound:
                break
            if mz2 < low_bound:
                lowest_idx = peak2_idx
            else:
                matches.append((peak1_idx, peak2_idx))

    return np.array(matches, dtype=np.int32)


@nb.njit
def collect_peak_pairs(spec1: np.ndarray, spec2: np.ndarray,
                       tolerance: float, mz_power: float,
                       intensity_power: float, shift: float = 0.0) -> np.ndarray:
    """Find and score matching peak pairs between spectra."""
    # Find matching indices
    matches = find_matches(spec1[:, 0], spec2[:, 0], tolerance, shift)
    if len(matches) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Calculate scores for matches
    matching_pairs = np.zeros((len(matches), 4), dtype=np.float32)
    for i, (idx1, idx2) in enumerate(matches):
        power_prod_spec1 = (spec1[idx1, 0] ** mz_power) * \
                           (spec1[idx1, 1] ** intensity_power)
        power_prod_spec2 = (spec2[idx2, 0] ** mz_power) * \
                           (spec2[idx2, 1] ** intensity_power)
        matching_pairs[i] = [idx1, idx2,
                             power_prod_spec1 * power_prod_spec2,
                             power_prod_spec2]

    # Sort by score descending
    sort_idx = np.argsort(matching_pairs[:, 2])[::-1]
    return matching_pairs[sort_idx]


@nb.njit
def score_matches(matching_pairs: np.ndarray, spec1: np.ndarray,
                  spec2: np.ndarray, mz_power: float,
                  intensity_power: float, reverse: bool) -> Tuple[float, int]:
    """Calculate final similarity score from matching peaks."""
    if len(matching_pairs) == 0:
        return 0.0, 0

    score = 0.0
    used_matches = 0
    used1 = np.zeros(len(spec1), dtype=np.bool_)
    used2 = np.zeros(len(spec2), dtype=np.bool_)
    matched_indices = []

    # Find best non-overlapping matches
    for i in range(len(matching_pairs)):
        idx1 = int(matching_pairs[i, 0])
        idx2 = int(matching_pairs[i, 1])
        if not used1[idx1] and not used2[idx2]:
            score += matching_pairs[i, 2]
            used1[idx1] = True
            used2[idx2] = True
            used_matches += 1
            matched_indices.append(idx2)

    if used_matches == 0:
        return 0.0, 0

    # Create matched peaks array
    spec2_matched = np.zeros((used_matches, 2), dtype=np.float32)
    for i in range(used_matches):
        spec2_matched[i] = spec2[matched_indices[i]]

    # Normalize score
    spec1_power = np.power(spec1[:, 0], mz_power) * \
                  np.power(spec1[:, 1], intensity_power)
    if reverse:
        spec2_power = np.power(spec2_matched[:, 0], mz_power) * \
                      np.power(spec2_matched[:, 1], intensity_power)
    else:
        spec2_power = np.power(spec2[:, 0], mz_power) * \
                      np.power(spec2[:, 1], intensity_power)

    norm1 = np.sqrt(np.sum(spec1_power ** 2))
    norm2 = np.sqrt(np.sum(spec2_power ** 2))

    if norm1 == 0 or norm2 == 0:
        return 0.0, used_matches

    score /= (norm1 * norm2)
    return float(score), used_matches


class CosineGreedy:
    """Calculate cosine similarity between mass spectra."""

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0,
                 intensity_power: float = 1.0, reverse: bool = False):
        """
        Parameters
        ----------
        tolerance: float
            Maximum m/z difference for matching peaks
        mz_power: float
            Power to raise m/z values to
        intensity_power: float
            Power to raise intensities to
        reverse: bool
            If True, use reverse cosine mode where normalization is only done
            with peaks that match in the second spectrum
        """
        self.tolerance = np.float32(tolerance)
        self.mz_power = np.float32(mz_power)
        self.intensity_power = np.float32(intensity_power)
        self.reverse = reverse

    def pair(self, spectrum1, spectrum2) -> Tuple[float, int]:
        """Calculate similarity between two spectra.

        spectrum1: Spectrum
            Query spectrum
        spectrum2: Spectrum
            Reference spectrum
        """
        query, reference = spectrum1, spectrum2

        matching_pairs = collect_peak_pairs(
            reference.peaks, query.peaks,
            self.tolerance, self.mz_power,
            self.intensity_power
        )

        return score_matches(
            matching_pairs, reference.peaks, query.peaks,
            self.mz_power, self.intensity_power, self.reverse
        )


if __name__ == "__main__":
    from _utils import Spectrum
    # Example usage with the simplified Spectrum class
    spectrum_1 = Spectrum(
        'test', 0, 500.0, 0.0, 1, 1000,
        peaks=np.array([
            [100., 0.7],
            [150., 0.2],
            [200., 0.1],
            [201., 0.2]
        ], dtype=np.float32)
    )

    spectrum_2 = Spectrum(
        'test', 1, 500.0, 0.0, 1, 1000,
        peaks=np.array([
            [105., 0.4],
            [150., 0.2],
            [190., 0.1],
            [200., 0.5]
        ], dtype=np.float32)
    )

    # Example with reverse=False (forward cosine)
    cosine_standard = CosineGreedy(tolerance=0.05, reverse=False)
    score_standard, n_matches_standard = cosine_standard.pair(spectrum_1, spectrum_2)
    print(f"Standard Score: {score_standard:.3f}, Matches: {n_matches_standard}")

    # Example with reverse=True (reverse cosine)
    cosine_reverse = CosineGreedy(tolerance=0.05, reverse=True)
    score_reverse, n_matches_reverse = cosine_reverse.pair(spectrum_1, spectrum_2)
    print(f"Reverse Score: {score_reverse:.3f}, Matches: {n_matches_reverse}")