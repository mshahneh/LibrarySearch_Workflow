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
    """
    Calculate entropy similarity score from matching peaks.
    spec1 is the reference spectrum, spec2 is the query spectrum.
    """
    if len(matching_pairs) == 0:
        return 0.0, 0

    # apply weight to intensity
    spec1 = apply_weight_to_intensity(spec1)
    spec2 = apply_weight_to_intensity(spec2)

    used_matches = 0
    used1 = np.zeros(len(spec1), dtype=np.bool_)
    used2 = np.zeros(len(spec2), dtype=np.bool_)

    # Initialize arrays for matched intensities
    matched_a_intensities = np.zeros(len(matching_pairs), dtype=np.float32)
    matched_b_intensities = np.zeros(len(matching_pairs), dtype=np.float32)

    # Find best non-overlapping matches
    for i in range(len(matching_pairs)):
        idx1 = int(matching_pairs[i, 0])
        idx2 = int(matching_pairs[i, 1])
        if not used1[idx1] and not used2[idx2]:
            # Store the intensities after applying power
            matched_a_intensities[used_matches] = (spec1[idx1, 0] ** mz_power) * (spec1[idx1, 1] ** intensity_power)
            matched_b_intensities[used_matches] = (spec2[idx2, 0] ** mz_power) * (spec2[idx2, 1] ** intensity_power)
            used1[idx1] = True
            used2[idx2] = True
            used_matches += 1

    if used_matches == 0:
        return 0.0, 0

    # Trim arrays to used matches only
    matched_a_intensities = matched_a_intensities[:used_matches]
    matched_b_intensities = matched_b_intensities[:used_matches]

    # Normalize intensities to sum to 1
    sum_a = np.sum((spec1[:, 0] ** mz_power) * (spec1[:, 1] ** intensity_power))
    if reverse:
        sum_b = np.sum(matched_b_intensities)
    else:
        sum_b = np.sum((spec2[:, 0] ** mz_power) * (spec2[:, 1] ** intensity_power))

    if sum_a <= 0 or sum_b <= 0:
        return 0.0, used_matches

    matched_a_intensities /= sum_a
    matched_b_intensities /= sum_b

    # Calculate entropy similarity
    peak_ab_intensities = matched_a_intensities + matched_b_intensities

    entropy_sim = 0.0
    for i in range(used_matches):
        if peak_ab_intensities[i] > 0:
            ab_term = peak_ab_intensities[i] * np.log2(peak_ab_intensities[i])
        else:
            ab_term = 0.0

        if matched_a_intensities[i] > 0:
            a_term = matched_a_intensities[i] * np.log2(matched_a_intensities[i])
        else:
            a_term = 0.0

        if matched_b_intensities[i] > 0:
            b_term = matched_b_intensities[i] * np.log2(matched_b_intensities[i])
        else:
            b_term = 0.0

        entropy_sim += ab_term - a_term - b_term

    entropy_sim /= 2
    return float(entropy_sim), used_matches


class EntropyGreedy:
    """Calculate entropy similarity between mass spectra."""

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


@nb.njit
def apply_weight_to_intensity(peaks: np.ndarray) -> np.ndarray:
    """
    Apply a weight to the intensity of a spectrum based on spectral entropy.
    """
    if peaks.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    # normalize intensity
    peak_sum = np.sum(peaks[:, 1])
    if peak_sum > 0:
        peaks[:, 1] /= peak_sum
    else:
        return peaks

    # Calculate the spectral entropy.
    entropy = 0.0
    if peaks.shape[0] > 0:
        entropy = -np.sum(peaks[:, 1] * np.log(peaks[:, 1]))

    # Copy the peaks.
    weighted_peaks = peaks.copy()

    # Apply the weight.
    if entropy < 3:
        weight = 0.25 + 0.25 * entropy
        weighted_peaks[:, 1] = np.power(peaks[:, 1], weight)
        intensity_sum = np.sum(weighted_peaks[:, 1])
        weighted_peaks[:, 1] /= intensity_sum

    return weighted_peaks


if __name__ == "__main__":
    from _utils import Spectrum
    # Example usage with the simplified Spectrum class
    spectrum_1 = Spectrum(
        peaks=np.array([[69, 8.0], [86, 100.0], [99, 50.0]], dtype=np.float32)
    )

    spectrum_2 = Spectrum(
        peaks=np.array([[41, 38.0], [69, 66.0], [86, 999.0]], dtype=np.float32)
    )

    # Example with reverse=False (forward entropy)
    entropy_standard = EntropyGreedy(tolerance=0.05, reverse=False)
    score_standard, n_matches_standard = entropy_standard.pair(spectrum_1, spectrum_2)
    print(f"Standard Score: {score_standard:.3f}, Matches: {n_matches_standard}")

    # Example with reverse=True (reverse entropy)
    entropy_reverse = EntropyGreedy(tolerance=0.05, reverse=True)
    score_reverse, n_matches_reverse = entropy_reverse.pair(spectrum_1, spectrum_2)
    print(f"Reverse Score: {score_reverse:.3f}, Matches: {n_matches_reverse}")

