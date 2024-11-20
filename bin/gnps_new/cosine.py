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
    max_matches = min(len(spec1_mz), len(spec2_mz))
    matches = np.zeros((max_matches, 2), dtype=np.int32)
    match_count = 0
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
                matches[match_count, 0] = peak1_idx
                matches[match_count, 1] = peak2_idx
                match_count += 1

    return matches[:match_count]


@nb.njit
def collect_peak_pairs(spec1: np.ndarray, spec2: np.ndarray,
                       tolerance: float, mz_power: float,
                       intensity_power: float, shift: float = 0.0) -> np.ndarray:
    """Find and score matching peak pairs between spectra."""

    if len(spec1) == 0 or len(spec2) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Extract m/z values
    spec1_mz = spec1[:, 0].copy()  # Create copy to ensure contiguous array
    spec2_mz = spec2[:, 0].copy()  # Create copy to ensure contiguous array

    matches = find_matches(spec1_mz, spec2_mz, tolerance, shift)

    if len(matches) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Calculate scores for matches
    matching_pairs = np.zeros((len(matches), 4), dtype=np.float32)
    for i in range(len(matches)):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]
        power_prod_spec1 = (spec1[idx1, 0] ** mz_power) * \
                           (spec1[idx1, 1] ** intensity_power)
        power_prod_spec2 = (spec2[idx2, 0] ** mz_power) * \
                           (spec2[idx2, 1] ** intensity_power)
        matching_pairs[i, 0] = idx1
        matching_pairs[i, 1] = idx2
        matching_pairs[i, 2] = power_prod_spec1 * power_prod_spec2
        matching_pairs[i, 3] = power_prod_spec2

    # Sort by score descending
    sort_idx = np.argsort(-matching_pairs[:, 2])  # Negative for descending order
    return matching_pairs[sort_idx]


@nb.njit
def score_matches(matching_pairs: np.ndarray, spec1: np.ndarray,
                  spec2: np.ndarray, mz_power: float,
                  intensity_power: float, reverse: bool) -> Tuple[float, int]:
    """
    Calculate final similarity score from matching peaks.
    spec1 is the reference spectrum, spec2 is the query spectrum.
    """
    if len(matching_pairs) == 0:
        return 0.0, 0

    score = 0.0
    used_matches = 0
    used1 = np.zeros(len(spec1), dtype=nb.boolean)
    used2 = np.zeros(len(spec2), dtype=nb.boolean)
    matched_indices = np.zeros(len(matching_pairs), dtype=np.int32)

    # Find best non-overlapping matches
    for i in range(len(matching_pairs)):
        idx1 = int(matching_pairs[i, 0])
        idx2 = int(matching_pairs[i, 1])
        if not used1[idx1] and not used2[idx2]:
            score += matching_pairs[i, 2]
            used1[idx1] = True
            used2[idx2] = True
            matched_indices[used_matches] = idx2
            used_matches += 1

    if used_matches == 0:
        return 0.0, 0

    # Create matched peaks array
    spec2_matched = np.zeros((used_matches, 2), dtype=np.float32)
    for i in range(used_matches):
        spec2_matched[i, 0] = spec2[matched_indices[i], 0]
        spec2_matched[i, 1] = spec2[matched_indices[i], 1]

    # Calculate powers for normalization
    spec1_powers = np.zeros(len(spec1), dtype=np.float32)
    for i in range(len(spec1)):
        spec1_powers[i] = (spec1[i, 0] ** mz_power) * \
                          (spec1[i, 1] ** intensity_power)

    if reverse:
        spec2_powers = np.zeros(used_matches, dtype=np.float32)
        for i in range(used_matches):
            spec2_powers[i] = (spec2_matched[i, 0] ** mz_power) * \
                              (spec2_matched[i, 1] ** intensity_power)
    else:
        spec2_powers = np.zeros(len(spec2), dtype=np.float32)
        for i in range(len(spec2)):
            spec2_powers[i] = (spec2[i, 0] ** mz_power) * \
                              (spec2[i, 1] ** intensity_power)

    norm1 = np.sqrt(np.sum(spec1_powers * spec1_powers))
    norm2 = np.sqrt(np.sum(spec2_powers * spec2_powers))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0, used_matches

    score = score / (norm1 * norm2)
    return min(float(score), 1.0), used_matches


class CosineGreedy:
    """Calculate cosine similarity between mass spectra."""

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0,
                 intensity_power: float = 1.0, reverse: bool = False):
        """Initialize with given parameters."""
        self.tolerance = np.float32(tolerance)
        self.mz_power = np.float32(mz_power)
        self.intensity_power = np.float32(intensity_power)
        self.reverse = reverse

    def pair(self, qry_spec, ref_spec) -> Tuple[float, int]:
        """
        Calculate similarity between two spectra.
        """
        # Handle empty inputs
        if qry_spec is None or ref_spec is None:
            return 0.0, 0

        try:
            qry_spec = np.asarray(qry_spec, dtype=np.float32)
            ref_spec = np.asarray(ref_spec, dtype=np.float32)

            if qry_spec.size == 0 or ref_spec.size == 0:
                return 0.0, 0

            if qry_spec.ndim == 1:
                qry_spec = qry_spec.reshape(-1, 2)
            if ref_spec.ndim == 1:
                ref_spec = ref_spec.reshape(-1, 2)
        except:
            return 0.0, 0

        matching_pairs = collect_peak_pairs(
            ref_spec, qry_spec,
            self.tolerance, self.mz_power,
            self.intensity_power
        )

        return score_matches(
            matching_pairs, ref_spec, qry_spec,
            self.mz_power, self.intensity_power, self.reverse
        )


if __name__ == "__main__":
    peaks1=np.array([
        [100., 0.7],
        [150., 0.2],
        [200., 0.1],
        [201., 0.2]
    ], dtype=np.float32)

    peaks2=np.array([
            [105., 0.4],
            [150., 0.2],
            [190., 0.1],
            [200., 0.5]
        ], dtype=np.float32)

    # Example with reverse=False (forward cosine)
    cosine_standard = CosineGreedy(tolerance=0.05, reverse=False)
    score_standard, n_matches_standard = cosine_standard.pair(peaks1, peaks2)
    print(f"Standard Score: {score_standard:.3f}, Matches: {n_matches_standard}")

    # Example with reverse=True (reverse cosine)
    cosine_reverse = CosineGreedy(tolerance=0.05, reverse=True)
    score_reverse, n_matches_reverse = cosine_reverse.pair(peaks1, peaks2)
    print(f"Reverse Score: {score_reverse:.3f}, Matches: {n_matches_reverse}")