from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional

from ..core.types import MechanismResult, MechanismType

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

logger = logging.getLogger(__name__)


def _analyze_bit_planes(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze bit plane patterns for AI generation signatures.
    AI images often have different bit plane distributions.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Ensure 8-bit values
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Extract bit planes
        bit_planes = []
        for bit in range(8):
            bit_plane = ((gray >> bit) & 1) * 255
            bit_planes.append(bit_plane)

        # Analyze bit plane statistics
        bit_entropies = []
        bit_variances = []

        for plane in bit_planes:
            # Calculate entropy
            hist, _ = np.histogram(plane, bins=256, range=(0, 255))
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            bit_entropies.append(entropy)

            # Calculate variance
            variance = np.var(plane)
            bit_variances.append(variance)

        # AI images often have more uniform bit plane distributions
        # Real photos have more varied bit plane patterns

        # Check for suspicious bit plane patterns
        # Very low entropy in higher bits suggests AI generation
        high_bit_entropy = np.mean(bit_entropies[4:])  # Bits 4-7
        low_bit_entropy = np.mean(bit_entropies[:4])   # Bits 0-3

        if high_bit_entropy < 0.1:  # Very low high-bit entropy
            bit_score = 0.8
        elif high_bit_entropy < 0.3:
            bit_score = 0.6
        elif low_bit_entropy > 0.8 and high_bit_entropy < 0.5:
            bit_score = 0.7
        else:
            bit_score = 0.3

        diagnostics = {
            "bit_entropies": bit_entropies,
            "bit_variances": bit_variances,
            "high_bit_entropy": float(high_bit_entropy),
            "low_bit_entropy": float(low_bit_entropy),
        }

        return float(bit_score), diagnostics

    except Exception as e:
        logger.debug(f"Bit plane analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _detect_histogram_gaps(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect histogram gaps characteristic of AI generation.
    AI tools sometimes create images with missing pixel values.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Ensure 8-bit values
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Calculate histogram
        hist, bins = np.histogram(gray, bins=256, range=(0, 255))
        hist = hist / np.sum(hist)  # Normalize

        # Find gaps (consecutive zeros in histogram)
        gaps = []
        current_gap = 0

        for count in hist:
            if count == 0:
                current_gap += 1
            else:
                if current_gap > 0:
                    gaps.append(current_gap)
                    current_gap = 0

        if current_gap > 0:
            gaps.append(current_gap)

        # Large gaps suggest AI generation
        max_gap = max(gaps) if gaps else 0
        total_gap_pixels = sum(gaps)

        if max_gap > 10:  # Large gap
            gap_score = 0.8
        elif total_gap_pixels > 50:  # Many gaps
            gap_score = 0.6
        else:
            gap_score = 0.3

        diagnostics = {
            "max_gap": float(max_gap),
            "total_gaps": float(total_gap_pixels),
            "num_gaps": len(gaps),
        }

        return float(gap_score), diagnostics

    except Exception as e:
        logger.debug(f"Histogram gap analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _analyze_value_clustering(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze pixel value clustering patterns.
    AI images often have different clustering characteristics.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Ensure 8-bit values
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Reshape to 1D array
        pixels = gray.flatten()

        # Calculate pixel value distribution
        unique_values, counts = np.unique(pixels, return_counts=True)

        # Sort by count
        sorted_indices = np.argsort(-counts)
        top_values = unique_values[sorted_indices][:10]
        top_counts = counts[sorted_indices][:10]

        # Analyze clustering patterns
        # AI images often have more uniform distributions
        total_pixels = len(pixels)
        top_10_ratio = np.sum(top_counts) / total_pixels

        # Calculate coefficient of variation for top values
        if len(top_counts) > 1:
            cv = np.std(top_counts) / np.mean(top_counts)
        else:
            cv = 0

        # Very high concentration in few values suggests AI generation
        if top_10_ratio > 0.8:  # 80% of pixels in top 10 values
            clustering_score = 0.8
        elif cv < 0.2:  # Very uniform distribution
            clustering_score = 0.7
        else:
            clustering_score = 0.3

        diagnostics = {
            "top_10_ratio": float(top_10_ratio),
            "top_values_cv": float(cv),
            "num_unique_values": len(unique_values),
            "most_common_value": int(top_values[0]) if len(top_values) > 0 else 0,
        }

        return float(clustering_score), diagnostics

    except Exception as e:
        logger.debug(f"Value clustering analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _analyze_channel_correlation(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze RGB channel correlations.
    AI images often have different correlation patterns.
    """
    if np is None or len(img_array.shape) != 3:
        return 0.5, {"error": "NumPy not available or not RGB image"}

    try:
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        # Calculate correlations between channels
        r_g_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
        r_b_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
        g_b_corr = np.corrcoef(g.flatten(), b.flatten())[0, 1]

        # Average correlation
        avg_corr = (abs(r_g_corr) + abs(r_b_corr) + abs(g_b_corr)) / 3

        # AI images often have more extreme correlations
        if avg_corr > 0.95:  # Very high correlation
            corr_score = 0.8
        elif avg_corr > 0.9:  # High correlation
            corr_score = 0.6
        else:
            corr_score = 0.3

        # Also check for unusual correlation patterns
        corr_variance = np.var([abs(r_g_corr), abs(r_b_corr), abs(g_b_corr)])

        if corr_variance > 0.1:  # High variance in correlations
            corr_score = max(corr_score, 0.7)

        diagnostics = {
            "r_g_correlation": float(r_g_corr),
            "r_b_correlation": float(r_b_corr),
            "g_b_correlation": float(g_b_corr),
            "avg_correlation": float(avg_corr),
            "correlation_variance": float(corr_variance),
        }

        return float(corr_score), diagnostics

    except Exception as e:
        logger.debug(f"Channel correlation analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _analyze_saturation_distribution(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze color saturation distribution patterns.
    AI images often have different saturation characteristics.
    """
    if np is None or len(img_array.shape) != 3:
        return 0.5, {"error": "NumPy not available or not RGB image"}

    try:
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        # Calculate saturation (distance from grayscale)
        gray = (r + g + b) / 3.0
        saturation = np.sqrt(((r - gray)**2 + (g - gray)**2 + (b - gray)**2) / 3)

        # Analyze saturation statistics
        sat_mean = float(np.mean(saturation))
        sat_std = float(np.std(saturation))
        sat_max = float(np.max(saturation))

        # Calculate saturation histogram
        sat_hist, _ = np.histogram(saturation, bins=50, range=(0, np.sqrt(3*255**2/3)))

        # Find peaks in saturation distribution
        peaks = []
        for i in range(1, len(sat_hist) - 1):
            if sat_hist[i] > sat_hist[i-1] and sat_hist[i] > sat_hist[i+1]:
                peaks.append(i)

        # AI images often have bimodal or unusual saturation distributions
        if len(peaks) > 3:  # Multiple peaks
            sat_score = 0.7
        elif sat_std / sat_mean < 0.3:  # Very low relative standard deviation
            sat_score = 0.6
        elif sat_max > 200:  # Very high saturation
            sat_score = 0.7
        else:
            sat_score = 0.3

        diagnostics = {
            "saturation_mean": sat_mean,
            "saturation_std": sat_std,
            "saturation_max": sat_max,
            "num_saturation_peaks": len(peaks),
        }

        return float(sat_score), diagnostics

    except Exception as e:
        logger.debug(f"Saturation distribution analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _analyze_gradient_patterns(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze gradient patterns for AI generation signatures.
    AI images often have different gradient distributions.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Calculate gradients
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)

        # Calculate gradient magnitude and direction
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)

        # Analyze gradient statistics
        mag_mean = float(np.mean(grad_mag))
        mag_std = float(np.std(grad_mag))

        # Calculate gradient direction histogram
        dir_hist, _ = np.histogram(grad_dir, bins=16, range=(-np.pi, np.pi))
        dir_hist = dir_hist / np.sum(dir_hist)

        # Calculate entropy of gradient directions
        dir_entropy = -np.sum(dir_hist * np.log2(dir_hist + 1e-10))

        # AI images often have more uniform gradient directions
        if dir_entropy > 3.5:  # Very high entropy (very uniform)
            gradient_score = 0.8
        elif dir_entropy > 3.0:  # High entropy
            gradient_score = 0.6
        elif mag_std < mag_mean * 0.5:  # Low gradient variation
            gradient_score = 0.7
        else:
            gradient_score = 0.3

        diagnostics = {
            "gradient_magnitude_mean": mag_mean,
            "gradient_magnitude_std": mag_std,
            "gradient_direction_entropy": float(dir_entropy),
        }

        return float(gradient_score), diagnostics

    except Exception as e:
        logger.debug(f"Gradient pattern analysis failed: {e}")
        return 0.5, {"error": str(e)}


def analyze_pixel_statistics(img: "Image.Image") -> MechanismResult:
    """
    Comprehensive pixel-level statistical analysis for AI detection.

    Analyzes:
    - Bit plane distributions
    - Histogram gaps and clustering
    - Channel correlations
    - Saturation patterns
    - Gradient distributions

    Returns MechanismResult with probability_ai and detailed diagnostics.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    if np is None:
        logger.warning("NumPy not available - pixel statistics analysis returning neutral score")
        return MechanismResult(
            mechanism=MechanismType.PIXEL_STATISTICS,
            probability_ai=0.5,
            notes="NumPy not installed - pixel statistics analysis skipped",
        )

    try:
        # Convert PIL image to numpy array
        img_array = np.asarray(img, dtype=np.float32)

        logger.debug("Analyzing bit planes...")
        bit_score, bit_diag = _analyze_bit_planes(img_array)

        logger.debug("Detecting histogram gaps...")
        gap_score, gap_diag = _detect_histogram_gaps(img_array)

        logger.debug("Analyzing value clustering...")
        cluster_score, cluster_diag = _analyze_value_clustering(img_array)

        logger.debug("Analyzing channel correlations...")
        corr_score, corr_diag = _analyze_channel_correlation(img_array)

        logger.debug("Analyzing saturation distribution...")
        sat_score, sat_diag = _analyze_saturation_distribution(img_array)

        logger.debug("Analyzing gradient patterns...")
        grad_score, grad_diag = _analyze_gradient_patterns(img_array)

        # Weighted combination
        weights = {
            "bit_planes": 0.20,
            "histogram_gaps": 0.15,
            "value_clustering": 0.20,
            "channel_correlation": 0.15,
            "saturation": 0.15,
            "gradients": 0.15,
        }

        combined_score = (
            weights["bit_planes"] * bit_score +
            weights["histogram_gaps"] * gap_score +
            weights["value_clustering"] * cluster_score +
            weights["channel_correlation"] * corr_score +
            weights["saturation"] * sat_score +
            weights["gradients"] * grad_score
        )

        # Build comprehensive diagnostics
        diagnostics = {
            "bit_plane_analysis": bit_diag,
            "histogram_gaps": gap_diag,
            "value_clustering": cluster_diag,
            "channel_correlations": corr_diag,
            "saturation_analysis": sat_diag,
            "gradient_patterns": grad_diag,
            "combined_score": float(combined_score),
        }

        # Generate notes
        notes_list = []
        if bit_score > 0.6:
            notes_list.append(f"Unusual bit plane patterns (score: {bit_score:.2f})")
        if gap_score > 0.6:
            notes_list.append(f"Histogram gaps detected (score: {gap_score:.2f})")
        if cluster_score > 0.6:
            notes_list.append(f"Unusual value clustering (score: {cluster_score:.2f})")
        if corr_score > 0.6:
            notes_list.append(f"Atypical channel correlations (score: {corr_score:.2f})")
        if sat_score > 0.6:
            notes_list.append(f"Unusual saturation patterns (score: {sat_score:.2f})")
        if grad_score > 0.6:
            notes_list.append(f"Unusual gradient patterns (score: {grad_score:.2f})")

        if not notes_list:
            notes_list.append("Pixel statistics within natural ranges")

        logger.debug(f"Pixel statistics analysis complete: AI probability = {combined_score:.3f}")

        return MechanismResult(
            mechanism=MechanismType.PIXEL_STATISTICS,
            probability_ai=float(combined_score),
            notes="; ".join(notes_list),
            extra=diagnostics,
        )

    except Exception as e:
        logger.error(f"Pixel statistics analysis failed: {e}", exc_info=True)
        return MechanismResult(
            mechanism=MechanismType.PIXEL_STATISTICS,
            probability_ai=0.5,
            notes=f"Pixel statistics analysis failed: {str(e)[:100]}",
        )
