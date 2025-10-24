from __future__ import annotations

import logging
import struct
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


def _analyze_jpeg_structure(file_path: str) -> Tuple[float, Dict[str, any]]:
    """
    Analyze JPEG file structure for AI generation signatures.
    AI tools often use different JPEG settings than cameras.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        # Look for JPEG markers
        markers = []
        pos = 0

        while pos < len(data) - 1:
            if data[pos] == 0xFF:
                marker = data[pos + 1]

                # SOI (Start of Image) - should be first
                if marker == 0xD8 and pos != 0:
                    markers.append(("SOI", pos))
                    break

                # APP segments (EXIF, etc.)
                elif 0xE0 <= marker <= 0xEF:
                    length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    markers.append(("APP", marker, pos, length))

                # Quantization tables
                elif marker == 0xDB:
                    length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    markers.append(("DQT", pos, length))

                # Start of Frame
                elif marker == 0xC0:
                    length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    markers.append(("SOF", pos, length))

                # Start of Scan
                elif marker == 0xDA:
                    length = struct.unpack('>H', data[pos+2:pos+4])[0]
                    markers.append(("SOS", pos, length))

                pos += 1
            else:
                pos += 1

        # Analyze marker patterns
        app_count = sum(1 for m in markers if m[0] == "APP")
        dqt_count = sum(1 for m in markers if m[0] == "DQT")

        # AI images often have fewer APP segments (less metadata)
        # But more DQT segments (custom quantization)
        if app_count == 0 and dqt_count > 2:
            ai_score = 0.8  # Likely AI-generated
        elif app_count == 0:
            ai_score = 0.7
        elif dqt_count > 2:
            ai_score = 0.6
        else:
            ai_score = 0.3

        diagnostics = {
            "total_markers": len(markers),
            "app_segments": app_count,
            "dqt_segments": dqt_count,
        }

        return float(ai_score), diagnostics

    except Exception as e:
        logger.debug(f"JPEG structure analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _analyze_quantization_tables(file_path: str) -> Tuple[float, Dict[str, any]]:
    """
    Analyze quantization tables for AI generation signatures.
    AI tools often use non-standard quantization settings.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        # Find all quantization tables (0xFFDB markers)
        qtables = []
        pos = 0

        while pos < len(data) - 1:
            if data[pos:pos+2] == b'\xFF\xDB':
                # Found quantization table
                length = struct.unpack('>H', data[pos+2:pos+4])[0]
                if length >= 67:  # Minimum size for quantization table
                    qtable_data = data[pos+4:pos+length]

                    # Extract quantization values (skip precision byte)
                    qvals = []
                    for i in range(1, min(65, len(qtable_data))):  # Skip precision byte
                        qvals.append(qtable_data[i])

                    if len(qvals) == 64:  # Standard 8x8 table
                        qtables.append(qvals)

                pos += length
            else:
                pos += 1

        if not qtables:
            return 0.5, {"error": "No quantization tables found"}

        # Analyze quantization patterns
        qtable_array = np.array(qtables)

        # Calculate statistics for each table
        qt_stats = []
        for qt in qtables:
            qt_np = np.array(qt)
            variance = float(np.var(qt_np))
            mean_val = float(np.mean(qt_np))
            std_val = float(np.std(qt_np))

            # AI tools often use more uniform quantization
            uniformity_score = 1.0 / (1.0 + variance / 1000.0)
            qt_stats.append({
                "variance": variance,
                "mean": mean_val,
                "std": std_val,
                "uniformity": uniformity_score,
            })

        # Average uniformity across tables
        avg_uniformity = float(np.mean([s["uniformity"] for s in qt_stats]))

        # High uniformity suggests AI generation
        if avg_uniformity > 0.8:
            ai_score = 0.8
        elif avg_uniformity > 0.6:
            ai_score = 0.6
        else:
            ai_score = 0.3

        diagnostics = {
            "num_qtables": len(qtables),
            "avg_uniformity": avg_uniformity,
            "qt_stats": qt_stats,
        }

        return float(ai_score), diagnostics

    except Exception as e:
        logger.debug(f"Quantization table analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _detect_double_compression(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect double JPEG compression artifacts.
    AI tools often save images that were already compressed.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Apply DCT to detect compression artifacts
        # Use block-based analysis for 8x8 blocks
        h, w = gray.shape
        block_size = 8

        # Pad image to be divisible by 8
        pad_h = (block_size - h % block_size) % block_size
        pad_w = (block_size - w % block_size) % block_size
        if pad_h > 0 or pad_w > 0:
            gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')

        # Extract 8x8 blocks
        blocks = []
        for i in range(0, gray.shape[0] - block_size + 1, block_size):
            for j in range(0, gray.shape[1] - block_size + 1, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                blocks.append(block)

        if not blocks:
            return 0.5, {"error": "No blocks extracted"}

        # Analyze each block for compression artifacts
        block_scores = []

        for block in blocks:
            # Calculate DCT coefficients
            dct_coeffs = np.fft.dct(np.fft.dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

            # Look for compression artifacts in high-frequency coefficients
            # Real cameras compress differently than double-compressed images
            hf_coeffs = dct_coeffs[4:, 4:]  # High-frequency region

            # Calculate statistics of high-frequency coefficients
            hf_mean = np.mean(np.abs(hf_coeffs))
            hf_variance = np.var(hf_coeffs)

            # Double-compressed images often have different HF patterns
            if hf_variance < 10:  # Very low high-frequency variance
                block_scores.append(0.8)
            elif hf_variance > 1000:  # Excessive high-frequency content
                block_scores.append(0.7)
            else:
                block_scores.append(0.3)

        # Average block scores
        avg_score = float(np.mean(block_scores))

        diagnostics = {
            "num_blocks": len(blocks),
            "avg_block_score": avg_score,
            "hf_variance": float(np.var([np.var(block[4:, 4:]) for block in blocks])),
        }

        return avg_score, diagnostics

    except Exception as e:
        logger.debug(f"Double compression analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _analyze_compression_history(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze compression history and artifacts.
    AI-generated images often show specific compression patterns.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}

    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array

        # Analyze pixel value distributions for compression artifacts
        # JPEG compression creates characteristic patterns

        # 1. Check for 8x8 blocking artifacts
        h, w = gray.shape
        block_h, block_w = 8, 8

        # Calculate block variance
        block_variances = []
        for i in range(0, h - block_h + 1, block_h):
            for j in range(0, w - block_w + 1, block_w):
                block = gray[i:i+block_h, j:j+block_w]
                block_variances.append(np.var(block))

        if not block_variances:
            return 0.5, {"error": "No blocks for analysis"}

        # 2. Analyze variance distribution
        block_var_array = np.array(block_variances)
        var_mean = np.mean(block_var_array)
        var_std = np.std(block_var_array)

        # AI images often have more uniform block variances
        if var_std < var_mean * 0.1:  # Very low variance in variances
            compression_score = 0.8
        elif var_std < var_mean * 0.2:
            compression_score = 0.6
        else:
            compression_score = 0.3

        # 3. Check for ringing artifacts (halo effects)
        # Apply edge detection and look for ringing patterns
        edges = np.gradient(gray)
        edge_magnitude = np.sqrt(edges[0]**2 + edges[1]**2)

        # Look for ringing patterns (oscillations near edges)
        edge_threshold = np.percentile(edge_magnitude, 95)
        strong_edges = edge_magnitude > edge_threshold

        if np.any(strong_edges):
            # Check for oscillatory patterns near strong edges
            # This is a simplified ringing detection
            edge_positions = np.where(strong_edges)
            if len(edge_positions[0]) > 0:
                # Sample a few edge locations
                sample_size = min(100, len(edge_positions[0]))
                indices = np.random.choice(len(edge_positions[0]), sample_size, replace=False)

                ringing_scores = []
                for idx in indices:
                    y, x = edge_positions[0][idx], edge_positions[1][idx]

                    # Look for oscillatory pattern in a small neighborhood
                    neighborhood = gray[max(0, y-2):min(h, y+3), max(0, x-2):min(w, x+3)]
                    if neighborhood.size > 4:
                        # Check for alternating pattern
                        flat_neighborhood = neighborhood.flatten()
                        oscillations = np.abs(np.diff(flat_neighborhood))
                        avg_oscillation = np.mean(oscillations)

                        if avg_oscillation > 20:  # High oscillation suggests ringing
                            ringing_scores.append(0.8)
                        else:
                            ringing_scores.append(0.3)

                if ringing_scores:
                    ringing_score = float(np.mean(ringing_scores))
                    compression_score = max(compression_score, ringing_score)

        diagnostics = {
            "block_variance_std": float(var_std),
            "block_variance_mean": float(var_mean),
            "compression_score": compression_score,
            "num_blocks": len(block_variances),
        }

        return float(compression_score), diagnostics

    except Exception as e:
        logger.debug(f"Compression history analysis failed: {e}")
        return 0.5, {"error": str(e)}


def analyze_compression(img: "Image.Image", file_path: str) -> MechanismResult:
    """
    Comprehensive JPEG compression analysis for AI detection.

    Analyzes:
    - JPEG file structure and markers
    - Quantization table patterns
    - Double compression artifacts
    - Compression history signatures
    - Blocking and ringing artifacts

    Returns MechanismResult with probability_ai and detailed diagnostics.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    if np is None:
        logger.warning("NumPy not available - compression analysis returning neutral score")
        return MechanismResult(
            mechanism=MechanismType.COMPRESSION,
            probability_ai=0.5,
            notes="NumPy not installed - compression analysis skipped",
        )

    try:
        # Convert PIL image to numpy array
        img_array = np.asarray(img, dtype=np.float32)

        logger.debug("Analyzing JPEG file structure...")
        structure_score, structure_diag = _analyze_jpeg_structure(file_path)

        logger.debug("Analyzing quantization tables...")
        quant_score, quant_diag = _analyze_quantization_tables(file_path)

        logger.debug("Detecting double compression...")
        double_comp_score, double_comp_diag = _detect_double_compression(img_array)

        logger.debug("Analyzing compression history...")
        history_score, history_diag = _analyze_compression_history(img_array)

        # Weighted combination
        weights = {
            "structure": 0.25,
            "quantization": 0.30,
            "double_compression": 0.25,
            "history": 0.20,
        }

        combined_score = (
            weights["structure"] * structure_score +
            weights["quantization"] * quant_score +
            weights["double_compression"] * double_comp_score +
            weights["history"] * history_score
        )

        # Build comprehensive diagnostics
        diagnostics = {
            "jpeg_structure": structure_diag,
            "quantization_tables": quant_diag,
            "double_compression": double_comp_diag,
            "compression_history": history_diag,
            "combined_score": float(combined_score),
        }

        # Generate notes
        notes_list = []
        if structure_score > 0.6:
            notes_list.append(f"Suspicious JPEG structure (score: {structure_score:.2f})")
        if quant_score > 0.6:
            notes_list.append(f"Unusual quantization patterns (score: {quant_score:.2f})")
        if double_comp_score > 0.6:
            notes_list.append(f"Double compression detected (score: {double_comp_score:.2f})")
        if history_score > 0.6:
            notes_list.append(f"Compression artifacts found (score: {history_score:.2f})")

        if not notes_list:
            notes_list.append("Compression analysis shows natural patterns")

        logger.debug(f"Compression analysis complete: AI probability = {combined_score:.3f}")

        return MechanismResult(
            mechanism=MechanismType.COMPRESSION,
            probability_ai=float(combined_score),
            notes="; ".join(notes_list),
            extra=diagnostics,
        )

    except Exception as e:
        logger.error(f"Compression analysis failed: {e}", exc_info=True)
        return MechanismResult(
            mechanism=MechanismType.COMPRESSION,
            probability_ai=0.5,
            notes=f"Compression analysis failed: {str(e)[:100]}",
        )
