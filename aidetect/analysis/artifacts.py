from __future__ import annotations

import logging
from typing import Dict, List, Tuple

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


def _detect_checkerboard_artifacts(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect checkerboard upsampling artifacts from GANs.
    """
    if np is None or len(img_array.shape) < 2:
        return 0.5, {"error": "Invalid input"}
    
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        # Apply 2D FFT to detect periodic patterns
        fft_result = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft_result)
        magnitude = np.abs(fft_shift)
        
        # Look for peaks at specific frequencies indicating checkerboard
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Check for peaks at Nyquist frequency (checkerboard signature)
        nyquist_h = magnitude[center_h + h//4, center_w]
        nyquist_v = magnitude[center_h, center_w + w//4]
        
        avg_magnitude = np.mean(magnitude)
        
        # High Nyquist peaks indicate checkerboard artifacts
        if nyquist_h > avg_magnitude * 5 or nyquist_v > avg_magnitude * 5:
            score = 0.8
        elif nyquist_h > avg_magnitude * 3 or nyquist_v > avg_magnitude * 3:
            score = 0.6
        else:
            score = 0.3
        
        diagnostics = {
            "nyquist_h": float(nyquist_h),
            "nyquist_v": float(nyquist_v),
            "avg_magnitude": float(avg_magnitude),
        }
        
        return float(score), diagnostics
        
    except Exception as e:
        logger.debug(f"Checkerboard detection failed: {e}")
        return 0.5, {"error": str(e)}


def _detect_gan_fingerprints(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect GAN-specific fingerprints in pixel patterns.
    """
    if np is None or len(img_array.shape) < 2:
        return 0.5, {"error": "Invalid input"}
    
    try:
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        # Analyze boundary regions (GANs often have artifacts at edges)
        h, w = gray.shape
        border_size = min(h, w) // 20  # 5% border
        
        if border_size < 2:
            return 0.5, {"error": "Image too small"}
        
        # Extract borders
        top = gray[:border_size, :]
        bottom = gray[-border_size:, :]
        left = gray[:, :border_size]
        right = gray[:, -border_size:]
        
        # Calculate statistics for borders vs center
        center = gray[border_size:-border_size, border_size:-border_size]
        
        border_std = float(np.std([np.std(top), np.std(bottom), np.std(left), np.std(right)]))
        center_std = float(np.std(center))
        
        # GANs often have different statistics at boundaries
        if border_std > center_std * 2:  # High border variance
            score = 0.7
        elif border_std > center_std * 1.5:
            score = 0.6
        else:
            score = 0.3
        
        diagnostics = {
            "border_std": border_std,
            "center_std": center_std,
            "ratio": border_std / (center_std + 1e-10),
        }
        
        return float(score), diagnostics
        
    except Exception as e:
        logger.debug(f"GAN fingerprint detection failed: {e}")
        return 0.5, {"error": str(e)}


def _detect_diffusion_patterns(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect diffusion model noise patterns.
    """
    if np is None or len(img_array.shape) < 2:
        return 0.5, {"error": "Invalid input"}
    
    try:
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        # Analyze high-frequency noise typical of diffusion models
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Diffusion models often have specific noise characteristics
        noise_std = float(np.std(gradient_mag))
        noise_mean = float(np.mean(gradient_mag))
        
        # Analyze noise distribution
        hist, _ = np.histogram(gradient_mag, bins=50)
        hist_normalized = hist / np.sum(hist)
        entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
        
        # High entropy in gradient distribution suggests diffusion artifacts
        if entropy > 3.5:
            score = 0.7
        elif entropy > 3.0:
            score = 0.6
        else:
            score = 0.3
        
        diagnostics = {
            "noise_std": noise_std,
            "noise_mean": noise_mean,
            "gradient_entropy": float(entropy),
        }
        
        return float(score), diagnostics
        
    except Exception as e:
        logger.debug(f"Diffusion pattern detection failed: {e}")
        return 0.5, {"error": str(e)}


def analyze_artifacts(img: "Image.Image") -> MechanismResult:
    """
    Detect AI generation artifacts including GAN upsampling, diffusion noise,
    and checkerboard patterns characteristic of neural network generation.
    
    Analyzes:
    - Checkerboard upsampling artifacts (common in GANs)
    - GAN boundary fingerprints
    - Diffusion model noise patterns
    
    Returns MechanismResult with probability_ai and detailed diagnostics.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    
    if np is None:
        logger.warning("NumPy not available - artifact analysis returning neutral score")
        return MechanismResult(
            mechanism=MechanismType.ARTIFACTS,
            probability_ai=0.5,
            notes="NumPy not installed - artifact analysis skipped",
        )
    
    try:
        # Convert to numpy array
        img_array = np.asarray(img, dtype=np.float32)
        
        logger.debug("Detecting checkerboard artifacts...")
        checkerboard_score, checkerboard_diag = _detect_checkerboard_artifacts(img_array)
        
        logger.debug("Detecting GAN fingerprints...")
        gan_score, gan_diag = _detect_gan_fingerprints(img_array)
        
        logger.debug("Detecting diffusion patterns...")
        diffusion_score, diffusion_diag = _detect_diffusion_patterns(img_array)
        
        # Weighted combination
        weights = {
            "checkerboard": 0.40,
            "gan_fingerprints": 0.35,
            "diffusion": 0.25,
        }
        
        combined_score = (
            weights["checkerboard"] * checkerboard_score +
            weights["gan_fingerprints"] * gan_score +
            weights["diffusion"] * diffusion_score
        )
        
        # Build diagnostics
        diagnostics = {
            "checkerboard_analysis": checkerboard_diag,
            "gan_fingerprint_analysis": gan_diag,
            "diffusion_pattern_analysis": diffusion_diag,
            "combined_score": float(combined_score),
        }
        
        # Generate notes
        notes_list = []
        if checkerboard_score > 0.6:
            notes_list.append(f"Checkerboard upsampling artifacts detected (score: {checkerboard_score:.2f})")
        if gan_score > 0.6:
            notes_list.append(f"GAN boundary fingerprints found (score: {gan_score:.2f})")
        if diffusion_score > 0.6:
            notes_list.append(f"Diffusion model noise patterns detected (score: {diffusion_score:.2f})")
        
        if not notes_list:
            notes_list.append("No significant AI generation artifacts detected")
        
        logger.debug(f"Artifact analysis complete: AI probability = {combined_score:.3f}")
        
        return MechanismResult(
            mechanism=MechanismType.ARTIFACTS,
            probability_ai=float(combined_score),
            notes="; ".join(notes_list),
            extra=diagnostics,
        )
        
    except Exception as e:
        logger.error(f"Artifact analysis failed: {e}", exc_info=True)
        return MechanismResult(
            mechanism=MechanismType.ARTIFACTS,
            probability_ai=0.5,
            notes=f"Artifact analysis failed: {str(e)[:100]}",
        )
