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

try:
    from scipy import ndimage
except Exception:  # pragma: no cover
    ndimage = None

logger = logging.getLogger(__name__)


def _measure_sharpness(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Measure image sharpness using Laplacian variance.
    AI images are often suspiciously sharp or uniformly blurred.
    """
    if np is None:
        return 0.5, {}
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    # Laplacian kernel
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    # Convolve
    if ndimage is not None:
        laplacian_img = ndimage.convolve(gray, laplacian)
    else:
        # Simple manual convolution if scipy unavailable
        laplacian_img = gray  # Fallback
    
    # Variance of Laplacian
    sharpness = float(np.var(laplacian_img))
    
    # Normalize to reasonable range (typical values 0-1000)
    normalized_sharpness = sharpness / 1000.0
    
    diagnostics = {
        "sharpness_variance": sharpness,
        "normalized": float(np.clip(normalized_sharpness, 0, 1)),
    }
    
    # Extremely high or low sharpness is suspicious
    # Expanded normal range for modern smartphones (0.1-0.9)
    if normalized_sharpness > 0.9:
        suspicion = 0.4  # Very sharp - reduced from 0.6 (phones are sharp)
    elif normalized_sharpness < 0.1:
        suspicion = 0.7  # Too blurry
    else:
        suspicion = 0.3  # Normal range (now broader)
    
    return float(suspicion), diagnostics


def _detect_chromatic_aberration(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect chromatic aberration (color fringing) at edges.
    Real lenses have it; AI-generated images often don\'t.
    """
    if np is None or len(img_array.shape) != 3:
        return 0.5, {}
    
    # Extract RGB channels
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Compute edge differences between channels
    # Real lenses show color misalignment at edges
    rg_diff = np.abs(r.astype(float) - g.astype(float))
    rb_diff = np.abs(r.astype(float) - b.astype(float))
    
    # Focus on high-gradient areas (edges)
    if ndimage is not None:
        edges = ndimage.sobel(g.astype(float))
        edge_mask = edges > np.percentile(edges, 90)
    else:
        # Simple edge detection fallback
        edge_mask = np.ones_like(g, dtype=bool)
    
    # Measure chromatic aberration at edges
    ca_rg = float(np.mean(rg_diff[edge_mask]))
    ca_rb = float(np.mean(rb_diff[edge_mask]))
    ca_score = (ca_rg + ca_rb) / 2.0
    
    diagnostics = {
        "chromatic_aberration_rg": ca_rg,
        "chromatic_aberration_rb": ca_rb,
        "ca_score": ca_score,
    }
    
    # Presence of CA suggests real camera = low AI probability
    # BUT modern phones correct CA digitally, so absence is not suspicious
    # Normalize to 0-1 range (typical CA values 0-10)
    normalized_ca = np.clip(ca_score / 10.0, 0, 1)
    
    # If CA is very low (phones correct it), return neutral instead of suspicious
    if normalized_ca < 0.2:
        ai_probability = 0.5  # Neutral - phones correct CA
    else:
        ai_probability = 1.0 - normalized_ca
    
    return float(ai_probability), diagnostics


def _analyze_vignetting(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect vignetting (edge darkening) pattern.
    Real camera lenses show natural vignetting; AI often doesn\'t.
    """
    if np is None:
        return 0.5, {}
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    h, w = gray.shape
    center_h, center_w = h // 2, w // 2
    
    # Compare center brightness to edge brightness
    center_region = gray[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8]
    center_brightness = float(np.mean(center_region))
    
    # Edge regions (top, bottom, left, right strips)
    edge_width = min(h, w) // 20
    edges = np.concatenate([
        gray[:edge_width, :].flatten(),
        gray[-edge_width:, :].flatten(),
        gray[:, :edge_width].flatten(),
        gray[:, -edge_width:].flatten(),
    ])
    edge_brightness = float(np.mean(edges))
    
    vignetting_ratio = center_brightness / (edge_brightness + 1e-10)
    
    diagnostics = {
        "center_brightness": center_brightness,
        "edge_brightness": edge_brightness,
        "vignetting_ratio": float(vignetting_ratio),
    }
    
    # Vignetting ratio > 1.0 suggests natural lens darkening
    # BUT modern phones correct vignetting digitally, so absence is neutral
    if vignetting_ratio > 1.1:
        ai_probability = 0.3  # Natural vignetting present
    elif vignetting_ratio > 1.05:
        ai_probability = 0.4  # Mild vignetting
    else:
        ai_probability = 0.5  # No vignetting - neutral (phones correct it)
    
    return float(ai_probability), diagnostics


def _check_color_distribution(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Validate natural color histogram patterns.
    AI images sometimes have unnatural color distributions.
    """
    if np is None or len(img_array.shape) != 3:
        return 0.5, {}
    
    # Compute per-channel histograms
    r_hist, _ = np.histogram(img_array[:,:,0], bins=256, range=(0, 256))
    g_hist, _ = np.histogram(img_array[:,:,1], bins=256, range=(0, 256))
    b_hist, _ = np.histogram(img_array[:,:,2], bins=256, range=(0, 256))
    
    # Normalize
    r_hist = r_hist.astype(float) / np.sum(r_hist)
    g_hist = g_hist.astype(float) / np.sum(g_hist)
    b_hist = b_hist.astype(float) / np.sum(b_hist)
    
    # Check for unnatural peaks or gaps
    # AI images sometimes have spiky histograms
    r_spikiness = float(np.max(r_hist))
    g_spikiness = float(np.max(g_hist))
    b_spikiness = float(np.max(b_hist))
    avg_spikiness = (r_spikiness + g_spikiness + b_spikiness) / 3.0
    
    diagnostics = {
        "r_max_bin": r_spikiness,
        "g_max_bin": g_spikiness,
        "b_max_bin": b_spikiness,
        "avg_spikiness": float(avg_spikiness),
    }
    
    # Very spiky histogram is suspicious
    # Adjusted for vibrant smartphone processing
    if avg_spikiness > 0.15:  # Increased from 0.1
        ai_probability = 0.7
    elif avg_spikiness > 0.08:  # Increased from 0.05
        ai_probability = 0.5
    else:
        ai_probability = 0.3
    
    return float(ai_probability), diagnostics


def _edge_sharpness_uniformity(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect suspiciously uniform sharpness across the image.
    Real photos have varying sharpness (depth of field); AI is often uniform.
    """
    if np is None or ndimage is None:
        return 0.5, {}
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    # Compute Laplacian (edge strength) across the image
    laplacian = ndimage.laplace(gray.astype(float))
    
    # Divide into regions and measure sharpness variance
    h, w = gray.shape
    grid_size = 4
    region_h, region_w = h // grid_size, w // grid_size
    
    regional_sharpness = []
    for i in range(grid_size):
        for j in range(grid_size):
            y_start, x_start = i * region_h, j * region_w
            y_end = (i + 1) * region_h if i < grid_size - 1 else h
            x_end = (j + 1) * region_w if j < grid_size - 1 else w
            
            region = laplacian[y_start:y_end, x_start:x_end]
            sharpness = np.var(region)
            regional_sharpness.append(sharpness)
    
    # Compute variance of regional sharpness
    sharpness_variance = float(np.var(regional_sharpness))
    sharpness_mean = float(np.mean(regional_sharpness))
    
    # Coefficient of variation
    cv = sharpness_variance / (sharpness_mean + 1e-10)
    
    diagnostics = {
        "regional_sharpness_variance": sharpness_variance,
        "regional_sharpness_mean": sharpness_mean,
        "coefficient_of_variation": float(cv),
    }
    
    # Low variance = too uniform = suspicious
    # BUT allow more uniformity - phones use focus stacking, computational DOF
    if cv < 0.05:  # Stricter threshold - was 0.1
        ai_probability = 0.8  # Very uniform
    elif cv < 0.2:  # More lenient - was 0.3
        ai_probability = 0.5  # Moderately uniform - neutral for phones
    else:
        ai_probability = 0.3  # Natural variation
    
    return float(ai_probability), diagnostics


def _detect_oversaturation(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Flag unnatural color intensity and saturation patterns.
    AI images sometimes have unrealistic color saturation.
    """
    if np is None or len(img_array.shape) != 3:
        return 0.5, {}
    
    # Count pixels at extreme values (0 or 255)
    clipped_pixels = np.sum((img_array == 0) | (img_array == 255))
    total_pixels = img_array.size
    clipped_ratio = clipped_pixels / total_pixels
    
    # Check saturation (distance from grayscale)
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    gray_approx = (r.astype(float) + g.astype(float) + b.astype(float)) / 3.0
    
    saturation = np.mean([
        np.abs(r - gray_approx),
        np.abs(g - gray_approx),
        np.abs(b - gray_approx),
    ])
    
    diagnostics = {
        "clipped_ratio": float(clipped_ratio),
        "avg_saturation": float(saturation),
    }
    
    # High clipping or extreme saturation is suspicious
    # Adjusted for vibrant smartphone processing
    if clipped_ratio > 0.08 or saturation > 70:  # Increased thresholds
        ai_probability = 0.7
    elif clipped_ratio > 0.04 or saturation > 50:  # More lenient
        ai_probability = 0.5
    else:
        ai_probability = 0.3
    
    return float(ai_probability), diagnostics


def _analyze_sensor_noise(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze sensor noise patterns (PRNU - Photo Response Non-Uniformity).
    Real cameras have characteristic noise patterns; AI images lack them.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}
    
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        # Apply high-pass filter to extract noise
        # Use a simple difference filter
        noise = gray - ndimage.gaussian_filter(gray, sigma=1.0) if ndimage else gray - gray
        
        # Analyze noise characteristics
        noise_std = float(np.std(noise))
        noise_mean = float(np.mean(noise))
        noise_skewness = float(_calculate_skewness(noise))
        noise_kurtosis = float(_calculate_kurtosis(noise))
        
        # Real camera noise typically has specific statistical properties
        # AI images often lack natural noise patterns
        if noise_std < 5:  # Very low noise
            noise_score = 0.8
        elif noise_std > 50:  # Excessive noise
            noise_score = 0.7
        elif abs(noise_skewness) > 2:  # Unusual skewness
            noise_score = 0.6
        elif abs(noise_kurtosis) > 5:  # Unusual kurtosis
            noise_score = 0.6
        else:
            noise_score = 0.3
        
        diagnostics = {
            "noise_std": noise_std,
            "noise_mean": noise_mean,
            "noise_skewness": noise_skewness,
            "noise_kurtosis": noise_kurtosis,
        }
        
        return float(noise_score), diagnostics
        
    except Exception as e:
        logger.debug(f"Sensor noise analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _detect_lens_distortion(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect lens distortion patterns.
    Real cameras have lens distortion; AI images often lack it.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}
    
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        
        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        
        # Calculate distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = distance / max_distance
        
        # Analyze radial distortion patterns
        # Real lenses typically have barrel or pincushion distortion
        # AI images often lack these natural distortions
        
        # Sample points at different radii
        radii = [0.2, 0.4, 0.6, 0.8]
        distortion_scores = []
        
        for radius in radii:
            mask = (normalized_distance >= radius - 0.1) & (normalized_distance <= radius + 0.1)
            if np.any(mask):
                # Analyze gradient patterns in this radial band
                band_gradients = np.gradient(gray[mask])
                gradient_variance = np.var(band_gradients)
                distortion_scores.append(gradient_variance)
        
        if not distortion_scores:
            return 0.5, {"error": "No valid radial bands found"}
        
        # AI images often have more uniform distortion patterns
        distortion_variance = float(np.var(distortion_scores))
        distortion_mean = float(np.mean(distortion_scores))
        
        # Very low variance suggests AI generation
        if distortion_variance < 10:
            distortion_score = 0.8
        elif distortion_variance < 50:
            distortion_score = 0.6
        else:
            distortion_score = 0.3
        
        diagnostics = {
            "distortion_variance": distortion_variance,
            "distortion_mean": distortion_mean,
            "num_radial_bands": len(distortion_scores),
        }
        
        return float(distortion_score), diagnostics
        
    except Exception as e:
        logger.debug(f"Lens distortion analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _detect_banding_artifacts(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Detect banding artifacts common in AI-generated images.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}
    
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        # Analyze horizontal and vertical gradients for banding
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        
        # Look for regular patterns in gradients
        # Banding creates periodic patterns in gradients
        
        # Analyze horizontal banding
        h_gradients = np.mean(grad_x, axis=0)  # Average gradient per column
        h_fft = np.fft.fft(h_gradients)
        h_power = np.abs(h_fft[1:len(h_fft)//2])  # Skip DC component
        
        # Analyze vertical banding
        v_gradients = np.mean(grad_y, axis=1)  # Average gradient per row
        v_fft = np.fft.fft(v_gradients)
        v_power = np.abs(v_fft[1:len(v_fft)//2])  # Skip DC component
        
        # Look for strong periodic components
        h_max_power = float(np.max(h_power))
        v_max_power = float(np.max(v_power))
        
        # Calculate banding score
        banding_score = 0.5
        
        if h_max_power > 100:  # Strong horizontal banding
            banding_score += 0.3
        if v_max_power > 100:  # Strong vertical banding
            banding_score += 0.3
        
        # Check for color banding in RGB channels
        if len(img_array.shape) == 3:
            for channel in range(3):
                channel_data = img_array[:, :, channel]
                channel_grad = np.gradient(channel_data, axis=1)
                channel_fft = np.fft.fft(np.mean(channel_grad, axis=0))
                channel_power = np.max(np.abs(channel_fft[1:len(channel_fft)//2]))
                
                if channel_power > 50:
                    banding_score += 0.1
        
        banding_score = min(banding_score, 1.0)
        
        diagnostics = {
            "horizontal_banding_power": h_max_power,
            "vertical_banding_power": v_max_power,
            "banding_score": banding_score,
        }
        
        return float(banding_score), diagnostics
        
    except Exception as e:
        logger.debug(f"Banding artifact analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _detect_micro_contrast(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze micro-contrast patterns.
    Real photos have natural micro-contrast; AI images may lack it.
    """
    if np is None:
        return 0.5, {"error": "NumPy not available"}
    
    try:
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        # Apply local contrast enhancement
        # Use a small kernel to detect micro-contrast
        kernel_size = 3
        local_mean = ndimage.uniform_filter(gray, size=kernel_size) if ndimage else gray
        local_contrast = gray - local_mean
        
        # Analyze local contrast patterns
        contrast_std = float(np.std(local_contrast))
        contrast_mean = float(np.mean(np.abs(local_contrast)))
        
        # Real photos typically have varied micro-contrast
        # AI images often have more uniform micro-contrast
        if contrast_std < 5:  # Very low contrast variation
            micro_contrast_score = 0.8
        elif contrast_std > 50:  # Excessive contrast variation
            micro_contrast_score = 0.7
        elif contrast_mean < 2:  # Very low average contrast
            micro_contrast_score = 0.6
        else:
            micro_contrast_score = 0.3
        
        diagnostics = {
            "contrast_std": contrast_std,
            "contrast_mean": contrast_mean,
        }
        
        return float(micro_contrast_score), diagnostics
        
    except Exception as e:
        logger.debug(f"Micro-contrast analysis failed: {e}")
        return 0.5, {"error": str(e)}


def _calculate_skewness(data: "np.ndarray") -> float:
    """Calculate skewness of data."""
    if np is None or len(data) == 0:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    skewness = np.mean(((data - mean) / std) ** 3)
    return float(skewness)


def _calculate_kurtosis(data: "np.ndarray") -> float:
    """Calculate kurtosis of data."""
    if np is None or len(data) == 0:
        return 0.0
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    
    kurtosis = np.mean(((data - mean) / std) ** 4) - 3
    return float(kurtosis)


def analyze_quality_metrics(img: "Image.Image", img_array: Optional["np.ndarray"] = None) -> MechanismResult:
    """
    Analyze image quality metrics to detect AI-generated characteristics.
    
    Analyzes:
    - Sharpness (Laplacian variance)
    - Chromatic aberration (color fringing at edges)
    - Vignetting (natural edge darkening)
    - Color distribution patterns
    - Edge sharpness uniformity
    - Oversaturation detection
    
    Real camera photos have characteristic imperfections and variations.
    AI-generated images often lack these natural artifacts or are too perfect.
    
    Returns MechanismResult with probability_ai and detailed diagnostics.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    
    # Graceful fallback if numpy/scipy not available
    if np is None:
        logger.warning("numpy not available - quality metrics returning neutral score")
        return MechanismResult(
            mechanism=MechanismType.QUALITY_METRICS,
            probability_ai=0.5,
            notes="numpy not installed - quality metrics skipped",
        )
    
    try:
        # Convert to numpy if not provided
        if img_array is None:
            img_array = np.asarray(img, dtype=np.uint8)
        
        logger.debug("Measuring image sharpness...")
        sharpness_score, sharpness_diag = _measure_sharpness(img_array)
        
        logger.debug("Detecting chromatic aberration...")
        ca_score, ca_diag = _detect_chromatic_aberration(img_array)
        
        logger.debug("Analyzing vignetting pattern...")
        vignetting_score, vignetting_diag = _analyze_vignetting(img_array)
        
        logger.debug("Checking color distribution...")
        color_score, color_diag = _check_color_distribution(img_array)
        
        logger.debug("Measuring edge sharpness uniformity...")
        uniformity_score, uniformity_diag = _edge_sharpness_uniformity(img_array)
        
        logger.debug("Detecting oversaturation...")
        saturation_score, saturation_diag = _detect_oversaturation(img_array)
        
        # New enhanced quality analyses
        logger.debug("Analyzing sensor noise patterns...")
        noise_score, noise_diag = _analyze_sensor_noise(img_array)
        
        logger.debug("Detecting lens distortion...")
        distortion_score, distortion_diag = _detect_lens_distortion(img_array)
        
        logger.debug("Detecting banding artifacts...")
        banding_score, banding_diag = _detect_banding_artifacts(img_array)
        
        logger.debug("Analyzing micro-contrast...")
        micro_contrast_score, micro_contrast_diag = _detect_micro_contrast(img_array)
        
        # Weighted combination with updated weights
        weights = {
            "sharpness": 0.12,
            "chromatic_aberration": 0.15,
            "vignetting": 0.15,
            "color_distribution": 0.12,
            "uniformity": 0.15,
            "saturation": 0.08,
            "noise": 0.10,
            "distortion": 0.08,
            "banding": 0.03,
            "micro_contrast": 0.02,
        }
        
        combined_score = (
            weights["sharpness"] * sharpness_score +
            weights["chromatic_aberration"] * ca_score +
            weights["vignetting"] * vignetting_score +
            weights["color_distribution"] * color_score +
            weights["uniformity"] * uniformity_score +
            weights["saturation"] * saturation_score +
            weights["noise"] * noise_score +
            weights["distortion"] * distortion_score +
            weights["banding"] * banding_score +
            weights["micro_contrast"] * micro_contrast_score
        )
        
        # Build diagnostics
        diagnostics = {
            "sharpness": {**sharpness_diag, "score": sharpness_score},
            "chromatic_aberration": {**ca_diag, "score": ca_score},
            "vignetting": {**vignetting_diag, "score": vignetting_score},
            "color_distribution": {**color_diag, "score": color_score},
            "edge_uniformity": {**uniformity_diag, "score": uniformity_score},
            "saturation": {**saturation_diag, "score": saturation_score},
            "sensor_noise": {**noise_diag, "score": noise_score},
            "lens_distortion": {**distortion_diag, "score": distortion_score},
            "banding_artifacts": {**banding_diag, "score": banding_score},
            "micro_contrast": {**micro_contrast_diag, "score": micro_contrast_score},
            "combined_score": float(combined_score),
        }
        
        # Generate notes
        notes_list = []
        if sharpness_score > 0.6:
            notes_list.append(f"Unusual sharpness pattern (score: {sharpness_score:.2f})")
        if ca_score > 0.7:
            notes_list.append(f"Minimal chromatic aberration (score: {ca_score:.2f})")
        if vignetting_score > 0.6:
            notes_list.append(f"Absent/weak vignetting (score: {vignetting_score:.2f})")
        if uniformity_score > 0.6:
            notes_list.append(f"Unnaturally uniform sharpness (score: {uniformity_score:.2f})")
        if saturation_score > 0.6:
            notes_list.append(f"Oversaturation detected (score: {saturation_score:.2f})")
        if noise_score > 0.6:
            notes_list.append(f"Unusual noise patterns (score: {noise_score:.2f})")
        if distortion_score > 0.6:
            notes_list.append(f"Lack of natural lens distortion (score: {distortion_score:.2f})")
        if banding_score > 0.6:
            notes_list.append(f"Banding artifacts detected (score: {banding_score:.2f})")
        if micro_contrast_score > 0.6:
            notes_list.append(f"Unusual micro-contrast patterns (score: {micro_contrast_score:.2f})")
        
        if not notes_list:
            notes_list.append("Quality metrics within natural ranges")
        
        logger.debug(f"Quality analysis complete: AI probability = {combined_score:.3f}")
        
        return MechanismResult(
            mechanism=MechanismType.QUALITY_METRICS,
            probability_ai=float(combined_score),
            notes="; ".join(notes_list[:3]),  # First 3 findings
            extra=diagnostics,
        )
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}", exc_info=True)
        return MechanismResult(
            mechanism=MechanismType.QUALITY_METRICS,
            probability_ai=0.5,
            notes=f"Quality analysis encountered an error: {str(e)[:100]}",
        )
