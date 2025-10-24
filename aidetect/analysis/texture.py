from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

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
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    from skimage.filters import gabor
except Exception:  # pragma: no cover
    local_binary_pattern = None
    graycomatrix = None
    graycoprops = None
    gabor = None

logger = logging.getLogger(__name__)


def _compute_lbp(gray_img: "np.ndarray", radius: int = 1, n_points: int = 8) -> Optional["np.ndarray"]:
    """
    Compute Local Binary Pattern for a grayscale image.
    Uses uniform patterns for rotation invariance.
    """
    if local_binary_pattern is None or np is None:
        return None
    
    try:
        lbp = local_binary_pattern(gray_img, n_points, radius, method="uniform")
        return lbp
    except Exception as e:
        logger.warning(f"LBP computation failed: {e}")
        return None


def _multiscale_lbp(gray_img: "np.ndarray") -> List["np.ndarray"]:
    """
    Extract LBP at multiple scales (radius 1, 2, 3).
    Multi-scale analysis captures texture at different granularities.
    """
    scales = [(1, 8), (2, 16), (3, 24)]  # (radius, n_points)
    lbp_images = []
    
    for radius, n_points in scales:
        lbp = _compute_lbp(gray_img, radius, n_points)
        if lbp is not None and len(lbp) > 0:
            lbp_images.append(lbp)
    
    return lbp_images


def _lbp_histogram(lbp_image: "np.ndarray", n_bins: int = 26) -> "np.ndarray":
    """
    Create normalized histogram of LBP values.
    Uniform LBP with 8 points has 59 patterns, but we use binning.
    """
    if lbp_image is None or len(lbp_image) == 0 or np is None:
        return np.array([])
    
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    # Normalize
    hist_normalized = hist.astype(float) / (np.sum(hist) + 1e-10)
    
    return hist_normalized


def _regional_lbp_analysis(gray_img: "np.ndarray", grid_size: int = 4) -> List["np.ndarray"]:
    """
    Divide image into grid and compute LBP histogram for each region.
    This captures spatial texture variation.
    """
    if np is None or len(gray_img) == 0:
        return []
    
    h, w = gray_img.shape
    region_h = h // grid_size
    region_w = w // grid_size
    
    regional_histograms = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            y_start = i * region_h
            y_end = (i + 1) * region_h if i < grid_size - 1 else h
            x_start = j * region_w
            x_end = (j + 1) * region_w if j < grid_size - 1 else w
            
            region = gray_img[y_start:y_end, x_start:x_end]
            lbp = _compute_lbp(region)
            
            if lbp is not None and len(lbp) > 0:
                hist = _lbp_histogram(lbp)
                regional_histograms.append(hist)
    
    return regional_histograms


def _texture_consistency_score(regional_histograms: List["np.ndarray"]) -> float:
    """
    Measure texture consistency across regions.
    AI images sometimes have unnaturally uniform textures.
    """
    if len(regional_histograms) < 2 or np is None:
        return 0.5
    
    # Compute pairwise histogram distances
    distances = []
    for i in range(len(regional_histograms)):
        for j in range(i + 1, len(regional_histograms)):
            # Chi-square distance
            h1 = regional_histograms[i]
            h2 = regional_histograms[j]
            distance = np.sum((h1 - h2) ** 2 / (h1 + h2 + 1e-10))
            distances.append(distance)
    
    if len(distances) == 0:
        return 0.5
    
    # Low variance = too consistent = suspicious
    variance = float(np.var(distances))
    mean_dist = float(np.mean(distances))
    
    # If textures are too similar across regions, suspicious
    # Natural photos have varying textures
    # Reduced multiplier from 10 to 5 to be less sensitive to uniformity
    consistency_score = 1.0 / (1.0 + mean_dist * 5)
    
    return float(np.clip(consistency_score, 0, 1))


def _detect_repetitive_patterns(lbp_histograms: List["np.ndarray"]) -> float:
    """
    Detect repetitive patterns characteristic of GANs.
    GANs sometimes create subtle repeating textures.
    """
    if len(lbp_histograms) < 4 or np is None:
        return 0.5
    
    # Check for near-identical histograms (sign of repetition)
    identical_pairs = 0
    total_pairs = 0
    
    threshold = 0.05  # Stricter similarity threshold (was 0.1) - require very high correlation
    
    for i in range(len(lbp_histograms)):
        for j in range(i + 1, len(lbp_histograms)):
            h1 = lbp_histograms[i]
            h2 = lbp_histograms[j]
            
            # Correlation distance
            correlation = np.corrcoef(h1, h2)[0, 1] if len(h1) > 0 and len(h2) > 0 else 0
            
            if correlation > (1.0 - threshold):
                identical_pairs += 1
            total_pairs += 1
    
    if total_pairs == 0:
        return 0.5
    
    repetition_ratio = identical_pairs / total_pairs
    
    # High repetition = more AI-like
    # Increased multiplier from 2.0 to 3.0 - require stronger evidence
    return float(np.clip(repetition_ratio * 3.0, 0, 1))


def _compare_lbp_distributions(hist1: "np.ndarray", hist2: "np.ndarray") -> float:
    """
    Compare two LBP histograms using chi-square distance.
    Helper function for distribution comparison.
    """
    if len(hist1) == 0 or len(hist2) == 0 or np is None:
        return 1.0
    
    chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-10))
    return float(chi_square)


def _gabor_filter_analysis(gray_img: "np.ndarray") -> float:
    """
    Apply Gabor filters to analyze texture orientation patterns.
    AI images often have different orientation distributions.
    """
    if gabor is None or np is None:
        return 0.5
    
    try:
        # Define Gabor filter parameters
        frequencies = [0.1, 0.2, 0.3]  # Different frequencies
        orientations = [0, 45, 90, 135]  # Different orientations (degrees)
        
        gabor_responses = []
        
        for freq in frequencies:
            for theta in orientations:
                # Convert theta to radians
                theta_rad = np.radians(theta)
                
                # Apply Gabor filter
                real, _ = gabor(gray_img, frequency=freq, theta=theta_rad)
                
                # Calculate response statistics
                response_mean = np.mean(real)
                response_std = np.std(real)
                response_energy = np.sum(real ** 2)
                
                gabor_responses.extend([response_mean, response_std, response_energy])
        
        if not gabor_responses:
            return 0.5
        
        # Analyze Gabor response patterns
        gabor_array = np.array(gabor_responses)
        
        # AI images often have more uniform Gabor responses
        gabor_variance = np.var(gabor_array)
        gabor_mean = np.mean(np.abs(gabor_array))
        
        # Low variance in Gabor responses suggests AI generation
        if gabor_variance < 100:  # Very low variance
            return 0.8
        elif gabor_variance < 500:  # Low variance
            return 0.6
        else:
            return 0.3
            
    except Exception as e:
        logger.debug(f"Gabor filter analysis failed: {e}")
        return 0.5


def _glcm_analysis(gray_img: "np.ndarray") -> float:
    """
    Gray-Level Co-occurrence Matrix (GLCM) analysis for texture patterns.
    """
    if graycomatrix is None or graycoprops is None or np is None:
        return 0.5
    
    try:
        # Ensure image is in correct format for GLCM
        if gray_img.dtype != np.uint8:
            gray_img = (gray_img * 255).astype(np.uint8)
        
        # Define distances and angles for GLCM
        distances = [1, 2]
        angles = [0, 45, 90, 135]  # degrees
        angles_rad = [np.radians(a) for a in angles]
        
        # Calculate GLCM
        glcm = graycomatrix(gray_img, distances=distances, angles=angles_rad, 
                           levels=256, symmetric=True, normed=True)
        
        # Extract texture properties
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        glcm_features = []
        
        for prop in properties:
            feature = graycoprops(glcm, prop)
            glcm_features.extend(feature.flatten())
        
        if not glcm_features:
            return 0.5
        
        # Analyze GLCM feature patterns
        features_array = np.array(glcm_features)
        
        # AI images often have different GLCM patterns
        # Check for unusual homogeneity and energy values
        homogeneity_values = [f for i, f in enumerate(glcm_features) if i % 5 == 3]  # Homogeneity is 4th property
        energy_values = [f for i, f in enumerate(glcm_features) if i % 5 == 4]  # Energy is 5th property
        
        if homogeneity_values and energy_values:
            avg_homogeneity = np.mean(homogeneity_values)
            avg_energy = np.mean(energy_values)
            
            # AI images often have very high homogeneity and low energy
            if avg_homogeneity > 0.8 and avg_energy < 0.1:
                return 0.8
            elif avg_homogeneity > 0.7 and avg_energy < 0.2:
                return 0.6
            else:
                return 0.3
        else:
            return 0.5
            
    except Exception as e:
        logger.debug(f"GLCM analysis failed: {e}")
        return 0.5


def _fractal_dimension_analysis(gray_img: "np.ndarray") -> float:
    """
    Calculate fractal dimension to analyze image complexity.
    AI images often have different complexity patterns.
    """
    if np is None:
        return 0.5
    
    try:
        # Box-counting method for fractal dimension
        def box_counting(image, box_size):
            """Count boxes needed to cover the image at given box size."""
            h, w = image.shape
            boxes_h = h // box_size
            boxes_w = w // box_size
            
            count = 0
            for i in range(boxes_h):
                for j in range(boxes_w):
                    box = image[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size]
                    if np.any(box > 0):  # Non-empty box
                        count += 1
            return count
        
        # Test different box sizes
        box_sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for size in box_sizes:
            if size < min(gray_img.shape):
                count = box_counting(gray_img, size)
                counts.append(count)
        
        if len(counts) < 3:
            return 0.5
        
        # Calculate fractal dimension using linear regression
        log_sizes = np.log(box_sizes[:len(counts)])
        log_counts = np.log(counts)
        
        # Linear regression: log(count) = -D * log(size) + C
        if len(log_sizes) > 1:
            slope = np.polyfit(log_sizes, log_counts, 1)[0]
            fractal_dimension = -slope
            
            # AI images often have different fractal dimensions
            # Natural images typically have D around 2.0-2.5
            if fractal_dimension < 1.5:  # Very low complexity
                return 0.8
            elif fractal_dimension < 2.0:  # Low complexity
                return 0.6
            elif fractal_dimension > 2.8:  # Very high complexity
                return 0.7
            else:
                return 0.3
        else:
            return 0.5
            
    except Exception as e:
        logger.debug(f"Fractal dimension analysis failed: {e}")
        return 0.5


def _edge_coherence_analysis(gray_img: "np.ndarray") -> float:
    """
    Analyze edge coherence patterns.
    AI images often have different edge characteristics.
    """
    if np is None:
        return 0.5
    
    try:
        # Simple edge detection using gradient
        grad_x = np.gradient(gray_img, axis=1)
        grad_y = np.gradient(gray_img, axis=0)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Threshold for edge pixels
        edge_threshold = np.percentile(magnitude, 90)  # Top 10% of gradients
        edge_mask = magnitude > edge_threshold
        
        if not np.any(edge_mask):
            return 0.5
        
        # Analyze edge direction coherence
        edge_directions = direction[edge_mask]
        
        # Calculate direction histogram
        hist, bins = np.histogram(edge_directions, bins=8, range=(-np.pi, np.pi))
        
        # AI images often have more uniform edge directions
        direction_variance = np.var(hist)
        direction_entropy = -np.sum((hist / np.sum(hist)) * np.log(hist / np.sum(hist) + 1e-10))
        
        # Low variance and high entropy suggest AI generation
        if direction_variance < 10 and direction_entropy > 2.0:
            return 0.8
        elif direction_variance < 20 and direction_entropy > 1.5:
            return 0.6
        else:
            return 0.3
            
    except Exception as e:
        logger.debug(f"Edge coherence analysis failed: {e}")
        return 0.5


def analyze_texture_lbp(img: "Image.Image", img_array: Optional["np.ndarray"] = None) -> MechanismResult:
    """
    Analyze image texture using Local Binary Patterns (LBP).
    
    LBP captures micro-texture patterns. Real photos have organic texture
    variation, while AI-generated images often have subtle repetitive patterns
    or unnatural uniformity.
    
    Analyzes:
    - Multi-scale LBP (3 scales)
    - Regional texture consistency
    - Repetitive pattern detection
    
    Returns MechanismResult with probability_ai and detailed diagnostics.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    
    # Graceful fallback if scikit-image not available
    if local_binary_pattern is None or np is None:
        logger.warning("scikit-image not available - texture analysis returning neutral score")
        return MechanismResult(
            mechanism=MechanismType.TEXTURE_LBP,
            probability_ai=0.5,
            notes="scikit-image not installed - texture analysis skipped",
        )
    
    try:
        # Convert to numpy if not provided
        if img_array is None:
            img_array = np.asarray(img, dtype=np.uint8)
        
        # Convert to grayscale for LBP
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = img_array
        
        logger.debug("Computing multi-scale LBP...")
        lbp_scales = _multiscale_lbp(gray)
        
        if len(lbp_scales) == 0:
            return MechanismResult(
                mechanism=MechanismType.TEXTURE_LBP,
                probability_ai=0.5,
                notes="LBP computation failed",
            )
        
        logger.debug("Performing regional LBP analysis...")
        regional_hists = _regional_lbp_analysis(gray)
        
        logger.debug("Computing texture consistency...")
        consistency_score = _texture_consistency_score(regional_hists)
        
        logger.debug("Detecting repetitive patterns...")
        repetition_score = _detect_repetitive_patterns(regional_hists)
        
        # New enhanced texture analyses
        logger.debug("Applying Gabor filters...")
        gabor_score = _gabor_filter_analysis(gray)
        
        logger.debug("Computing GLCM features...")
        glcm_score = _glcm_analysis(gray)
        
        logger.debug("Calculating fractal dimension...")
        fractal_score = _fractal_dimension_analysis(gray)
        
        logger.debug("Analyzing edge coherence...")
        edge_score = _edge_coherence_analysis(gray)
        
        # Combine scores with updated weights
        weights = {
            "consistency": 0.25,
            "repetition": 0.25,
            "gabor": 0.20,
            "glcm": 0.15,
            "fractal": 0.10,
            "edge": 0.05,
        }
        
        combined_score = (
            weights["consistency"] * consistency_score +
            weights["repetition"] * repetition_score +
            weights["gabor"] * gabor_score +
            weights["glcm"] * glcm_score +
            weights["fractal"] * fractal_score +
            weights["edge"] * edge_score
        )
        
        # Build diagnostics
        diagnostics = {
            "num_scales": len(lbp_scales),
            "num_regions": len(regional_hists),
            "consistency_score": float(consistency_score),
            "repetition_score": float(repetition_score),
            "gabor_score": float(gabor_score),
            "glcm_score": float(glcm_score),
            "fractal_score": float(fractal_score),
            "edge_score": float(edge_score),
            "combined_score": float(combined_score),
        }
        
        # Generate notes
        notes_list = []
        if consistency_score > 0.6:
            notes_list.append(f"Unnaturally uniform texture (score: {consistency_score:.2f})")
        if repetition_score > 0.5:
            notes_list.append(f"Repetitive patterns detected (score: {repetition_score:.2f})")
        if gabor_score > 0.6:
            notes_list.append(f"Unusual orientation patterns (score: {gabor_score:.2f})")
        if glcm_score > 0.6:
            notes_list.append(f"Atypical texture co-occurrence (score: {glcm_score:.2f})")
        if fractal_score > 0.6:
            notes_list.append(f"Unusual complexity patterns (score: {fractal_score:.2f})")
        if edge_score > 0.6:
            notes_list.append(f"Unnatural edge coherence (score: {edge_score:.2f})")
        
        if not notes_list:
            notes_list.append("Texture analysis shows natural variation")
        
        logger.debug(f"Texture analysis complete: AI probability = {combined_score:.3f}")
        
        return MechanismResult(
            mechanism=MechanismType.TEXTURE_LBP,
            probability_ai=float(combined_score),
            notes="; ".join(notes_list),
            extra=diagnostics,
        )
        
    except Exception as e:
        logger.error(f"Texture analysis failed: {e}", exc_info=True)
        return MechanismResult(
            mechanism=MechanismType.TEXTURE_LBP,
            probability_ai=0.5,
            notes=f"Texture analysis encountered an error: {str(e)[:100]}",
        )
