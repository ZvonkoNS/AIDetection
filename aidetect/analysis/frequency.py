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
    from scipy import fft
    from scipy.stats import kurtosis, skew
except Exception:  # pragma: no cover
    fft = None
    kurtosis = None
    skew = None

logger = logging.getLogger(__name__)


def _extract_dct_coefficients(img_array: "np.ndarray") -> "np.ndarray":
    """
    Extract DCT coefficients from image blocks (8x8 for JPEG-like analysis).
    Returns flattened array of DCT coefficients.
    """
    if fft is None:
        return np.array([])
    
    # Convert to grayscale for DCT analysis
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    h, w = gray.shape
    # Process 8x8 blocks (JPEG DCT standard)
    block_size = 8
    dct_coeffs = []
    
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = gray[i:i+block_size, j:j+block_size].astype(float)
            # Apply 2D DCT
            dct_block = fft.dctn(block, norm='ortho')
            dct_coeffs.extend(dct_block.flatten())
    
    return np.array(dct_coeffs)


def _analyze_dct_distribution(dct_coeffs: "np.ndarray") -> Dict[str, float]:
    """
    Analyze statistical properties of DCT coefficient distribution.
    AI-generated images often have unusual DCT statistics.
    """
    if len(dct_coeffs) == 0 or kurtosis is None:
        return {}
    
    return {
        "mean": float(np.mean(dct_coeffs)),
        "std": float(np.std(dct_coeffs)),
        "skewness": float(skew(dct_coeffs)),
        "kurtosis": float(kurtosis(dct_coeffs)),
    }


def _detect_quantization_artifacts(dct_coeffs: "np.ndarray") -> float:
    """
    Detect unnatural quantization patterns in DCT coefficients.
    Returns a score 0-1 indicating likelihood of artificial quantization.
    """
    if len(dct_coeffs) == 0:
        return 0.5
    
    # Check for unusual concentration at specific coefficient values
    # AI images sometimes show clustering at certain DCT values
    hist, _ = np.histogram(dct_coeffs, bins=50)
    hist_normalized = hist / np.sum(hist)
    
    # High entropy = natural, Low entropy = suspicious
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
    max_entropy = np.log2(50)  # Maximum for 50 bins
    
    # Lower entropy suggests quantization artifacts (more AI-like)
    normalized_entropy = entropy / max_entropy
    artifact_score = 1.0 - normalized_entropy
    
    return float(np.clip(artifact_score, 0, 1))


def _analyze_noise_spectrum(img_array: "np.ndarray") -> Tuple[float, Dict[str, float]]:
    """
    Analyze high-frequency noise using 2D FFT.
    Real camera sensor noise has characteristic patterns; AI is different.
    """
    if fft is None or np is None:
        return 0.5, {}
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_array
    
    # Compute 2D FFT
    f_transform = fft.fft2(gray)
    f_shift = fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Analyze high-frequency content
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Define high-frequency region (outer 30% of spectrum)
    radius = min(center_h, center_w)
    high_freq_threshold = int(0.7 * radius)
    
    # Create mask for high frequencies
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    high_freq_mask = distance > high_freq_threshold
    
    high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
    total_energy = np.mean(magnitude_spectrum)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
    
    diagnostics = {
        "high_freq_energy": float(high_freq_energy),
        "total_energy": float(total_energy),
        "high_freq_ratio": float(high_freq_ratio),
    }
    
    # AI images often have lower high-frequency content (smoother)
    # High ratio = more noise = more human-like
    # Reduced multiplier from 2.0 to 1.5 to be more lenient on clean photos
    ai_probability = 1.0 - np.clip(high_freq_ratio * 1.5, 0, 1)
    
    return float(ai_probability), diagnostics


def _detect_periodic_artifacts(magnitude_spectrum: "np.ndarray") -> float:
    """
    Detect periodic artifacts from GAN upsampling in the power spectrum.
    GANs often leave periodic patterns in frequency domain.
    """
    if np is None:
        return 0.5
    
    # Look for unusual peaks in the spectrum
    # Flatten and find peaks
    spectrum_flat = magnitude_spectrum.flatten()
    threshold = np.percentile(spectrum_flat, 99)
    peaks = spectrum_flat > threshold
    num_peaks = np.sum(peaks)
    
    # Too many strong peaks suggests periodic artifacts
    # Normalize by image size
    total_values = len(spectrum_flat)
    peak_ratio = num_peaks / total_values
    
    # Higher peak ratio = more periodic = more AI-like
    periodic_score = np.clip(peak_ratio * 100, 0, 1)
    
    return float(periodic_score)


def _benford_law_check(dct_coeffs: "np.ndarray") -> float:
    """
    Check if DCT coefficient first digits follow Benford's Law.
    Natural images tend to follow it; AI-generated may not.
    """
    if len(dct_coeffs) == 0:
        return 0.5
    
    # Get absolute values and extract first significant digit
    abs_coeffs = np.abs(dct_coeffs)
    # Filter out zeros and very small values
    significant = abs_coeffs[abs_coeffs > 1.0]
    
    if len(significant) < 100:  # Need enough samples
        return 0.5
    
    # Extract first digit
    first_digits = []
    for val in significant:
        # Get first digit
        while val >= 10:
            val /= 10
        first_digits.append(int(val))
    
    # Benford's Law expected distribution
    benford_expected = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }
    
    # Calculate observed distribution
    observed = {}
    for digit in range(1, 10):
        count = first_digits.count(digit)
        observed[digit] = count / len(first_digits)
    
    # Chi-square test
    chi_square = 0
    for digit in range(1, 10):
        expected = benford_expected[digit]
        obs = observed.get(digit, 0)
        chi_square += ((obs - expected) ** 2) / expected
    
    # Normalize chi-square to 0-1 range
    # Higher chi-square = less Benford-like = more AI-like
    # Increased divisor from 10.0 to 15.0 to be more lenient
    benford_deviation = np.clip(chi_square / 15.0, 0, 1)
    
    return float(benford_deviation)


def _analyze_jpeg_quantization_tables(img_path: str) -> float:
    """
    Analyze JPEG quantization tables for AI generation signatures.
    AI tools often use different quantization settings than cameras.
    """
    try:
        import struct
        
        with open(img_path, 'rb') as f:
            data = f.read()
        
        # Look for quantization table markers (0xFFDB)
        qtable_scores = []
        pos = 0
        while pos < len(data) - 1:
            if data[pos:pos+2] == b'\xFF\xDB':
                # Found quantization table
                length = struct.unpack('>H', data[pos+2:pos+4])[0]
                if length >= 67:  # Minimum size for quantization table
                    qtable_data = data[pos+4:pos+length]
                    if len(qtable_data) >= 64:  # Standard 8x8 table
                        # Extract quantization values
                        qvals = []
                        for i in range(0, min(64, len(qtable_data)), 1):
                            if i < len(qtable_data):
                                qvals.append(qtable_data[i])
                        
                        if len(qvals) == 64:
                            # Analyze quantization pattern
                            qvals_array = np.array(qvals)
                            
                            # AI tools often use different quantization patterns
                            # Check for unusual patterns
                            variance = np.var(qvals_array)
                            mean_val = np.mean(qvals_array)
                            
                            # High variance suggests AI-generated quantization
                            if variance > 1000:  # Threshold for suspicious variance
                                qtable_scores.append(0.8)
                            elif variance > 500:
                                qtable_scores.append(0.6)
                            else:
                                qtable_scores.append(0.3)
                
                pos += length
            else:
                pos += 1
        
        if qtable_scores:
            return float(np.mean(qtable_scores))
        else:
            return 0.5  # No quantization tables found
            
    except Exception as e:
        logger.debug(f"JPEG quantization analysis failed: {e}")
        return 0.5


def _detect_block_artifacts(dct_coeffs: "np.ndarray") -> float:
    """
    Detect 8x8 DCT boundary artifacts characteristic of AI generation.
    """
    if len(dct_coeffs) == 0:
        return 0.5
    
    # Reshape DCT coefficients into 8x8 blocks
    block_size = 64  # 8x8 = 64 coefficients
    if len(dct_coeffs) < block_size:
        return 0.5
    
    # Group into blocks
    num_blocks = len(dct_coeffs) // block_size
    blocks = dct_coeffs[:num_blocks * block_size].reshape(num_blocks, block_size)
    
    # Analyze DC coefficients (first coefficient in each block)
    dc_coeffs = blocks[:, 0]
    
    # Check for block boundary artifacts
    # AI images often have more uniform DC coefficients
    dc_variance = np.var(dc_coeffs)
    dc_mean = np.mean(np.abs(dc_coeffs))
    
    # Low variance in DC coefficients suggests AI generation
    if dc_variance < 100:  # Very low variance
        return 0.8
    elif dc_variance < 500:  # Low variance
        return 0.6
    else:
        return 0.3


def _spectral_residual_analysis(fft_magnitude: "np.ndarray") -> float:
    """
    Analyze spectral residual patterns for GAN fingerprints.
    """
    if np is None or len(fft_magnitude) == 0:
        return 0.5
    
    # Compute spectral residual (log magnitude)
    log_magnitude = np.log(fft_magnitude + 1e-10)
    
    # Apply high-pass filter to detect GAN artifacts
    h, w = log_magnitude.shape
    center_h, center_w = h // 2, w // 2
    
    # Create high-pass filter
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    high_pass_filter = distance > min(center_h, center_w) * 0.3
    
    # Apply filter
    filtered = log_magnitude * high_pass_filter
    
    # Analyze residual patterns
    residual_variance = np.var(filtered)
    residual_mean = np.mean(np.abs(filtered))
    
    # GAN images often have specific spectral patterns
    if residual_variance > 2.0:  # High variance in high frequencies
        return 0.7
    elif residual_variance > 1.0:
        return 0.5
    else:
        return 0.3


def analyze_frequency(img: "Image.Image", file_path: str = None) -> MechanismResult:
    """
    Enhanced frequency domain analysis using DCT, FFT, and statistical methods.
    
    Analyzes:
    - DCT coefficient distributions (JPEG-like block analysis)
    - High-frequency noise patterns via FFT
    - Periodic artifacts from GAN upsampling
    - Benford's Law compliance on DCT coefficients
    
    Returns MechanismResult with probability_ai and detailed diagnostics.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")
    
    # Fallback if scipy not available
    if np is None or fft is None:
        logger.warning("scipy not available - frequency analysis returning neutral score")
        return MechanismResult(
            mechanism=MechanismType.FREQUENCY,
            probability_ai=0.5,
            notes="scipy not installed - frequency analysis skipped",
        )
    
    try:
        # Convert to numpy array
        img_array = np.asarray(img, dtype=np.uint8)
        
        logger.debug("Extracting DCT coefficients...")
        dct_coeffs = _extract_dct_coefficients(img_array)
        
        logger.debug("Analyzing DCT distribution...")
        dct_stats = _analyze_dct_distribution(dct_coeffs)
        
        logger.debug("Detecting quantization artifacts...")
        quant_score = _detect_quantization_artifacts(dct_coeffs)
        
        logger.debug("Analyzing noise spectrum with FFT...")
        noise_score, noise_diagnostics = _analyze_noise_spectrum(img_array)
        
        # Compute FFT for periodic artifact detection
        if len(img_array.shape) == 3:
            gray = np.dot(img_array[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            gray = img_array
        
        f_transform = fft.fft2(gray)
        f_shift = fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        logger.debug("Detecting periodic artifacts...")
        periodic_score = _detect_periodic_artifacts(magnitude_spectrum)
        
        logger.debug("Checking Benford's Law compliance...")
        benford_score = _benford_law_check(dct_coeffs)
        
        # New enhanced analyses
        jpeg_quant_score = 0.5
        block_artifact_score = 0.5
        spectral_residual_score = 0.5
        
        if file_path:
            logger.debug("Analyzing JPEG quantization tables...")
            jpeg_quant_score = _analyze_jpeg_quantization_tables(file_path)
        
        logger.debug("Detecting block artifacts...")
        block_artifact_score = _detect_block_artifacts(dct_coeffs)
        
        logger.debug("Analyzing spectral residuals...")
        spectral_residual_score = _spectral_residual_analysis(magnitude_spectrum)
        
        # Combine scores with weights
        weights = {
            "quantization": 0.15,
            "noise": 0.25,
            "periodic": 0.25,
            "benford": 0.15,
            "jpeg_quant": 0.10,
            "block_artifacts": 0.05,
            "spectral_residual": 0.05,
        }
        
        combined_score = (
            weights["quantization"] * quant_score +
            weights["noise"] * noise_score +
            weights["periodic"] * periodic_score +
            weights["benford"] * benford_score +
            weights["jpeg_quant"] * jpeg_quant_score +
            weights["block_artifacts"] * block_artifact_score +
            weights["spectral_residual"] * spectral_residual_score
        )
        
        # Build diagnostic information
        diagnostics = {
            "dct_statistics": dct_stats,
            "quantization_score": float(quant_score),
            "noise_analysis": noise_diagnostics,
            "noise_score": float(noise_score),
            "periodic_score": float(periodic_score),
            "benford_score": float(benford_score),
            "jpeg_quantization_score": float(jpeg_quant_score),
            "block_artifacts_score": float(block_artifact_score),
            "spectral_residual_score": float(spectral_residual_score),
            "combined_score": float(combined_score),
            "num_dct_coeffs": len(dct_coeffs),
        }
        
        # Generate notes with adjusted thresholds
        notes_list = []
        if quant_score > 0.7:  # Increased from 0.6
            notes_list.append(f"Unusual quantization pattern detected (score: {quant_score:.2f})")
        if noise_score > 0.6:
            notes_list.append(f"Atypical noise spectrum (score: {noise_score:.2f})")
        if periodic_score > 0.5:  # Keep aggressive - AI-specific
            notes_list.append(f"Periodic artifacts detected (score: {periodic_score:.2f})")
        if benford_score > 0.5:  # Increased from 0.4
            notes_list.append(f"DCT coefficients deviate from Benford's Law (score: {benford_score:.2f})")
        if jpeg_quant_score > 0.6:
            notes_list.append(f"Unusual JPEG quantization tables (score: {jpeg_quant_score:.2f})")
        if block_artifact_score > 0.6:
            notes_list.append(f"Block artifacts detected (score: {block_artifact_score:.2f})")
        if spectral_residual_score > 0.6:
            notes_list.append(f"Spectral residual patterns suggest AI generation (score: {spectral_residual_score:.2f})")
        
        if not notes_list:
            notes_list.append("Frequency analysis shows natural characteristics")
        
        logger.debug(f"Frequency analysis complete: AI probability = {combined_score:.3f}")
        
        return MechanismResult(
            mechanism=MechanismType.FREQUENCY,
            probability_ai=float(combined_score),
            notes="; ".join(notes_list),
            extra=diagnostics,
        )
        
    except Exception as e:
        logger.error(f"Frequency analysis failed: {e}", exc_info=True)
        return MechanismResult(
            mechanism=MechanismType.FREQUENCY,
            probability_ai=0.5,
            notes=f"Frequency analysis encountered an error: {str(e)[:100]}",
        )
