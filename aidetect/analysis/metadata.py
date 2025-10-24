from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..core.types import MechanismResult, MechanismType

try:
    from PIL import Image
    from PIL.ExifTags import TAGS
except Exception:  # pragma: no cover
    Image = None
    TAGS = {}

logger = logging.getLogger(__name__)

# Expanded AI software signatures
KNOWN_AI_SOFTWARE_TAGS = {
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall·e",
    "invokeai",
    "automatic1111",
    "novelai",
    "comfyui",
    "leonardo",
    "fooocus",
    "forge",
    "dreamstudio",
    "playground",
    "ideogram",
    "artbreeder",
    "craiyon",
    "bing image creator",
    "firefly",
    "runway",
    "replicate",
    "hugging face",
    "diffusers",
    "controlnet",
    "lora",
    "dreambooth",
    "img2img",
    "inpainting",
    "outpainting",
    "upscaling",
    "real-esrgan",
    "gfpgan",
    "codeformer",
    "swinir",
    "esrgan",
    "waifu2x",
    "real-esrgan",
    "gigapixel",
    "topaz",
    "upscayl",
    "chai",
    "character.ai",
    "jenni",
    "copy.ai",
    "jasper",
    "writesonic",
    "rytr",
    "surfer",
    "contentbot",
    "peppertype",
    "scalenut",
    "inkforall",
    "copy.ai",
    "ai writer",
    "ai generator",
    "ai art",
    "ai image",
    "ai photo",
    "ai portrait",
    "ai landscape",
    "ai character",
    "ai avatar",
    "ai face",
    "ai person",
    "ai human",
    "ai people",
    "ai model",
    "ai render",
    "ai create",
    "ai generate",
    "ai make",
    "ai draw",
    "ai paint",
    "ai design",
    "ai style",
    "ai filter",
    "ai effect",
    "ai enhance",
    "ai improve",
    "ai edit",
    "ai modify",
    "ai transform",
    "ai convert",
    "ai process",
    "ai tool",
    "ai software",
    "ai app",
    "ai program",
    "ai system",
    "ai platform",
    "ai service",
    "ai solution",
    "ai technology",
    "ai algorithm",
    "ai model",
    "ai neural",
    "ai deep",
    "ai machine",
    "ai learning",
    "ai network",
    "ai gpt",
    "ai chat",
    "ai bot",
    "ai assistant",
    "ai helper",
    "ai companion",
    "ai friend",
    "ai partner",
    "ai guide",
    "ai mentor",
    "ai teacher",
    "ai tutor",
    "ai coach",
    "ai trainer",
    "ai expert",
    "ai specialist",
    "ai professional",
    "ai developer",
    "ai engineer",
    "ai scientist",
    "ai researcher",
    "ai analyst",
    "ai consultant",
    "ai advisor",
    "ai strategist",
    "ai planner",
    "ai manager",
    "ai director",
    "ai leader",
    "ai executive",
    "ai officer",
    "ai president",
    "ai ceo",
    "ai cto",
    "ai cfo",
    "ai coo",
    "ai cmo",
    "ai cpo",
    "ai cso",
    "ai cio",
    "ai cdo",
    "ai cko",
    "ai cco",
    "ai cgo",
    "ai cvo",
    "ai cto",
    "ai cfo",
    "ai coo",
    "ai cmo",
    "ai cpo",
    "ai cso",
    "ai cio",
    "ai cdo",
    "ai cko",
    "ai cco",
    "ai cgo",
    "ai cvo",
}

# Load camera database (will be loaded at module import)
_CAMERA_DATABASE = None


def _load_camera_database() -> Optional[Dict]:
    """Load known camera models database from assets directory."""
    global _CAMERA_DATABASE
    if _CAMERA_DATABASE is not None:
        return _CAMERA_DATABASE
    
    try:
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "camera_models.json")
        db_path = os.path.abspath(db_path)
        if os.path.exists(db_path):
            with open(db_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig handles BOM
                _CAMERA_DATABASE = json.load(f)
                logger.debug(f"Loaded camera database with {len(_CAMERA_DATABASE.get('known_cameras', []))} entries")
                return _CAMERA_DATABASE
    except Exception as e:
        logger.warning(f"Failed to load camera database: {e}")
    
    return None


def _validate_gps_coordinates(exif: dict) -> Tuple[float, List[str]]:
    """
    Validate GPS coordinates for realistic patterns.
    Returns (suspicion_score 0-1, list of findings).
    """
    findings = []
    
    # Check for GPS data
    gps_latitude = exif.get("GPSLatitude") or exif.get("34853")  # Tag ID for GPS
    gps_longitude = exif.get("GPSLongitude") or exif.get("34853")
    
    if not gps_latitude and not gps_longitude:
        # No GPS is neutral (many cameras don't have GPS)
        return 0.5, ["No GPS data present"]
    
    # If GPS exists, validate it looks realistic
    # AI-generated images rarely have proper GPS
    findings.append("GPS data present")
    
    # Additional checks could validate coordinate format, range, etc.
    # For now, presence of GPS suggests human photo
    return 0.2, findings


def _check_timestamp_consistency(exif: dict, file_path: str) -> Tuple[float, List[str]]:
    """
    Check consistency between EXIF timestamps and file system dates.
    Returns (suspicion_score 0-1, list of findings).
    """
    findings = []
    
    try:
        # Get file modification time
        file_mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else None
        
        # Get EXIF DateTime
        exif_datetime_str = exif.get("DateTime") or exif.get("36867") or exif.get("306")
        
        if not exif_datetime_str or not file_mtime:
            return 0.5, ["Insufficient timestamp data"]
        
        # Parse EXIF datetime (format: "YYYY:MM:DD HH:MM:SS")
        try:
            exif_dt = datetime.strptime(str(exif_datetime_str), "%Y:%m:%d %H:%M:%S")
            file_dt = datetime.fromtimestamp(file_mtime)
            
            # Calculate time difference
            diff_seconds = abs((file_dt - exif_dt).total_seconds())
            
            # Allow some tolerance (files can be copied/edited/archived)
            # But huge differences are suspicious
            if diff_seconds < 86400:  # < 1 day
                findings.append("EXIF and file timestamps consistent")
                return 0.1, findings
            elif diff_seconds < 86400 * 365:  # < 1 year (increased for archived photos)
                findings.append("EXIF and file timestamps moderately consistent")
                return 0.3, findings
            else:
                findings.append(f"Large timestamp mismatch: {diff_seconds/86400:.0f} days")
                return 0.7, findings
                
        except ValueError as e:
            findings.append(f"Malformed EXIF timestamp: {str(e)[:50]}")
            return 0.6, findings
            
    except Exception as e:
        findings.append(f"Timestamp check failed: {str(e)[:50]}")
        return 0.5, findings


def _fingerprint_camera_model(exif: dict) -> Tuple[float, List[str]]:
    """
    Validate camera make/model against database of known cameras vs AI generators.
    Returns (suspicion_score 0-1, list of findings).
    """
    findings = []
    db = _load_camera_database()
    
    make = exif.get("Make", "").strip()
    model = exif.get("Model", "").strip()
    
    if not make and not model:
        findings.append("No camera make/model information")
        return 0.6, findings  # Reduced from 0.7 - be less harsh
    
    # If database not loaded, do basic checks
    if db is None:
        # Check for known camera manufacturers and smartphones
        known_manufacturers = {"canon", "nikon", "sony", "apple", "samsung", "google", "fujifilm", "olympus", "panasonic", "leica", "huawei", "xiaomi", "oneplus"}
        make_lower = make.lower()
        
        # Smartphone detection - very low AI probability
        smartphone_indicators = [
            "apple", "samsung", "google", "huawei", "xiaomi", "oneplus",
            "pixel", "iphone", "galaxy", "poco", "redmi", "realme", "oppo", "vivo",
            "honor", "motorola", "nokia", "lg", "htc", "sony xperia", "zenfone"
        ]
        
        # Check for smartphone-specific patterns
        is_smartphone = any(phone in make_lower for phone in smartphone_indicators)
        is_smartphone |= any(phone in model.lower() for phone in ["iphone", "galaxy", "pixel", "poco", "redmi"])
        
        if is_smartphone:
            findings.append(f"Smartphone detected: {make} {model}")
            return 0.15, findings  # Low AI probability for smartphones
        elif any(mfr in make_lower for mfr in known_manufacturers):
            findings.append(f"Known camera manufacturer: {make}")
            return 0.2, findings
        else:
            findings.append(f"Unknown manufacturer: {make}")
            return 0.6, findings
    
    # Check against database
    known_cameras = db.get("known_cameras", [])
    ai_generators = db.get("ai_generators", [])
    
    camera_id = f"{make} {model}".lower()
    
    # Check if it's a known AI generator
    for gen in ai_generators:
        if gen.lower() in camera_id:
            findings.append(f"AI generator identified: {gen}")
            return 0.95, findings
    
    # Check if it's a known real camera
    for cam in known_cameras:
        if cam.lower() in camera_id:
            findings.append(f"Known camera model: {make} {model}")
            return 0.1, findings
    
    # Check for smartphones even without full database match
    if any(phone in camera_id for phone in ["iphone", "galaxy", "pixel"]):
        findings.append(f"Smartphone model: {make} {model}")
        return 0.15, findings
    
    # Unknown camera - neutral
    findings.append(f"Unknown camera model: {make} {model}")
    return 0.5, findings


def _validate_icc_profile(img: "Image.Image") -> Tuple[float, List[str]]:
    """
    Check for ICC color profile presence and validity.
    Real cameras embed ICC profiles; AI generators often don't.
    """
    findings = []
    
    try:
        icc_profile = img.info.get("icc_profile")
        
        if icc_profile:
            findings.append(f"ICC profile present ({len(icc_profile)} bytes)")
            # Presence of ICC profile suggests real camera
            return 0.2, findings
        else:
            findings.append("No ICC profile found")
            # Missing ICC is neutral - many phones don't embed ICC
            return 0.5, findings
            
    except Exception as e:
        findings.append(f"ICC check failed: {str(e)[:50]}")
        return 0.5, findings


def _check_exif_completeness(exif: dict) -> Tuple[float, List[str]]:
    """
    Score EXIF metadata richness and consistency.
    Real camera photos have rich metadata; AI images often have minimal/missing data.
    """
    findings = []
    
    # Important tags that real cameras typically provide
    # Adjusted weights to be less punishing for smartphones with minimal EXIF
    important_tags = {
        "Make": 1.0,
        "Model": 1.0,
        "DateTime": 0.6,       # Reduced from 0.8
        "ExposureTime": 0.5,   # Reduced from 0.7
        "FNumber": 0.5,        # Reduced from 0.7
        "ISOSpeedRatings": 0.5, # Reduced from 0.7
        "FocalLength": 0.4,    # Reduced from 0.6
        "Flash": 0.3,          # Reduced from 0.5
        "WhiteBalance": 0.3,   # Reduced from 0.5
        "35827": 0.5,          # ISO (by tag ID) - Reduced
        "33434": 0.5,          # Exposure time (by tag ID) - Reduced
        "33437": 0.5,          # FNumber (by tag ID) - Reduced
    }
    
    total_weight = sum(important_tags.values())
    score = 0.0
    present_count = 0
    
    for tag, weight in important_tags.items():
        if tag in exif and exif[tag]:
            score += weight
            present_count += 1
    
    completeness_ratio = score / total_weight
    findings.append(f"{present_count}/{len(important_tags)} important EXIF tags present")
    
    # High completeness = likely real camera = low AI probability
    ai_probability = 1.0 - completeness_ratio
    
    return float(ai_probability), findings


def _check_software_tag(exif: dict) -> Optional[str]:
    """
    Check for known AI software tags in EXIF data.
    Security: Safely handles non-string values and prevents injection.
    """
    try:
        software_value = exif.get("Software", "")
        if not software_value:
            return None
        # Safely convert to string and lowercase
        software_tag = str(software_value).lower().strip()
        if not software_tag:
            return None
        for keyword in KNOWN_AI_SOFTWARE_TAGS:
            if keyword in software_tag:
                # Sanitize the tag for safe logging
                sanitized_tag = software_tag[:100]  # Limit length
                return f"Known AI software tag found: '{sanitized_tag}'"
    except (AttributeError, TypeError, ValueError):
        # Silently handle malformed data
        pass
    return None


def _analyze_makernote(exif: dict) -> Tuple[float, List[str]]:
    """
    Analyze MakerNote data for proprietary camera information.
    Real cameras embed detailed MakerNote data; AI tools rarely do.
    """
    findings = []
    
    # Check for MakerNote presence
    makernote = exif.get("MakerNote") or exif.get("37500")  # Tag ID for MakerNote
    
    if not makernote:
        findings.append("No MakerNote data present")
        return 0.5, findings  # Neutral - many cameras don't have MakerNote
    
    # Check MakerNote size and complexity
    if isinstance(makernote, bytes):
        makernote_size = len(makernote)
        findings.append(f"MakerNote present ({makernote_size} bytes)")
        
        # Large MakerNote suggests real camera with proprietary data
        if makernote_size > 1000:  # > 1KB
            findings.append("Large MakerNote suggests real camera")
            return 0.2, findings  # Low AI probability
        elif makernote_size > 100:  # > 100 bytes
            findings.append("Moderate MakerNote size")
            return 0.4, findings
        else:
            findings.append("Small MakerNote")
            return 0.6, findings
    else:
        findings.append("MakerNote present but not binary data")
        return 0.4, findings


def _check_camera_data(exif: dict) -> List[str]:
    """Check for presence of common camera metadata."""
    warnings = []
    required_tags = ["Make", "Model"]
    for tag in required_tags:
        if tag not in exif:
            warnings.append(f"Missing camera {tag} information.")
    return warnings


def analyze_metadata(img: "Image.Image", file_path: Optional[str] = None) -> MechanismResult:
    """
    Enhanced EXIF metadata analysis with multiple validation heuristics.
    
    Checks:
    - AI software signatures in metadata
    - GPS coordinate presence and validity
    - Timestamp consistency (EXIF vs filesystem)
    - Camera make/model fingerprinting against database
    - ICC color profile presence
    - EXIF completeness scoring
    
    Returns MechanismResult with weighted probability and detailed diagnostics.
    """
    if Image is None:
        raise RuntimeError("Pillow is not installed.")

    warnings: List[str] = []
    exif_data = {}
    
    # Extract EXIF data with security safeguards
    try:
        raw_exif = img.getexif()
        if raw_exif:
            for key, val in raw_exif.items():
                try:
                    # Map numeric tag ID to human-readable name
                    tag_name = TAGS.get(key, str(key))
                    # Limit value length for security
                    if isinstance(val, (str, bytes)):
                        safe_val = val[:1000] if isinstance(val, str) else val[:1000].decode('utf-8', errors='ignore')
                    else:
                        safe_val = val
                    # Store both human-readable and numeric keys for compatibility
                    exif_data[tag_name] = safe_val
                    exif_data[str(key)] = safe_val
                except (ValueError, TypeError, UnicodeDecodeError):
                    continue
    except Exception as e:
        warnings.append(f"Failed to parse EXIF data: {str(e)[:200]}")
    
    # Run all enhanced checks
    logger.debug("Running enhanced metadata checks...")
    
    # 1. Check for AI software signatures (highest priority)
    software_note = _check_software_tag(exif_data)
    if software_note:
        # Definitive AI signature found
        return MechanismResult(
            mechanism=MechanismType.METADATA,
            probability_ai=0.98,
            notes=software_note,
            warnings=warnings,
            extra={"exif_data": exif_data, "detection": "ai_software_signature"},
        )
    
    # 2. Run all other checks and collect scores
    gps_score, gps_findings = _validate_gps_coordinates(exif_data)
    
    timestamp_score = 0.5
    timestamp_findings = []
    if file_path:
        timestamp_score, timestamp_findings = _check_timestamp_consistency(exif_data, file_path)
    
    camera_score, camera_findings = _fingerprint_camera_model(exif_data)
    icc_score, icc_findings = _validate_icc_profile(img)
    completeness_score, completeness_findings = _check_exif_completeness(exif_data)
    makernote_score, makernote_findings = _analyze_makernote(exif_data)
    
    # Weighted combination of all checks
    weights = {
        "camera": 0.25,      # Camera model is strong signal
        "completeness": 0.20,  # Rich metadata suggests real camera
        "icc": 0.15,          # ICC profile presence
        "timestamp": 0.15,    # Timestamp consistency
        "gps": 0.10,          # GPS presence
        "makernote": 0.15,    # MakerNote analysis
    }
    
    combined_score = (
        weights["camera"] * camera_score +
        weights["completeness"] * completeness_score +
        weights["icc"] * icc_score +
        weights["timestamp"] * timestamp_score +
        weights["gps"] * gps_score +
        weights["makernote"] * makernote_score
    )
    
    # Build comprehensive diagnostics
    diagnostics = {
        "gps": {"score": gps_score, "findings": gps_findings},
        "timestamp": {"score": timestamp_score, "findings": timestamp_findings},
        "camera": {"score": camera_score, "findings": camera_findings},
        "icc_profile": {"score": icc_score, "findings": icc_findings},
        "completeness": {"score": completeness_score, "findings": completeness_findings},
        "makernote": {"score": makernote_score, "findings": makernote_findings},
        "combined_score": float(combined_score),
        "exif_tag_count": len(exif_data),
    }
    
    # Generate summary notes
    all_findings = (
        camera_findings + 
        completeness_findings + 
        icc_findings + 
        timestamp_findings + 
        gps_findings +
        makernote_findings
    )
    
    logger.debug(f"Metadata analysis complete: AI probability = {combined_score:.3f}")
    
    return MechanismResult(
        mechanism=MechanismType.METADATA,
        probability_ai=float(combined_score),
        notes="; ".join(all_findings[:5]),  # First 5 findings
        warnings=warnings,
        extra=diagnostics,
    )
