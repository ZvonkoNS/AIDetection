# Metadata Signatures Reference

NEXT SIGHT cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com  
**Last Updated**: October 2, 2025

This document catalogs known metadata signatures for camera equipment and AI image generators, used by the Enhanced Metadata Validation mechanism.

---

## Known Camera Manufacturers

Real camera photos typically contain:
- **Make**: Manufacturer name (Canon, Nikon, Sony, Apple, etc.)
- **Model**: Specific camera/phone model
- **Rich EXIF**: Exposure settings, GPS, timestamps, lens info
- **ICC Profile**: Embedded color calibration
- **Consistent timestamps**: EXIF datetime matches file creation

### Top Camera Brands
- Canon (EOS series, PowerShot)
- Nikon (D-series, Z-series)
- Sony (Alpha series, Cyber-shot)
- Apple (iPhone 6-15 series)
- Samsung (Galaxy series)
- Google (Pixel series)
- Fujifilm (X-series, GFX)
- Olympus (OM-D series)
- Panasonic (LUMIX series)
- Leica (M-series, Q-series)

---

## Known AI Generator Signatures

AI-generated images often have:
- **Software tag**: Generator name in EXIF
- **Missing camera info**: No Make/Model
- **Minimal EXIF**: Missing exposure, GPS, lens data
- **No ICC profile**: Missing color calibration
- **Timestamp mismatches**: EXIF vs file time inconsistencies

### AI Image Generators

| Generator | Common Signatures | Detection Priority |
|-----------|------------------|-------------------|
| **Stable Diffusion** | "stable diffusion", "sd", metadata includes "parameters" | HIGH |
| **Midjourney** | "midjourney", often no EXIF at all | HIGH |
| **DALL-E** | "dall-e", "dallÂ·e", "openai" | HIGH |
| **InvokeAI** | "invokeai" | HIGH |
| **Automatic1111** | "automatic1111", "a1111" | HIGH |
| **ComfyUI** | "comfyui" | MEDIUM |
| **Leonardo.AI** | "leonardo" | MEDIUM |
| **Fooocus** | "fooocus" | MEDIUM |
| **Forge** | "forge" | MEDIUM |
| **DreamStudio** | "dreamstudio" | MEDIUM |
| **Playground AI** | "playground" | MEDIUM |
| **Ideogram** | "ideogram" | MEDIUM |
| **NovelAI** | "novelai" | LOW |

---

## Detection Heuristics

### High Confidence AI Detection (>90%)
1. Known AI software tag in EXIF
2. Metadata explicitly states generator parameters
3. "Prompt" or "negative_prompt" fields present

### Medium Confidence (60-80%)
1. Missing Make AND Model
2. No EXIF data at all (pristine file)
3. Timestamp inconsistencies >30 days
4. No ICC profile + missing exposure data

### Low Confidence (40-60%)
1. Minimal EXIF (only basic tags)
2. Unknown camera manufacturer
3. Missing GPS (neutral - many cameras lack GPS)

---

## Future Enhancements

- Lens distortion fingerprinting (optical signatures)
- Sensor noise pattern analysis (pixel-level)
- Bayer pattern detection (CFA artifacts)
- Defocus blur analysis (depth of field)

---

NEXT SIGHT cutting edge intelligence | contact us at info@next-sight.com | www.next-sight.com
