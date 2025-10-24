# Task List: Scoring Corrections & Conservative Mode

**Source**: False Positive Reduction Initiative  
**Critical Constraint**: Maintain or improve AI detection (55-80%), reduce real photo false positives (<40%)  
**Strategy**: Neutral scoring for absent data, aggressive scoring for positive AI evidence

---

## Relevant Files

### Files to Modify (Primary)
- `aidetect/analysis/metadata.py` - Adjust scoring logic for neutral "absent data", aggressive "AI signatures"
- `aidetect/analysis/quality.py` - Account for modern smartphone/computational photography
- `aidetect/analysis/frequency.py` - Fine-tune thresholds while maintaining AI detection
- `aidetect/analysis/texture.py` - Adjust consistency scoring for natural variation
- `aidetect/analysis/calibration.py` - Implement conservative mode and evidence requirements
- `aidetect/core/config.py` - Add conservative_mode flag and thresholds
- `aidetect/core/types.py` - Add evidence tracking and confidence level fields
- `aidetect/cli.py` - Add --conservative CLI flag
- `aidetect/runner/single.py` - Pass mode to calibration function

### Files to Update (Documentation)
- `README.md` - Document conservative mode and usage guidelines
- `USAGE.md` - Add examples with --conservative flag
- `docs/scoring-philosophy.md` - NEW: Document scoring rationale
- `docs/accuracy-phase1-results.md` - Update with before/after comparisons

### Test Files
- `tests/test_calibration.py` - Unit tests for conservative mode
- Manual testing on 5 real photos + 5 AI images

### Notes
- **Accuracy First**: Validate every change with real AND AI test images
- **No Degradation**: AI detection must remain 55-80%+
- **Target**: Real photos should score <40% AI probability
- **Evidence-Based**: Positive AI evidence > absence of camera evidence

---

## Tasks

- [ ] **1.0 Rebalance Metadata Scoring (Neutral Absent, Aggressive Present)**
  - [ ] 1.1 In `_validate_gps_coordinates()`: Change no GPS from (0.5, ...) to (0.5, ...) - already neutral, verify
  - [ ] 1.2 In `_fingerprint_camera_model()`: Change "no make/model" from 0.7 to 0.6
  - [ ] 1.3 In `_fingerprint_camera_model()`: Add smartphone detection - if Make in ["Apple", "Samsung", "Google"], return 0.25
  - [ ] 1.4 In `_validate_icc_profile()`: Change "no ICC" from 0.6 to 0.5 (neutral for phones)
  - [ ] 1.5 In `_check_timestamp_consistency()`: Increase tolerance from 30 days to 90 days for "moderately consistent"
  - [ ] 1.6 In `_check_exif_completeness()`: Adjust weights to be less punishing for minimal EXIF
  - [ ] 1.7 **VERIFY**: AI software tag detection still returns 0.98 (NO CHANGE)
  - [ ] 1.8 **VERIFY**: Known AI generator in camera model still returns 0.95 (NO CHANGE)
  - [ ] 1.9 Update metadata ensemble weights to prioritize positive evidence over absent data
  - [ ] 1.10 Test on real photo - target metadata score: 0.25-0.40
  - [ ] 1.11 Test on AI image - verify metadata score: 0.60-0.95 (depending on signatures)

- [ ] **2.0 Adjust Quality Metrics for Modern Photography**
  - [ ] 2.1 In `_measure_sharpness()`: Expand "normal range" to 0.1-0.9 (was 0.1-0.8) for modern phones
  - [ ] 2.2 In `_measure_sharpness()`: High sharpness (>0.8) returns 0.4 instead of 0.6
  - [ ] 2.3 In `_detect_chromatic_aberration()`: If CA is low, return 0.5 instead of 0.7 (phones correct CA)
  - [ ] 2.4 In `_analyze_vignetting()`: No vignetting returns 0.5 instead of 0.7 (phones correct it)
  - [ ] 2.5 In `_check_color_distribution()`: Adjust spikiness thresholds for vibrant phone photos
  - [ ] 2.6 In `_edge_sharpness_uniformity()`: Increase threshold - allow more uniformity before flagging
  - [ ] 2.7 In `_detect_oversaturation()`: Raise clipping threshold from 0.05 to 0.08
  - [ ] 2.8 Update quality metrics ensemble weights to reduce overall impact if all neutral
  - [ ] 2.9 Test on real photo - target quality score: 0.30-0.45
  - [ ] 2.10 Test on AI image - verify quality score: 0.45-0.70

- [ ] **3.0 Fine-Tune Frequency Analysis Thresholds**
  - [ ] 3.1 In `_detect_quantization_artifacts()`: Increase threshold from 0.6 to 0.7 before flagging
  - [ ] 3.2 In `_analyze_noise_spectrum()`: Adjust high_freq_ratio multiplier from 2.0 to 1.5 (more lenient)
  - [ ] 3.3 In `_detect_periodic_artifacts()`: Keep aggressive (this is AI-specific) - NO CHANGE
  - [ ] 3.4 In `_benford_law_check()`: Increase deviation threshold from 0.4 to 0.5 before flagging
  - [ ] 3.5 Update frequency internal weights to prioritize periodic artifacts (AI-specific)
  - [ ] 3.6 Test on real photo - target frequency score: 0.35-0.50
  - [ ] 3.7 Test on AI image - verify frequency score: 0.55-0.80

- [ ] **4.0 Calibrate Texture Analysis**
  - [ ] 4.1 In `_texture_consistency_score()`: Adjust mean_dist multiplier to be less sensitive
  - [ ] 4.2 In `_detect_repetitive_patterns()`: Increase correlation threshold from 0.9 to 0.95
  - [ ] 4.3 In `_detect_repetitive_patterns()`: Require higher repetition_ratio before flagging
  - [ ] 4.4 Update texture internal weights if needed
  - [ ] 4.5 Test on real photo - target texture score: 0.35-0.50
  - [ ] 4.6 Test on AI image - verify texture score: 0.60-0.85

- [ ] **5.0 Implement Evidence Classification System**
  - [ ] 5.1 Create `aidetect/analysis/evidence.py` with EvidenceType enum
  - [ ] 5.2 Define `EvidenceType.POSITIVE_AI` (software tags, periodic artifacts, etc.)
  - [ ] 5.3 Define `EvidenceType.NEGATIVE_AI` (camera signatures, GPS, ICC, natural artifacts)
  - [ ] 5.4 Define `EvidenceType.NEUTRAL` (missing data, inconclusive)
  - [ ] 5.5 Implement `classify_mechanism_evidence()` to categorize each MechanismResult
  - [ ] 5.6 Implement `has_positive_ai_evidence()` - check for strong AI signals
  - [ ] 5.7 Implement `count_agreeing_mechanisms()` - count mechanisms above threshold
  - [ ] 5.8 Implement `build_detection_rationale()` - explain what triggered verdict
  - [ ] 5.9 Export evidence functions from `analysis/__init__.py`
  - [ ] 5.10 Add unit tests for evidence classification

- [ ] **6.0 Implement Conservative Detection Mode**
  - [ ] 6.1 Add to `AnalysisResult` dataclass: `confidence_level: str` field (HIGH/MEDIUM/LOW)
  - [ ] 6.2 Add to `AnalysisResult` dataclass: `detection_rationale: List[str]` field
  - [ ] 6.3 Add to `AppConfig`: `conservative_mode: bool = False`
  - [ ] 6.4 Add to `AppConfig`: `ai_verdict_threshold: float = 0.5` (normal) or 0.65 (conservative)
  - [ ] 6.5 Update `calibrate_and_decide()` signature to accept `config: AppConfig`
  - [ ] 6.6 In `calibrate_and_decide()`: Use evidence system to check for positive AI signals
  - [ ] 6.7 In `calibrate_and_decide()`: In conservative mode, require 2+ mechanisms >0.6
  - [ ] 6.8 In `calibrate_and_decide()`: In conservative mode, use threshold of 0.65 instead of 0.5
  - [ ] 6.9 In `calibrate_and_decide()`: Add confidence_level based on agreement strength
  - [ ] 6.10 In `calibrate_and_decide()`: Build detection_rationale list
  - [ ] 6.11 Add `--conservative` flag to `cli.py` that sets `config.conservative_mode = True`
  - [ ] 6.12 Update runner to pass updated config to calibrate_and_decide()
  - [ ] 6.13 Test: Real photo in normal mode (target: <40%)
  - [ ] 6.14 Test: Real photo in conservative mode (target: <30%)
  - [ ] 6.15 Test: AI image in normal mode (target: 55-80%)
  - [ ] 6.16 Test: AI image in conservative mode (target: 50-75%, with HIGH confidence if obvious)

- [ ] **7.0 Add Transparency and Explainability**
  - [ ] 7.1 Update `format_text_report()` to show confidence level (HIGH/MEDIUM/LOW)
  - [ ] 7.2 Update `format_text_report()` to show detection rationale (top 3 reasons)
  - [ ] 7.3 Update JSON schema to include confidence_level and detection_rationale
  - [ ] 7.4 Add `--explain` flag to CLI for verbose output showing all sub-scores
  - [ ] 7.5 Create `docs/scoring-philosophy.md` explaining the evidence-based approach
  - [ ] 7.6 Update README with section on interpretation and conservative mode

- [ ] **8.0 Validation with Real Test Cases**
  - [ ] 8.1 Test with YOUR photo - document before/after score
  - [ ] 8.2 Test with 4 more real photos (phone, DSLR, diverse scenarios)
  - [ ] 8.3 Test with 5 AI-generated images from your collection
  - [ ] 8.4 Create comparison table: [Image Type] | Normal Mode | Conservative Mode
  - [ ] 8.5 Verify: Real photos average <40% in normal mode, <30% in conservative
  - [ ] 8.6 Verify: AI images average 55-80% in normal mode, 50-75% in conservative
  - [ ] 8.7 Verify: AI images with signatures 80-95% in both modes
  - [ ] 8.8 Document any edge cases or failures
  - [ ] 8.9 Adjust thresholds if needed based on real results
  - [ ] 8.10 Update `docs/accuracy-phase1-results.md` with actual validation results

- [ ] **9.0 Rebuild and Final Verification**
  - [ ] 9.1 Rebuild executable with corrected scoring: `.\build.bat`
  - [ ] 9.2 Test YOUR photo with new build (expect LIKELY_HUMAN or <40%)
  - [ ] 9.3 Test known AI image (expect AI-GENERATED with 60-80%)
  - [ ] 9.4 Test YOUR photo with --conservative (expect <30%)
  - [ ] 9.5 Test AI image with --conservative (expect still detectable if obvious)
  - [ ] 9.6 Verify performance still within budget (<15s)
  - [ ] 9.7 Update all documentation with final results
  - [ ] 9.8 Update version to v0.2.1-balanced
  - [ ] 9.9 Tag completion in tasks file

---

## Implementation Priority

### Phase A: Scoring Adjustments (Tasks 1.0-4.0)
**Time**: 1-2 hours  
**Impact**: Immediate improvement  
**Focus**: Make absent data neutral, keep AI evidence aggressive

### Phase B: Evidence System (Task 5.0)
**Time**: 1 hour  
**Impact**: Structural improvement  
**Focus**: Classify evidence types, require positive signals

### Phase C: Conservative Mode (Task 6.0)
**Time**: 1 hour  
**Impact**: User control  
**Focus**: Add --conservative flag for high-stakes decisions

### Phase D: Testing & Validation (Tasks 7.0-9.0)
**Time**: 2-3 hours  
**Impact**: Validation  
**Focus**: Ensure both real AND AI detection work correctly

---

## Success Metrics

### Before Corrections
- Real photo: ~65% AI âŒ FALSE POSITIVE
- AI image: 55-70% AI âœ… CORRECT

### After Corrections (Target)
- Real photo: <40% AI âœ… CORRECT
- Real photo (conservative): <30% AI âœ… HIGH CONFIDENCE
- AI image: 55-80% AI âœ… MAINTAINED
- AI image with signatures: 80-95% AI âœ… IMPROVED

---

**Next Sight** | www.next-sight.com | info@next-sight.com
