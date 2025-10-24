# Task List: False Positive Reduction & Calibration

**Goal**: Reduce false positives on real photos while maintaining or improving AI detection accuracy  
**Constraint**: AI detection must remain 55-70%+ ; Real photo detection must improve to <40%  
**Approach**: Smarter scoring logic + conservative confidence mode

---

## Relevant Files

### Files to Modify
- `aidetect/analysis/metadata.py` - Adjust scoring to be neutral on missing data, aggressive on AI signatures
- `aidetect/analysis/quality.py` - Account for modern smartphone photography characteristics
- `aidetect/analysis/frequency.py` - Fine-tune thresholds to reduce false positives
- `aidetect/analysis/texture.py` - Adjust uniformity thresholds for natural variation
- `aidetect/analysis/calibration.py` - Add conservative mode and multi-signal requirements
- `aidetect/core/types.py` - Add ConfidenceLevel enum and detection rationale fields
- `aidetect/core/config.py` - Add conservative_mode flag and detection_threshold settings
- `aidetect/cli.py` - Add --conservative flag to CLI
- `aidetect/runner/single.py` - Pass conservative mode to calibration
- `README.md` - Document conservative mode usage
- `USAGE.md` - Add examples with --conservative flag

### New Files to Create
- `aidetect/analysis/evidence.py` - Evidence aggregation and multi-signal validation logic
- `tests/test_calibration.py` - Unit tests for new calibration logic
- `docs/calibration-guide.md` - Documentation on scoring philosophy and thresholds

### Notes
- **Critical**: Do NOT weaken AI detection (maintain 55-70%+ on AI images)
- **Goal**: Improve real photo scores to <40% AI probability
- **Strategy**: Be neutral on "absence of data", aggressive on "presence of AI evidence"
- **Validation**: Test with both real photos AND AI images before/after

---

## Tasks

- [ ] **1.0 Rebalance Metadata Scoring for Real Photos**
  - [ ] 1.1 Change missing GPS from suspicious (0.7) to neutral (0.5) - GPS rare in cameras
  - [ ] 1.2 Change missing Make/Model from 0.7 to 0.6 - be slightly less harsh
  - [ ] 1.3 Change no ICC profile from 0.6 to 0.5 - many phones lack ICC
  - [ ] 1.4 Keep EXIF completeness scoring but adjust weights to be less punishing
  - [ ] 1.5 Make timestamp consistency more lenient (allow up to 90 days instead of 30)
  - [ ] 1.6 **CRITICAL**: Keep AI software signature detection at 0.98 (high confidence)
  - [ ] 1.7 **CRITICAL**: Keep known AI generator camera model at 0.95 (definitive)
  - [ ] 1.8 Add "smartphone" detection - if Make contains "Apple/Samsung/Google", reduce score
  - [ ] 1.9 Test metadata scoring on real photo (target: 0.3-0.4 AI probability)
  - [ ] 1.10 Test metadata scoring on AI image (verify still: 0.6-0.9 AI probability)

- [ ] **2.0 Adjust Quality Metrics for Modern Photography**
  - [ ] 2.1 Update sharpness thresholds to account for modern smartphone sharpening
  - [ ] 2.2 Change "no chromatic aberration" scoring - modern phones correct CA digitally
  - [ ] 2.3 Make vignetting absence less suspicious (many phones correct it)
  - [ ] 2.4 Adjust oversaturation thresholds - allow for vibrant smartphone processing
  - [ ] 2.5 Keep edge uniformity detection (still valid for AI detection)
  - [ ] 2.6 Add "natural imperfection" bonus - sensor noise, lens artifacts
  - [ ] 2.7 Test quality metrics on real photo (target: 0.3-0.5 AI probability)
  - [ ] 2.8 Test quality metrics on AI image (verify still: 0.4-0.7 AI probability)

- [ ] **3.0 Fine-Tune Frequency Analysis Thresholds**
  - [ ] 3.1 Increase Benford's Law deviation threshold before flagging as suspicious
  - [ ] 3.2 Adjust quantization artifact threshold (natural JPEG compression varies)
  - [ ] 3.3 Make noise spectrum scoring more conservative (high-end phones are very clean)
  - [ ] 3.4 Keep periodic artifact detection aggressive (this is AI-specific)
  - [ ] 3.5 Reweight internal frequency sub-scores to prioritize AI-specific signals
  - [ ] 3.6 Test frequency on real photo (target: 0.3-0.5 AI probability)
  - [ ] 3.7 Test frequency on AI image (verify still: 0.5-0.8 AI probability)

- [ ] **4.0 Calibrate Texture Analysis**
  - [ ] 4.1 Adjust texture consistency threshold (real photos CAN have uniform regions like sky/walls)
  - [ ] 4.2 Make repetitive pattern detection more strict (require stronger evidence)
  - [ ] 4.3 Add "natural texture" detection - organic randomness bonus
  - [ ] 4.4 Test texture on real photo (target: 0.3-0.5 AI probability)
  - [ ] 4.5 Test texture on AI image (verify still: 0.6-0.9 AI probability)

- [ ] **5.0 Implement Evidence-Based Detection System**
  - [ ] 5.1 Create new file `aidetect/analysis/evidence.py` with evidence aggregation logic
  - [ ] 5.2 Define "positive AI evidence" vs "absence of camera evidence"
  - [ ] 5.3 Implement `classify_evidence()` to categorize each mechanism result
  - [ ] 5.4 Implement `require_positive_evidence()` - need at least 1 strong AI signal
  - [ ] 5.5 Implement `multi_signal_consensus()` - require 2+ mechanisms >0.6 for high confidence
  - [ ] 5.6 Update `calibrate_and_decide()` to use evidence system
  - [ ] 5.7 Add detailed rationale field explaining WHY image was flagged
  - [ ] 5.8 Export evidence functions from `aidetect/analysis/__init__.py`

- [ ] **6.0 Implement Conservative Mode**
  - [ ] 6.1 Add `conservative_mode: bool = False` to `AppConfig`
  - [ ] 6.2 Add `detection_threshold: float = 0.5` to `AppConfig`
  - [ ] 6.3 Add `--conservative` flag to CLI in `aidetect/cli.py`
  - [ ] 6.4 Update `calibrate_and_decide()` to accept conservative mode parameter
  - [ ] 6.5 In conservative mode, raise threshold to 0.65 for AI-GENERATED verdict
  - [ ] 6.6 In conservative mode, require positive evidence (not just absence)
  - [ ] 6.7 In conservative mode, require consensus from 2+ mechanisms
  - [ ] 6.8 Add `confidence_level` field to AnalysisResult (HIGH/MEDIUM/LOW)
  - [ ] 6.9 Update text report to show confidence level
  - [ ] 6.10 Test conservative mode on real photo (should be LIKELY_HUMAN)
  - [ ] 6.11 Test conservative mode on AI image (should still detect if obvious)

- [ ] **7.0 Add Scoring Transparency and Debugging**
  - [ ] 7.1 Add `score_breakdown` field to AnalysisResult showing per-mechanism contribution
  - [ ] 7.2 Add `detection_reasons` list explaining what triggered AI verdict
  - [ ] 7.3 Add `human_indicators` list showing what suggests real photo
  - [ ] 7.4 Update JSON output to include full scoring breakdown
  - [ ] 7.5 Add `--explain` flag to show detailed scoring rationale
  - [ ] 7.6 Update text reporter to show top 3 AI indicators and top 3 human indicators
  - [ ] 7.7 Document scoring philosophy in `docs/scoring-philosophy.md`

- [ ] **8.0 Validation and Testing**
  - [ ] 8.1 Test with 5 real photos (smartphones, DSLRs, diverse scenarios)
  - [ ] 8.2 Document before/after scores for each real photo
  - [ ] 8.3 Test with 5 AI images (ensure detection didn't degrade)
  - [ ] 8.4 Document before/after scores for each AI image
  - [ ] 8.5 Verify real photo average drops to <40% AI probability
  - [ ] 8.6 Verify AI image average stays at 55-80% AI probability
  - [ ] 8.7 Test conservative mode on all 10 images
  - [ ] 8.8 Create comparison table showing default vs conservative mode results
  - [ ] 8.9 Update `docs/accuracy-phase1-results.md` with actual test results
  - [ ] 8.10 Document recommended usage: default for screening, conservative for court evidence

- [ ] **9.0 Rebuild and Smoke Test**
  - [ ] 9.1 Rebuild executable with updated scoring logic
  - [ ] 9.2 Test on YOUR photo (should show LIKELY_HUMAN or <40%)
  - [ ] 9.3 Test on known AI image (should still detect >55%)
  - [ ] 9.4 Test conservative mode on both
  - [ ] 9.5 Update README with guidance on when to use conservative mode
  - [ ] 9.6 Update USAGE.md with before/after examples

---

## Scoring Philosophy Changes

### BEFORE (Current - Too Aggressive)
```
Missing GPS          â†’ 0.5-0.7 AI probability
Missing camera data  â†’ 0.7 AI probability  
No ICC profile       â†’ 0.6 AI probability
Modern sharp photo   â†’ 0.6 AI probability
No vignetting        â†’ 0.7 AI probability

= Real photo gets ~60-70% AI score âŒ FALSE POSITIVE
```

### AFTER (Proposed - Balanced)
```
Missing GPS          â†’ 0.5 AI probability (neutral)
Missing camera data  â†’ 0.55 AI probability (slight suspicion)
No ICC profile       â†’ 0.5 AI probability (neutral)
Modern sharp photo   â†’ 0.4 AI probability (expected for phones)
No vignetting        â†’ 0.5 AI probability (phones correct it)

+ POSITIVE AI EVIDENCE required:
  - AI software tag  â†’ 0.95 AI probability âœ…
  - Known generator  â†’ 0.95 AI probability âœ…
  - Periodic artifacts â†’ 0.7-0.8 AI probability âœ…
  
= Real photo gets ~35-45% AI score âœ… CORRECT
= AI image with signatures gets 75-95% âœ… CORRECT  
= AI image without metadata still gets 55-70% âœ… CORRECT
```

---

## Success Criteria

### Real Photos (Improve)
- **Current**: ~60-70% AI probability (FALSE POSITIVE)
- **Target**: <40% AI probability
- **With Conservative Mode**: <30% AI probability

### AI Images (Maintain)
- **Current**: 55-70% AI probability
- **Target**: 55-80% AI probability (maintain or improve)
- **With Obvious Signatures**: 80-95% AI probability

---

**Should I proceed with implementing these fixes?** This will make the system much more balanced and reduce false positives significantly.

**Next Sight** | www.next-sight.com | info@next-sight.com
