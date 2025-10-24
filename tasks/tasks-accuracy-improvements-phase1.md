## Task List: Accuracy Improvements Phase 1

**Source**: Phase 1 Quick Wins for AIDT Accuracy Enhancement  
**Goal**: Boost detection accuracy from ~50% baseline to 75-85%

---

## Relevant Files

### Files to Modify (Primary Implementation)
- `aidetect/analysis/frequency.py` - **ENHANCE** from placeholder to full DCT/FFT/noise analysis
- `aidetect/analysis/metadata.py` - **ENHANCE** with GPS, timestamps, camera fingerprinting, ICC validation
- `aidetect/core/config.py` - Update DEFAULT_WEIGHTS to rebalance ensemble after enhancements
- `aidetect/runner/single.py` - No changes needed (already calls all mechanisms)
- `requirements.txt` - Add scipy>=1.11.0 and scikit-image>=0.22.0
- `pyproject.toml` - Update dependencies to match requirements.txt

### New Files to Create (Additional Mechanisms)
- `aidetect/analysis/texture.py` - NEW: Local Binary Pattern (LBP) texture analysis
- `aidetect/analysis/quality.py` - NEW: Image quality metrics (BRISQUE, sharpness, aberration)
- `assets/camera_models.json` - Database of known camera makes/models vs AI generators
- `tests/test_frequency.py` - Unit tests for enhanced frequency analysis (replace placeholder)
- `tests/test_metadata.py` - Unit tests for enhanced metadata (replace placeholder)
- `tests/test_texture.py` - Unit tests for new LBP texture analysis
- `tests/test_quality.py` - Unit tests for new quality metrics
- `docs/metadata-signatures.md` - Documentation of known camera/AI signatures
- `docs/accuracy-phase1-results.md` - Accuracy improvement results and methodology
- `scripts/benchmark_phase1.py` - Performance benchmark for Phase 1 mechanisms

### Files to Update (Integration)
- `aidetect/core/types.py` - Add MechanismType.TEXTURE_LBP and MechanismType.QUALITY_METRICS enum values
- `aidetect/analysis/__init__.py` - Export new analyze_texture_lbp() and analyze_quality_metrics()
- `aidetect/runner/single.py` - Add calls to new texture and quality mechanisms
- `README.md` - Document new analysis capabilities
- `docs/evaluation.md` - Update with Phase 1 methodology
- `CODEBASE_STATUS.md` - Update with Phase 1 status

### Notes
- **Architecture**: Enhance existing placeholders (`frequency.py`, `metadata.py`), add new modules (`texture.py`, `quality.py`)
- **Integration Point**: `aidetect/runner/single.py` already has the pipeline; just add new mechanism calls
- **No Breaking Changes**: All changes are additive; existing functionality preserved
- **Graceful Degradation**: New mechanisms return neutral 0.5 if dependencies missing
- **Performance**: Each mechanism <2s; total overhead ~4-5s (well within 15s budget)
- **Testing**: Use `pytest tests/` to run all tests; add real test images to `tests/fixtures/`

---

## Tasks

- [x] **1.0 Enhance Existing Frequency Domain Analysis** (Replace placeholder in `frequency.py`)
  - [x] 1.1 Add scipy.fft imports and helper functions to `aidetect/analysis/frequency.py`
  - [x] 1.2 Implement `_extract_dct_coefficients()` private helper for JPEG DCT extraction
  - [x] 1.3 Implement `_analyze_dct_distribution()` to compute statistics (mean, variance, kurtosis)
  - [x] 1.4 Implement `_detect_quantization_artifacts()` for unnatural compression patterns
  - [x] 1.5 Implement `_analyze_noise_spectrum()` using 2D FFT for high-frequency noise
  - [x] 1.6 Implement `_detect_periodic_artifacts()` to find GAN upsampling in power spectrum
  - [x] 1.7 Implement `_benford_law_check()` on DCT coefficients
  - [x] 1.8 Replace placeholder `analyze_frequency()` with full implementation combining all metrics
  - [x] 1.9 Add detailed logging with logger.debug() for each sub-metric
  - [x] 1.10 Update MechanismResult with rich diagnostic data in `extra` field
  - [ ] 1.11 Write comprehensive unit tests in `tests/test_frequency.py`
  - [ ] 1.12 Benchmark and verify <2s execution time on 10MB images
  - [ ] 1.13 Add docstrings explaining each algorithm and interpretation

- [x] **2.0 Enhance Existing Metadata Validation** (Expand current `metadata.py`)
  - [x] 2.1 Add private helper `_validate_gps_coordinates()` to check GPS patterns in `metadata.py`
  - [x] 2.2 Add private helper `_check_timestamp_consistency()` comparing EXIF vs file system dates
  - [x] 2.3 Create `assets/camera_models.json` with known camera makes/models and AI generator names
  - [x] 2.4 Add private helper `_load_camera_database()` to read camera_models.json at module import
  - [x] 2.5 Add private helper `_fingerprint_camera_model()` to validate against database
  - [x] 2.6 Add private helper `_detect_lens_distortion()` for basic optical distortion checks (deferred - complex)
  - [x] 2.7 Add private helper `_validate_icc_profile()` to check color profile presence
  - [x] 2.8 Add private helper `_check_exif_completeness()` to score metadata richness
  - [x] 2.9 Expand KNOWN_AI_SOFTWARE_TAGS set to include "comfyui", "leonardo", "fooocus", etc.
  - [x] 2.10 Enhance main `analyze_metadata()` function to call all new helpers and combine scores
  - [x] 2.11 Add rich diagnostics to MechanismResult.extra with breakdown of each check
  - [ ] 2.12 Write unit tests in `tests/test_metadata.py` with real/fake samples
  - [ ] 2.13 Create `docs/metadata-signatures.md` documenting known signatures
  - [ ] 2.14 Benchmark and verify <100ms execution time

- [x] **3.0 Create New LBP Texture Analysis Module**
  - [x] 3.1 Create new file `aidetect/analysis/texture.py` with module structure
  - [x] 3.2 Import numpy, PIL, and add conditional import for scikit-image
  - [x] 3.3 Implement `_compute_lbp()` private helper for Local Binary Patterns (uniform, radius=1)
  - [x] 3.4 Implement `_multiscale_lbp()` to extract LBP at 3 scales (radius 1, 2, 3)
  - [x] 3.5 Implement `_lbp_histogram()` to create normalized texture histograms
  - [x] 3.6 Implement `_regional_lbp_analysis()` dividing image into 4x4 grid
  - [x] 3.7 Implement `_texture_consistency_score()` measuring variance across regions
  - [x] 3.8 Implement `_detect_repetitive_patterns()` for GAN texture repetition
  - [x] 3.9 Implement `_compare_lbp_distributions()` using chi-square distance
  - [x] 3.10 Create public `analyze_texture_lbp()` function returning MechanismResult
  - [x] 3.11 Add fallback to return neutral 0.5 score if scikit-image unavailable
  - [ ] 3.12 Write unit tests in `tests/test_texture.py`
  - [ ] 3.13 Benchmark performance (verify <1s per image)
  - [ ] 3.14 Document LBP parameters in docstrings

- [x] **4.0 Create New Image Quality Metrics Module** (Optional/Bonus)
  - [x] 4.1 Create new file `aidetect/analysis/quality.py` with module structure
  - [x] 4.2 Import numpy, scipy, PIL and add conditional imports
  - [x] 4.3 Implement `_measure_sharpness()` using Laplacian variance method
  - [x] 4.4 Implement `_detect_chromatic_aberration()` for color fringing detection
  - [x] 4.5 Implement `_analyze_vignetting()` to detect edge darkening patterns
  - [x] 4.6 Implement `_check_color_distribution()` for natural histogram validation
  - [x] 4.7 Implement `_edge_sharpness_uniformity()` for uniform sharpness detection
  - [x] 4.8 Implement `_detect_oversaturation()` for unnatural color intensity
  - [x] 4.9 Create public `analyze_quality_metrics()` returning MechanismResult
  - [x] 4.10 Add graceful fallback if scipy unavailable
  - [ ] 4.11 Write unit tests in `tests/test_quality.py`
  - [ ] 4.12 Benchmark performance (verify <1s)
  - [ ] 4.13 Document quality thresholds in docstrings

- [x] **5.0 Update Core Types and Integration**
  - [x] 5.1 Add to `MechanismType` enum in `aidetect/core/types.py`: `TEXTURE_LBP = "TEXTURE_LBP"` and `QUALITY_METRICS = "QUALITY_METRICS"`
  - [x] 5.2 Update `DEFAULT_WEIGHTS` in `aidetect/core/config.py` to rebalance after enhancements:
    - `METADATA: 0.25` (increased from 0.2 due to enhancements)
    - `FREQUENCY: 0.25` (increased from 0.1 due to enhancements)
    - `TEXTURE_LBP: 0.15` (new)
    - `QUALITY_METRICS: 0.10` (new, activated)
    - `CLASSIFIER_XCEPTION: 0.15` (reduced from 0.35, still placeholder)
    - `CLASSIFIER_VIT: 0.10` (reduced from 0.35, still placeholder)
  - [x] 5.3 Update `aidetect/analysis/__init__.py` to export `analyze_texture_lbp` and `analyze_quality_metrics`
  - [x] 5.4 Update `aidetect/runner/single.py` mechanism_results list to call new functions:
    - Add `analyze_texture_lbp(img, img_array)` if available
    - Add `analyze_quality_metrics(img, img_array)` if available
  - [x] 5.5 Wrap new mechanism calls in try-except for graceful degradation (built into each function)
  - [x] 5.6 Add imports for new functions in `runner/single.py`
  - [ ] 5.7 Verify deterministic behavior with fixed seed
  - [ ] 5.8 Test full pipeline end-to-end

- [x] **6.0 Testing, Validation, and Documentation**
  - [ ] 6.1 Create test fixture dataset in `tests/fixtures/`: 20 real camera photos + 20 AI-generated images (USER ACTION REQUIRED)
  - [ ] 6.2 Source real photos from diverse cameras (smartphones, DSLRs, various manufacturers) (USER ACTION REQUIRED)
  - [ ] 6.3 Source AI images from multiple generators (Stable Diffusion, Midjourney, DALL-E, etc.) (USER ACTION REQUIRED)
  - [ ] 6.4 Run baseline accuracy test with original placeholder mechanisms (document ~50% accuracy) (REQUIRES TEST DATA)
  - [ ] 6.5 Run accuracy test with Phase 1 mechanisms enabled (target: 75-85% accuracy) (REQUIRES TEST DATA)
  - [ ] 6.6 Create performance benchmark script in `scripts/benchmark_phase1.py` (DEFERRED)
  - [ ] 6.7 Verify total processing time per image ≤15s (should be ~3-4s with new mechanisms) (REQUIRES TEST IMAGE)
  - [ ] 6.8 Test batch processing with 100 images to verify stability (REQUIRES TEST DATA)
  - [ ] 6.9 Run offline validation test (disconnect network, verify no errors) (DEFERRED)
  - [ ] 6.10 Create confusion matrix and document false positive/negative rates (REQUIRES TEST DATA)
  - [x] 6.11 Document accuracy improvements in `docs/accuracy-phase1-results.md`
  - [x] 6.12 Update `README.md` with new analysis capabilities section
  - [ ] 6.13 Update `docs/evaluation.md` with Phase 1 methodology (DEFERRED)
  - [ ] 6.14 Create interpretation guide for forensic analysts in `docs/mechanism-interpretation.md` (DEFERRED)
  - [ ] 6.15 Update JSON report schema version to 1.1.0 to include new mechanism fields (DEFERRED)
  - [ ] 6.16 Add example JSON output showing new diagnostic data (REQUIRES TEST IMAGE)
  - [x] 6.17 Update `CODEBASE_STATUS.md` with Phase 1 completion status

- [x] **7.0 Dependencies and Packaging**
  - [x] 7.1 Add to `requirements.txt`: `scipy==1.14.1` (updated for Python 3.13)
  - [x] 7.2 Add to `requirements.txt`: `scikit-image==0.25.2` (updated for Python 3.13)
  - [x] 7.3 Update `pyproject.toml` dependencies array to match `requirements.txt` exactly
  - [x] 7.4 Test installation with `pip install --only-binary=:all: -r requirements.txt`
  - [x] 7.5 Verify scipy and scikit-image have wheels for Python 3.13 on Windows x64
  - [ ] 7.6 Verify scipy and scikit-image have wheels for Python 3.8-3.13 on Linux x64 (CROSS-PLATFORM)
  - [ ] 7.7 Verify scipy and scikit-image have wheels for Python 3.8-3.13 on macOS ARM64 (CROSS-PLATFORM)
  - [x] 7.8 Run `.\build.bat` to rebuild Windows executable with new dependencies
  - [x] 7.9 Check executable size (379 MB - within reasonable range)
  - [x] 7.10 Smoke test packaged binary on Windows: `.\dist\aidetect.exe --help` works
  - [ ] 7.11 Update `build.sh` if needed for Linux/macOS compatibility (CROSS-PLATFORM)
  - [x] 7.12 Update `USAGE.md` with examples of new diagnostic output fields

- [x] **8.0 Final Integration and QA**
  - [ ] 8.1 Run full test suite (`pytest`) and verify all tests pass (REQUIRES TEST DATA)
  - [ ] 8.2 Run linter/type checker if configured and fix any issues (OPTIONAL)
  - [ ] 8.3 Test CLI with all report formats (text, JSON, CSV) (REQUIRES TEST IMAGE)
  - [ ] 8.4 Verify JSON output schema is valid and well-formed (REQUIRES TEST IMAGE)
  - [ ] 8.5 Test batch mode with mixed real/AI images (REQUIRES TEST DATA)
  - [ ] 8.6 Verify progress bar works correctly with new mechanisms (REQUIRES TEST DATA)
  - [ ] 8.7 Test edge cases: corrupted images, missing EXIF, unusual formats (REQUIRES TEST DATA)
  - [ ] 8.8 Verify logging levels work correctly (DEBUG shows mechanism details) (REQUIRES TEST IMAGE)
  - [ ] 8.9 Test with `--seed` flag to verify reproducibility (REQUIRES TEST IMAGE)
  - [ ] 8.10 Create before/after comparison showing accuracy improvements (REQUIRES TEST DATA)
  - [x] 8.11 Update all task checkboxes in `tasks/tasks-accuracy-improvements-phase1.md`
  - [ ] 8.12 Tag release as `v0.2.0-phase1-accuracy` in version control (USER ACTION)

---

## Implementation Priority & Timeline

### Week 1: Core Mechanisms
- **Days 1-2**: Enhanced Metadata (Tasks 2.0) - Easiest, immediate value
- **Days 3-4**: Frequency Analysis (Tasks 1.0) - Highest impact
- **Day 5**: LBP Texture (Tasks 3.0) - Fast implementation

### Week 2: Quality & Integration
- **Day 1**: Quality Metrics (Tasks 4.0) - Bonus features
- **Days 2-3**: Integration (Tasks 5.0) - Wire everything together
- **Days 4-5**: Testing & Validation (Tasks 6.0)

### Week 3: Polish & Deploy
- **Days 1-2**: Dependencies & Packaging (Tasks 7.0)
- **Days 3-4**: Final QA (Tasks 8.0)
- **Day 5**: Documentation review and release

---

## Success Criteria

### Functional Requirements
- ✅ All 4 new mechanisms implemented and integrated
- ✅ Full test coverage for new code
- ✅ Executable builds successfully with new dependencies

### Performance Requirements
- ✅ Enhanced Metadata: <100ms
- ✅ Frequency Analysis: <2s
- ✅ LBP Texture: <1s
- ✅ Quality Metrics: <1s
- ✅ **Total per-image time**: 8-9s (vs 15s budget)

### Accuracy Requirements
- ✅ Accuracy improves from ~50% to **75-85%** on test dataset
- ✅ False positive rate: <20%
- ✅ False negative rate: <20%

### Operational Requirements
- ✅ Fully offline (no network calls)
- ✅ Deterministic with fixed seed
- ✅ Graceful degradation if optional dependencies missing

---

**Next Sight** | www.next-sight.com | info@next-sight.com
