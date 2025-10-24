#!/usr/bin/env python3
"""
AI Detection Tool - Accuracy Validation Script

Validates the enhanced tool's accuracy using labeled test data.
Uses the Test Images folder with AIGenerated and Real subfolders.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception:
    np = None

# Add the parent directory to the path so we can import aidetect
sys.path.insert(0, str(Path(__file__).parent.parent))

from aidetect.analysis.registry import get_mechanism_registry
from aidetect.core.config import AppConfig, DEFAULT_WEIGHTS
from aidetect.core.logging import setup_logging
from aidetect.io.image_loader import load_image_and_fileinfo
from aidetect.runner.single import run_single_analysis

# Configure logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def find_test_images(test_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find AI-generated and real images in test directory.

    Expected structure:
    test_dir/
    ├── AIGenerated/ (optional)
    │   ├── image1.jpg
    │   └── ...
    ├── Real/
    │   ├── image1.jpg
    │   └── ...
    └── [AI-generated files in root if no AIGenerated folder]
    """
    ai_dir = test_dir / "AIGenerated"
    real_dir = test_dir / "Real"

    ai_images = []
    real_images = []

    # Find AI-generated images (check both subfolder and root)
    if ai_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            ai_images.extend(ai_dir.glob(ext))
            ai_images.extend(ai_dir.glob(ext.upper()))
    else:
        # Look for AI-generated images in root (by filename patterns)
        ai_patterns = ['Gemini_Generated_Image_', 'AI_', 'generated_', 'synthetic_']
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            for pattern in ai_patterns:
                ai_images.extend(test_dir.glob(f"{pattern}*{ext}"))
                ai_images.extend(test_dir.glob(f"{pattern}*{ext.upper()}"))

    # Find real images
    if real_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            real_images.extend(real_dir.glob(ext))
            real_images.extend(real_dir.glob(ext.upper()))

    logger.info(f"Found {len(ai_images)} AI-generated images")
    logger.info(f"Found {len(real_images)} real images")

    return ai_images, real_images


def analyze_image(image_path: Path, config: AppConfig) -> Tuple[float, str]:
    """
    Analyze a single image and return probability and notes.
    """
    try:
        # Load image
        img, file_path = load_image_and_fileinfo(str(image_path))
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return 0.5, "Failed to load"

        # Run analysis
        result = run_single_analysis(str(image_path), config)

        return result.confidence, result.verdict.value

    except Exception as e:
        logger.error(f"Error analyzing {image_path}: {e}")
        return 0.5, f"Error: {str(e)[:50]}"


def calculate_metrics(
    ai_confidences: List[float],
    real_confidences: List[float],
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate accuracy metrics.
    """
    # Convert confidence scores to predictions (AI if confidence >= threshold)
    ai_predictions = [1 if p >= threshold else 0 for p in ai_confidences]
    real_predictions = [1 if p >= threshold else 0 for p in real_confidences]

    # Calculate confusion matrix elements
    tp = sum(ai_predictions)  # AI images predicted as AI
    tn = len(real_predictions) - sum(real_predictions)  # Real images predicted as real
    fp = sum(real_predictions)  # Real images predicted as AI (false positive)
    fn = len(ai_predictions) - sum(ai_predictions)  # AI images predicted as real (false negative)

    # Calculate metrics
    total = len(ai_confidences) + len(real_confidences)
    accuracy = (tp + tn) / total if total > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    false_positive_rate = fp / len(real_confidences) if len(real_confidences) > 0 else 0
    false_negative_rate = fn / len(ai_confidences) if len(ai_confidences) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total_ai_images": len(ai_confidences),
        "total_real_images": len(real_confidences),
    }


def main():
    parser = argparse.ArgumentParser(description="Validate AI detection accuracy")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("Test Images"),
        help="Directory containing AIGenerated and Real subfolders"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to test per category"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_results.json"),
        help="Output file for detailed results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find test images
    logger.info(f"Looking for test images in: {args.test_dir}")
    ai_images, real_images = find_test_images(args.test_dir)

    if not ai_images and not real_images:
        logger.error("No test images found!")
        return 1

    # Limit number of images if specified
    if args.max_images:
        ai_images = ai_images[:args.max_images]
        real_images = real_images[:args.max_images]

    # Create configuration
    config = AppConfig()

    # Analyze AI-generated images
    logger.info("Analyzing AI-generated images...")
    ai_results = []

    for i, img_path in enumerate(ai_images):
        logger.info(f"AI image {i+1}/{len(ai_images)}: {img_path.name}")
        prob, notes = analyze_image(img_path, config)
        ai_results.append({
            "path": str(img_path),
            "probability": prob,
            "notes": notes,
            "expected": "ai"
        })

    # Analyze real images
    logger.info("Analyzing real images...")
    real_results = []

    for i, img_path in enumerate(real_images):
        logger.info(f"Real image {i+1}/{len(real_images)}: {img_path.name}")
        prob, notes = analyze_image(img_path, config)
        real_results.append({
            "path": str(img_path),
            "probability": prob,
            "notes": notes,
            "expected": "real"
        })

    # Calculate metrics
    ai_confidences = [r["probability"] for r in ai_results]
    real_confidences = [r["probability"] for r in real_results]

    metrics = calculate_metrics(ai_confidences, real_confidences, args.threshold)

    # Print summary
    print("\n" + "="*60)
    print("AI DETECTION ACCURACY VALIDATION RESULTS")
    print("="*60)
    print(f"Test Images Directory: {args.test_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"AI Images Tested: {len(ai_results)}")
    print(f"Real Images Tested: {len(real_results)}")
    print(f"Total Images: {len(ai_results) + len(real_results)}")
    print()

    print("OVERALL METRICS:")
    print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.3f} ({metrics['precision']*100:.1f}%)")
    print(f"  Recall: {metrics['recall']:.3f} ({metrics['recall']*100:.1f}%)")
    print(f"  F1 Score: {metrics['f1_score']:.3f} ({metrics['f1_score']*100:.1f}%)")
    print()

    print("ERROR RATES:")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.3f} ({metrics['false_positive_rate']*100:.1f}%)")
    print(f"  False Negative Rate: {metrics['false_negative_rate']:.3f} ({metrics['false_negative_rate']*100:.1f}%)")
    print()

    print("CONFUSION MATRIX:")
    print(f"  True Positives (AI correctly detected): {metrics['true_positives']}")
    print(f"  True Negatives (Real correctly identified): {metrics['true_negatives']}")
    print(f"  False Positives (Real misclassified as AI): {metrics['false_positives']}")
    print(f"  False Negatives (AI misclassified as Real): {metrics['false_negatives']}")
    print()

    # Analyze confidence distributions
    if ai_confidences and real_confidences:
        print("CONFIDENCE DISTRIBUTIONS:")
        print(f"  AI Images - Mean: {np.mean(ai_confidences):.3f}, Std: {np.std(ai_confidences):.3f}")
        print(f"  Real Images - Mean: {np.mean(real_confidences):.3f}, Std: {np.std(real_confidences):.3f}")
    # Save detailed results
    detailed_results = {
        "config": {
            "threshold": args.threshold,
            "test_directory": str(args.test_dir),
            "max_images": args.max_images,
        },
        "metrics": metrics,
        "ai_results": ai_results,
        "real_results": real_results,
    }

    with open(args.output, 'w') as f:
        json.dump(detailed_results, f, indent=2)

    print(f"\nDetailed results saved to: {args.output}")

    # Provide recommendations
    print("\nRECOMMENDATIONS:")
    if metrics['accuracy'] > 0.9:
        print("  ✅ Excellent accuracy! Tool is ready for production.")
    elif metrics['accuracy'] > 0.8:
        print("  ✅ Good accuracy. Consider fine-tuning thresholds.")
    elif metrics['accuracy'] > 0.7:
        print("  ⚠️  Moderate accuracy. Need more enhancements.")
    else:
        print("  ❌ Poor accuracy. Major improvements needed.")

    if metrics['false_positive_rate'] > 0.1:
        print("  ⚠️  High false positive rate. Consider adjusting threshold or enhancing mechanisms.")
    if metrics['false_negative_rate'] > 0.1:
        print("  ⚠️  High false negative rate. Consider adding more AI-specific detection mechanisms.")

    return 0 if metrics['accuracy'] > 0.8 else 1


if __name__ == "__main__":
    exit(main())
