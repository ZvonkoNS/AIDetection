# Evaluation Methodology

This document outlines the process for evaluating the Forensic AI Detection Tool (AIDT).

## 1. Accuracy Benchmarking

The primary success metric is ≥90% accuracy on a curated, internal benchmark dataset.

-   **Dataset**: The dataset will consist of a balanced set of human-captured and AI-generated images.
    -   **Human Images**: Sourced from diverse, known-good camera datasets (e.g., internal forensic cases, public datasets like COCO).
    -   **AI Images**: Generated using a variety of models (Stable Diffusion, Midjourney, DALL-E 3, etc.) with different prompts and settings.
-   **Procedure**: The `scripts/benchmark_accuracy.py` script will be used to run the tool against the entire dataset and compute accuracy, precision, recall, and F1-score.
-   **Acceptance Criteria**: The overall accuracy must be ≥90%.

## 2. Performance Benchmarking

The tool must analyze a ≤10MB image in under 15 seconds on a modern CPU.

-   **Hardware**: A standardized modern CPU (e.g., Intel Core i7-12700K or AMD Ryzen 7 5800X) will be used as the reference machine.
-   **Procedure**: The `scripts/benchmark_performance.py` script will be run against a set of standard test images. It will measure and report the median processing time.
-   **Acceptance Criteria**: The median time for any image ≤10MB must not exceed 15 seconds.

## 3. Offline Verification

The tool must operate in a fully air-gapped environment.

-   **Procedure**: End-to-end tests will be run with network access disabled (e.g., using `pytest-socket`). The packaged binary will be smoke-tested on a machine with no network connection.
-   **Acceptance Criteria**: The tool must execute successfully and produce correct results without any network-related errors.
