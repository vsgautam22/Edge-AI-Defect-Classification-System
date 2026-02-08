# Edge-AI-Defect-Classification-System
An Edge-AI capable system that can detect and classify defects in semiconductor wafer/die images using AI/ML, while balancing accuracy, latency, and compute efficiency to reflect real fab constraints.

Edge-deployable deep learning system for semiconductor wafer defect classification.

> Status: Dataset and model frozen. Results finalized.

## Overview
This repository contains the dataset structure, trained model artifacts, evaluation results, and inference code for an Edge-AI wafer defect classification system.

Problem Statement

Manual wafer inspection is slow, inconsistent, and non-scalable

Dataset Summary
Attribute	Value
Total images	2006
Classes	10
Clean vs Defective	Included
Labeling	Manual + audited
Split	Train / Val / Test (frozen)
Model Architecture

Backbone: ResNet-18

Training: Class-weighted cross-entropy

Export: ONNX (opset stable)

Inference-ready for edge deployment

Performance Metrics

Accuracy: ~86%

Weighted F1-score: ~0.86

Confusion matrix included

Class imbalance explicitly handled

Edge Readiness

Model size optimized

ONNX-compatible

No post-training required

Repository Contents

Brief explanation of each top-level folder.

How to Run Inference
python inference/onnx_inference.py --image sample.png

Limitations

Optical-only images

Minority defect classes constrained by data volume

Domain shift possible across fabs

License

Specify (MIT / Apache 2.0 preferred)
