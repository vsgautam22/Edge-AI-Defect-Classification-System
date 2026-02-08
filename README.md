# Edge-AI Defect Classification System

An Edge-AI–ready deep learning system for detecting and classifying defects in semiconductor wafer/die images, designed with real fabrication constraints such as class imbalance, limited compute, and unknown defect handling.

> **Status:** Dataset frozen • Model finalized • ONNX exported

---

## 1. Problem Statement

Manual and rule-based inspection of semiconductor wafers is time-consuming and brittle when faced with subtle or previously unseen defects.  
This project addresses **multi-class wafer defect classification** using an **edge-deployable deep learning model** that balances accuracy, robustness, and deployability.

---

## 2. Dataset Summary

- **Total images:** 2006  
- **Classes (10):**
  - bridges
  - clean
  - cmp_scratches
  - cracks
  - ler (Line Edge Roughness)
  - open
  - particles
  - residues
  - via
  - others (unknown / mixed defects)
- **Split:** Train / Validation / Test (70 / 15 / 15)
- **Labeling:** Manual + audited
- **Imbalance handling:** Explicit class weighting during training

> The **`others`** class is introduced to absorb unknown or mixed defects, improving robustness and reducing false positives in production scenarios.

---

## 3. Model Architecture & Training

- **Backbone:** ResNet-18 (transfer learning)
- **Loss:** Class-weighted cross-entropy
- **Input resolution:** 224 × 224
- **Training strategy:**
  - Data augmentation
  - Class-weighted loss for imbalance
  - Early stopping via validation monitoring

ResNet-18 was chosen to balance **accuracy and edge-deployability**, avoiding over-parameterization for the available dataset size.

---

## 4. Results

- **Test Accuracy:** ~86%
- **Weighted F1-score:** ~0.86
- **Macro F1-score:** ~0.76
- **Observations:**
  - Strong performance on CMP scratches, cracks, residues, and via defects
  - Partial confusion between visually similar classes (e.g., LER vs bridges), consistent with real-world inspection challenges
  - No dead or ignored classes

Detailed metrics and confusion matrix are included in the results report.

---

## 5. Edge Deployment Readiness

- ONNX export completed (opset-stable)
- No post-training quantization required
- Model size suitable for edge inference
- Confidence-calibrated predictions

---

## 6. Repository Contents

- train_phase1_multiclass.py # Final training script
- dataset_stats.py # Dataset statistics utility
- model_audit.py # Model sanity and audit checks
- export_to_onnx.py # ONNX export script
- requirements.txt # Python dependencies


> **Note:** Dataset images and trained weights are intentionally excluded due to size constraints.

---

## 7. Artifacts & Links

- **ONNX model:** https://drive.google.com/file/d/1vty0m3iSvfD-WojL4oeF3BHl7dksipGW/view?usp=drive_link
- **Results report (PDF):** <ADD LINK>

---

## 8. License

This project is released under the MIT License.
