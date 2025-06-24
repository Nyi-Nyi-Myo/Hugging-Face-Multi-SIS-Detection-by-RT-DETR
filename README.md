# Hugging-Face Multi-SIS Detection by RT-DETR

# 🔍 RT-DETR Surgical Instruments Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c?logo=pytorch)](https://pytorch.org/) 
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/)

A robust real-time object detection pipeline using **RT-DETR (Real-Time Detection Transformer)** to detect 6 types of surgical instruments on a custom dataset.

---

## 🧠 Model

- **Backbone**: RT-DETR with ResNet-101 ("PekingU/rtdetr_r101vd_coco_o365")
- **Framework**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

## 🗂 Dataset

- Format: COCO-style JSON
- Total Images:
  - Train: 2,855 images
  - Valid: 841 images
- Categories (6 total):
  - Grasper
  - Harmonic_Ace
  - Myoma_Screw
  - Needle_Holder
  - Suction
  - Trocar

---

## 🚀 Training Setup

| Component          | Setting                  |
|--------------------|---------------------------|
| Epochs             | 50                       |
| Mixed Precision    | ✅ (fp16)                |
| Evaluation Metric  | `eval_loss` (lower is better) |
| Save Best Model    | ✅                        |

---

## 📊 Evaluation (IoU=0.50:0.95)

| Class           | Precision | Recall | F1 Score |
|----------------|-----------|--------|----------|
| Grasper        | 0.6845    | 0.7235 | 0.7035   |
| Harmonic_Ace   | 0.8759    | 0.9050 | 0.8902   |
| Myoma_Screw    | 0.7757    | 0.8275 | 0.8008   |
| Needle_Holder  | 0.5642    | 0.5947 | 0.5790   |
| Suction        | 0.5811    | 0.6391 | 0.6087   |
| Trocar         | 0.7259    | 0.7525 | 0.7390   |

**Overall F1 Score**: `0.7202`

---

## 🖼 Sample Output

<div align="center">
  <img src="50_epochs_results.png" width="600"/>
</div>

---

## 🛠 How to Use

```bash
# Install dependencies
!pip install -q transformers datasets torchvision pycocotools
!pip install -q accelerate evaluate

# Run evaluation script
run_per_class_eval(
    model=model,
    val_dataset=valid_processed,  # RTDETRProcessedDataset
    processor=processor,
    annotation_file="datasets/surgical_instruments/annotations/instances_valid_det.json",
    device="cuda",
    class_names = ['no', 'Grasper', 'Harmonic_Ace', 'Myoma_Screw',
               'Needle_Holder', 'Suction', 'Trocar']
)
```

---

## 📌 Notes

- Use **RTDETR_Huggingface_det_results.ipynb** for inference results
- 50 epochs results > 30 epochs results 
---

## 📜 License

This project is released for research purposes only. Please cite appropriately if used in publications.

