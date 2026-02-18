# The Classification Lottery in Few-Shot Medical Image Transfer Learning


---

## Overview

This repository contains the complete experimental code for our IEEE Access paper. We systematically evaluate **144 configurations** across binary pneumonia detection and 4-class COVID-19 classification to characterize *initialization sensitivity* — a phenomenon we term the **classification lottery** — and identify practical mitigation strategies.

**Key Finding:** At 5 samples per class with head-only fine-tuning, binary classification exhibits up to **55.6 percentage-point accuracy spread** from identical configurations differing only in random seed. Counter-intuitively, 4-class classification shows **4.7× lower variance**, attributable to gradient diversity from increased output dimensionality.

---

## Results Summary

| Task | Strategy | Samples/Class | Accuracy | Variance |
|---|---|---|---|---|
| Binary (Pneumonia) | Head-Only | 5 | 60.7% | ±39.3% |
| Binary (Pneumonia) | Layer-Wise | 5 | 89.2% | ±0.5% |
| Binary (Pneumonia) | Layer-Wise | 20 | 91.1% | ±2.7% |
| 4-Class (COVID-19) | Head-Only | 5 | 25.2% | ±7.7% |
| 4-Class (COVID-19) | Layer-Wise | 5 | 58.7% | ±1.0% |
| 4-Class (COVID-19) | Layer-Wise | 100 | 87.8% | ±0.2% |

**Clinical-grade performance** (≥90% binary, ≥85% multi-class, variance <5%, AUC-ROC ≥0.95) is achievable with just **40–400 labeled images** using layer-wise fine-tuning.

---

## Repository Structure

```
├── main.py                          # Entry point — run all experiments
├── src/
│   ├── data_loading_and_preprocess.py   # Dataset class, transforms, data splits
│   ├── model_creation.py                # ResNet50, EfficientNet-B4, ViT-B/16 setup
│   ├── training_model.py                # Training loop, early stopping, validation
│   └── full_experiment.py               # Complete experiment pipeline
├── results_pneumonia/               # Experiment outputs (binary task)
├── results_covid19/                 # Experiment outputs (multi-class task)
│   └── training_log.txt
└── README.md
```

---

## Experimental Design

| Factor | Levels |
|---|---|
| Architectures | ResNet50, EfficientNet-B4, ViT-B/16 |
| Sample Sizes | 5, 10, 20, 50, 100, 200 per class |
| Fine-Tuning Strategies | Head-only (0.016% params), Layer-wise (76.3% params) |
| Random Seeds | 42, 456 |
| **Total Experiments** | **144** |

---

## Datasets

| Dataset | Task | Classes | Total Images |
|---|---|---|---|
| [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Binary | 2 | 5,856 |
| [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) | Multi-class | 4 | 21,165 |

Download both datasets and place them as follows:
```
data/
├── chest_xray/
│   ├── train/
│   └── test/
└── covid19_radiography/
    ├── COVID/
    ├── Lung_Opacity/
    ├── Normal/
    └── Viral Pneumonia/
```

---

## Installation

```bash
git clone https://github.com/mdtouhiduliskam/The-Classification-Lottery-in-Few-Shot-Medical-Image-Transfer-Learning.git
cd The-Classification-Lottery-in-Few-Shot-Medical-Image-Transfer-Learning

pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
Pillow>=9.5.0
tqdm>=4.65.0
```

---

## Usage

### Quick pipeline verification (run 1 experiment first)
```bash
python main.py --mode verify
```

### Run full experiments
```bash
python main.py --mode full
```

### Run single configuration
```python
from src.full_experiment import run_experiment

results = run_experiment(
    dataset_key='pneumonia',
    model_name='resnet50',
    sample_size=20,
    finetune_strategy='layer_wise',
    augmentation='standard',
    num_classes=2,
    run_seed=42,
    dataset_name='Chest X-Ray Pneumonia'
)
```

### Test mode (fast validation)
Set `TEST_MODE = True` in `main.py` to run a quick sanity check with reduced epochs and a single sample size before committing to full experiments.

---

## Fine-Tuning Strategies

**Head-Only:** Freezes all convolutional layers; trains only the classification head (4,098 trainable parameters, 0.016% of ResNet50).

**Layer-Wise:** Freezes only the first convolutional block; trains all later layers and the classification head (19,499,050 trainable parameters, 76.3% of ResNet50).

Architecture-specific layer-wise configurations:
- **EfficientNet-B4:** Freeze blocks 0–2, train blocks 3–7 + head
- **ViT-B/16:** Freeze transformer layers 0–3, train layers 4–11 + head

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| Batch Size | 32 |
| Max Epochs | 50 |
| Early Stopping Patience | 10 |
| GPU | Kaggle P100 (16GB) |

---

## Computational Requirements

| Strategy | 5-Shot | 20-Shot | 100-Shot | GPU Memory |
|---|---|---|---|---|
| Head-Only | 2–3 min | 4–6 min | 8–12 min | ~2.1 GB |
| Layer-Wise | 10–15 min | 15–22 min | 25–35 min | ~5.8 GB |

All experiments were run on Kaggle Notebooks (single P100 GPU). The full 144-experiment sweep takes approximately 24–48 hours.

---

## Clinical Deployment Thresholds

We define clinical-grade performance as simultaneously satisfying:
- Accuracy ≥ 90% (binary) or ≥ 85% (multi-class)
- Standard deviation < 5% across seeds
- AUC-ROC ≥ 0.95

**Minimum sample requirements (layer-wise fine-tuning):**

| Task | Architecture | Samples/Class | Accuracy | Variance | AUC-ROC |
|---|---|---|---|---|---|
| Binary | ResNet50 | 20 | 91.1% | ±2.7% | 0.978 |
| Binary | ViT-B/16 | 20 | 90.6% | ±1.1% | 0.971 |
| Binary | EfficientNet-B4 | 50 | 91.2% | ±1.2% | 0.976 |
| 4-Class | ResNet50 | 100 | 87.8% | ±0.2% | 0.968 |
| 4-Class | EfficientNet-B4 | 100 | 86.6% | ±0.4% | 0.957 |
| 4-Class | ViT-B/16 | 100 | 86.6% | ±0.6% | 0.962 |

---

## Citation

If you use this code or findings in your research, please cite:

<!-- ```bibtex
@article{hasan2025classificationlottery,
  title={The Classification Lottery in Few-Shot Medical Image Transfer Learning},
  author={Hasan, Ekramul and Rahaman, Mustafizur and Mosaddeque, Ahmde Inan 
          and Sultana, Ruksana and Islam, Md Touhidul},
  journal={IEEE Access},
  year={2025},
  note={Under review}
}
``` -->

---

## License

This project is licensed under the MIT License. The datasets are subject to their own licenses — please refer to the respective Kaggle dataset pages.

---

## Contact

**Corresponding Author:** Md Touhidul Islam  
TU Dortmund University, Dortmund, Germany  
mdtouhidul.islam@tu-dortmund.de