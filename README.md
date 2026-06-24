# Reliable Fault Detection and Isolation in PMSM Using Lightweight SE-CNN-LSTM for Predictive Health Monitoring

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Accuracy](https://img.shields.io/badge/Accuracy-99.45%25-brightgreen)
![Model Size](https://img.shields.io/badge/Model%20Size-111.57%20KB-blueviolet)


> A lightweight, data-driven deep learning framework for multi-class fault classification in Permanent Magnet Synchronous Motor (PMSM) drives — designed for both high accuracy and embedded deployment.

---

## 📖 Overview

Permanent Magnet Synchronous Motors (PMSMs) are critical components in high-stakes systems including electric vehicles, industrial robotics, and aerospace applications. Traditional fault protection methods rely on static threshold rules that can only detect hard failures — they completely miss early-stage, gradual degradation like the onset of overheating or a developing short circuit.

This project proposes a **hybrid deep learning framework** that combines:
- **1D-CNN** — for spatial feature extraction across sensor channels
- **LSTM** — for temporal trend tracking across time-series sequences
- **Squeeze-and-Excitation (SE) Attention Block** — for dynamic, automatic channel weighting

The result is a model that achieves **99.45% classification accuracy** across **9 fault states** while remaining small enough (**111.57 KB**) for deployment on low-power embedded microcontrollers via **TensorFlow Lite INT8 quantization**.

---

## 🎯 Key Features

- ✅ **Multi-class fault classification** — detects 9 distinct motor states (Healthy, Open-Circuit, Short-Circuit, Overheating)
- ✅ **SE Attention Block** — automatically prioritizes the most relevant sensor channels per fault type, eliminating manual feature engineering
- ✅ **Lightweight architecture** — only 28,561 parameters (111.57 KB), designed for MCU deployment
- ✅ **TF Lite quantized model** — INT8 quantization for embedded hardware (STM32, ARM Cortex-M)
- ✅ **Real-time inference simulation** — 1-second sliding window diagnostic pipeline with confidence output
- ✅ **Surpasses state-of-the-art** — existing literature reports 92–98% accuracy; this model achieves 99.45%

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│              SENSING LAYER                          │
│  Ia  Ib  VDC  IDC  T1  T2  T3  VD (8 sensors)     │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│           PREPROCESSING PIPELINE                    │
│  Z-Score Normalization → Sliding Window (10 steps)  │
│  Input Shape: (Batch, 10 time-steps, 8 features)    │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│         HYBRID DEEP LEARNING CORE                   │
│                                                     │
│  Conv1D (32 filters, k=3, padding='same')           │
│       ↓                                             │
│  SE Attention Block (Squeeze → Excite → Scale)      │
│       ↓                                             │
│  MaxPooling1D (pool=2)                              │
│       ↓                                             │
│  LSTM (64 units)  +  Dropout (0.3)                  │
│       ↓                                             │
│  Dense (32, ReLU) +  Dropout (0.2)                  │
│       ↓                                             │
│  Dense (9, Softmax)                                 │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│              DIAGNOSTIC OUTPUT                      │
│  9-Class Fault Label + Confidence Score             │
│  TF Lite INT8 Export → Embedded MCU/FPGA            │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Zenodo PMSM Inverter Fault Dataset](https://zenodo.org/) |
| **File** | `converted_dataset.csv` |
| **Total Samples** | 10,892 rows |
| **Sampling Rate** | 10 Hz |
| **Input Features** | 8 sensor channels |
| **Classes** | 9 balanced classes (F0–F8) |

### Sensor Channels

| Sensor | Description | Type |
|---|---|---|
| `Ia` | Phase A Current | Electrical |
| `Ib` | Phase B Current | Electrical |
| `VDC` | DC Bus Voltage | Electrical |
| `IDC` | DC Bus Current | Electrical |
| `T1` | Half-bridge Thermistor 1 | Thermal |
| `T2` | Half-bridge Thermistor 2 | Thermal |
| `T3` | Half-bridge Thermistor 3 | Thermal |
| `VD` | Gate Driver Voltage | Control |

### Fault Classes

| Class | Label | Description |
|---|---|---|
| F0 | Healthy | Normal Operation |
| F1, F2 | Open-Circuit | Phase disconnection (current drops to zero) |
| F3, F4, F5 | Short-Circuit | Phase short (massive current surge + voltage drop) |
| F6, F7, F8 | Overheating | Progressive thermal degradation (moderate → critical) |

---

## ⚙️ Model Details

### Architecture Summary

| Layer | Output Shape | Parameters |
|---|---|---|
| Input | (None, 10, 8) | 0 |
| Conv1D (32 filters, k=3) | (None, 10, 32) | 800 |
| SE Block — GlobalAvgPool | (None, 32) | 0 |
| SE Block — Dense (reduce) | (None, 8) | 264 |
| SE Block — Dense (restore) | (None, 32) | 288 |
| SE Block — Reshape + Multiply | (None, 10, 32) | 0 |
| MaxPooling1D (pool=2) | (None, 5, 32) | 0 |
| LSTM (64 units) | (None, 64) | 24,832 |
| Dropout (0.3) | (None, 64) | 0 |
| Dense (32, ReLU) | (None, 32) | 2,080 |
| Dropout (0.2) | (None, 32) | 0 |
| Dense / Softmax (9) | (None, 9) | 297 |
| **Total** | | **28,561 → 111.57 KB** |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Categorical Cross-Entropy |
| Epochs | 50 (with EarlyStopping, patience=5) |
| Train / Test Split | 80% / 20% |
| Training Samples | 8,705 |
| Test Samples | 2,177 |
| Window Size | 10 time-steps (1 second) |

---

## 📈 Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **99.45%** |
| **Model Size (.h5)** | **111.57 KB** |
| **Total Parameters** | **28,561** |
| Literature Baseline | 92–98% |

- **Confusion Matrix** confirms near-perfect diagonal classification across all 9 states.
- **Accuracy/Loss curves** show smooth convergence with no signs of overfitting.
- **SE Block contribution**: +~2.5% accuracy over plain CNN-LSTM baseline for only +552 additional parameters.

---

## 🚀 Getting Started

### Prerequisites

- Google Colab (recommended) or Python 3.8+ environment
- Required libraries: `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

### Running the Project

1. **Clone this repository** or download the project files.

2. **Open [Google Colab](https://colab.research.google.com/)** and upload:
   - `Minor_Project_.ipynb`
   - `converted_dataset.csv`

3. **Run all cells in order** (top to bottom). The notebook will automatically:
   - Load and explore the dataset
   - Apply Z-Score normalization and sliding window segmentation
   - Build and train the SE-CNN-LSTM model
   - Evaluate performance (accuracy, confusion matrix, classification report)
   - Export a quantized `.tflite` model for embedded deployment
   - Run a real-time inference simulation

### Project Files

| File | Description |
|---|---|
| `Minor_Project_.ipynb` | Main project notebook (entire pipeline) |
| `converted_dataset.csv` | PMSM sensor dataset (required input) |
| `pmsm_advanced_se_cnn_lstm.h5` | Saved trained model weights |
| `pmsm_se_cnn_lstm_int8.tflite` | Quantized TF Lite model (generated on run) |

---

## 🔬 The SE Block — Key Innovation

The **Squeeze-and-Excitation (SE) Block** is the core innovation of this architecture. Standard CNNs treat all sensor channels with equal importance at all times. The SE block solves this by:

1. **Squeeze** — Global Average Pooling summarizes each of the 32 CNN feature channels into a single representative value.
2. **Excite** — Two Dense layers learn a weight (0–1) for each channel based on the current input context.
3. **Scale** — These learned weights are multiplied back onto the original features — amplifying important channels and suppressing irrelevant ones.

This means the model automatically focuses on **current sensors** during a short-circuit event and on **temperature sensors** during an overheating event — without any manual feature engineering.

```python
def se_block(input_tensor, ratio=4):
    channels = input_tensor.shape[-1]
    squeeze = GlobalAveragePooling1D()(input_tensor)
    excitation = Dense(channels // ratio, activation='relu')(squeeze)
    excitation = Dense(channels, activation='sigmoid')(excitation)
    excitation = Reshape((1, channels))(excitation)
    return Multiply()([input_tensor, excitation])
```

---

## 🔮 Future Scope

- **Hardware Deployment** — Port the `.tflite` model to an STM32 ARM Cortex-M MCU for live physical testing
- **Online/Continual Learning** — Enable the model to adapt to drifting motor conditions without full retraining
- **Noise Robustness** — Validate under real-world electromagnetic interference (EMI) and sensor noise
- **Expanded Fault Classes** — Integrate vibration/acoustic sensors to detect mechanical faults (bearing wear, rotor eccentricity)
- **Real-Time Dashboard** — Build an embedded display for live fault probability monitoring

---

## 👥 Team

| Name | Role |
|---|---|
| A. Yashaswi | Team Lead |
| Rida Almas | Co-Developer |
| T. Harsha Vardhan | Co-Developer |
| Richard Luke | Co-Developer |

**Guide:** Dr. P. Tejaswi
**Institution:** VNR Vignana Jyothi Institute of Engineering and Technology (VNR VJIET)

---



> *"Intelligent, embedded health monitoring for electric motors is not a future concept — it is achievable today with lightweight deep learning."*
