# 🧠 SpikeCT-Restorer: Bio-Inspired Low-Dose CT Restoration

Welcome to the **SpikeCT-Restorer** repository. This project explore the intersection of neuromorphic computing and medical imaging, specifically focusing on the restoration of Low-Dose CT (LDCT) scans using bio-inspired Spiking Neural Networks.

By leveraging **Spiking Denoising Autoencoders (SDAEs)**, we aim to mimic biological neural efficiency to remove noise from reconstructed CT images, providing high-quality diagnostic visuals while potentially reducing the energy footprint of medical AI.

---

## 🔬 Project Overview

Traditional CT denoising relies on standard Convolutional Neural Networks (CNNs). This project introduces a bio-inspired approach using **Spiking Neural Networks (SNNs)**. Unlike standard artificial neurons, spiking neurons process information through discrete temporal events (spikes), offering a naturally robust mechanism for filtering high-frequency noise.

### Key Features:
*   **Dataset:** Trained on the **Mayo Clinic Low-Dose CT Dataset** (10 patients, paired full/low dose).
*   **Architectures:**
    *   **CNN-Final:** A 314k parameter Convolutional Denoising Autoencoder baseline.
    *   **SNN-Direct:** SNN using Direct Encoding (Integrate-and-Fire / Leaky Integrate-and-Fire).
    *   **SNN-Latency:** SNN using Temporal Latency Encoding for increased biological plausibility.
*   **Encoding Strategies:** Supports both Rate-based and Latency-based neural coding.

---

## 🛠 Project Structure

- `app/`: Interactive Gradio demo and sample slices for testing.
- `src/models/`: Core architectures (CNN, SNN, CDAE).
- `src/encoding/`: Spike encoders (Direct, Latency).
- `scripts/preprocessing/`: Data conversion from DICOM to HDF5.
- `scripts/train/`: Training pipelines for both CNN and SNN models.
- `scripts/eval/`: Quantitative and qualitative evaluation scripts.
- `checkpoints/`: Pre-trained model weights.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.9+
*   PyTorch (CUDA support highly recommended)
*   `SpikingJelly` (for SNN implementation)
*   Gradio (for the interactive demo)
*   h5py, pydicom, scikit-image, matplotlib

### Installation
```bash
git clone https://github.com/[username]/SpikeCT-Restorer.git
cd SpikeCT-Restorer
pip install -r requirements.txt
```

---

## 📖 Workflow

### 1. Data Preprocessing
Convert the Mayo DICOM dataset into a standardized HDF5 format:
```bash
python scripts/preprocessing/build_dataset.py
```
This script performs HU windowing ([-1000, 1000]) and normalization to [0, 1].

### 2. Training
To train the baseline CNN:
```bash
python scripts/train/train_cnn.py
```
To train the Spiking Neural Network (IF-based):
```bash
python scripts/train/train_snn_50.py
```

### 3. Interactive Demo
Experience the denoising performance first-hand:
```bash
python app/demo.py
```
This launches a web interface where you can upload low-dose slices and compare all models side-by-side.

---

## 🎨 Interactive Demo Features
*   **Multi-Model Comparison:** Run CNN-Final, SNN-Direct-IF, SNN-Direct-LIF, and SNN-IF Latency side-by-side.
*   **Interactive Gallery:** Click on results to zoom in and download denoised images.
*   **Quality Metrics:** Upload an optional Full-Dose reference to automatically calculate **PSNR** gain and **SSIM**.
*   **Real-time Latency:** Compare the inference speed of spiking models vs. traditional CNNs.

---

## 📊 Roadmap
- [x] Mayo Dataset Preprocessing Pipeline
- [x] Baseline Convolutional Autoencoder (CNN-Final)
- [x] Spiking Denoising Autoencoder (Direct IF/LIF)
- [x] Latency Encoding Implementation
- [x] Interactive Gradio Demo
- [ ] Energy Efficiency Benchmarking (SNN vs CNN)
- [ ] Sinogram-space Denoising Research

---

## 📝 Contributors
*   **Youssef Ahmed** - Lead Researcher/Student
*   **Shady Nagy** - Project Supervisor

---

## 📄 License
This project is licensed under the MIT License.
