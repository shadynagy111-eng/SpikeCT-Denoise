# 🧠 SpikeCT-Restorer: Bio-Inspired Low-Dose CT Restoration
Welcome to the **SpikeCT-Restorer** repository. This project explores the intersection of neuromorphic computing and medical imaging, specifically focusing on the restoration of Low-Dose CT (LDCT) scans.

By leveraging **Spiking Denoising Autoencoders (SDAEs)**, we aim to mimic biological neural efficiency to remove noise from **Sinograms** and reconstructed images, providing high-quality diagnostic visuals with significantly reduced radiation exposure.

---

## 🔬 Project Overview

Traditional CT denoising relies on heavy Convolutional Neural Networks (CNNs). This project introduces a bio-inspired approach using **Spiking Neural Networks (SNNs)**. Unlike standard artificial neurons, spiking neurons process information through discrete temporal events (spikes), offering a naturally robust mechanism for filtering high-frequency noise in medical data.

### Key Research Areas:

* **Sinogram Denoising:** Processing raw projection data before reconstruction to prevent artifact amplification.
* **Spiking vs. Non-Spiking:** Comparative analysis of Leaky Integrate-and-Fire (LIF) neurons against standard ReLU-based Autoencoders.
* **Image Restoration:** Enhancing the Structural Similarity Index (SSIM) and PSNR of low-dose medical scans.

---

## 🛠 Project Structure

The repository is organized into two primary phases:

1. **Phase 1: Baseline Implementation** * Development of a standard **Convolutional Denoising Autoencoder (CDAE)**.
* Establishment of benchmarking metrics (PSNR, SSIM, RMSE).
* Synthetic noise injection (Gaussian & Poisson) to simulate Low-Dose environments.


2. **Phase 2: Bio-Inspired Integration** * Implementation of **Spiking Autoencoders**.
* Data encoding strategies (Rate coding vs. Poisson coding for CT intensities).
* Evaluation of energy efficiency and denoising performance.



---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* `SpikingJelly` (for SNN implementation)
* Gradio (for the interactive demo)
* NumPy, Matplotlib, and Scikit-Image

### Installation

```bash
git clone https://github.com/[username]/SpikeCT-Restorer.git
cd SpikeCT-Restorer
pip install -r requirements.txt
```

### 🎨 Interactive Demo

Experience the denoising performance first-hand using our Gradio-based web interface. The demo allows you to compare the CNN baseline against various Spiking Neural Network architectures.

#### Running the Demo
```bash
python app/demo.py
```
This will launch a local server (usually at `http://localhost:7860`) and provide a public sharing link.

#### Features
*   **Multi-Model Comparison:** Run CNN-Final, SNN-Direct-IF, SNN-Direct-LIF, and SNN-IF Latency models side-by-side.
*   **Interactive Gallery:** Click on results to zoom in and download denoised images.
*   **Quality Metrics:** Upload an optional Full-Dose reference to automatically calculate **PSNR** gain and **SSIM**.
*   **Real-time Latency:** Compare the inference speed of spiking models vs. traditional CNNs.

#### Sample Data
You can find demo slices in `app/demo_slices/`. Use the `low_*.npy` files as input and the corresponding `full_*.npy` files as the reference.

---

### Usage (Baseline Autoencoder)

To run the initial denoising benchmark:

```bash
python scripts/train_baseline.py --epochs 50 --noise_level 0.1

```

---

## 📊 Roadmap

* [x] Repository setup and naming.
* [ ] Literature review: Spiking vs. Non-Spiking Autoencoders.
* [ ] Baseline Convolutional Autoencoder implementation.
* [ ] Sinogram preprocessing pipeline.
* [ ] Spiking neuron integration (LIF model).
* [ ] Comparative performance report.

---

## 📝 Contributors

* **Youssef Ahmed** - Lead Researcher/Student
* **Shady Nagy** - Project Supervisor

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

