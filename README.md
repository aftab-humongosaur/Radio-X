# Radio-X
Radio-X: AI-powered chest X-ray enhancement system using a Denoising Convolutional Autoencoder built with TensorFlow and Streamlit.

# 🩺 Radio-X — AI-Powered Chest X-Ray Enhancement System

> "Clearing the noise, sharpening the diagnosis."

Radio-X is a deep learning-based medical imaging tool that enhances low-quality chest X-rays using a Denoising Convolutional Autoencoder. Built as part of the NeuralHack 2026 Deep Learning Hackathon (MAI417-3).

---

## 🧠 Problem Statement

Rural hospitals in India often use outdated X-ray machines that produce noisy, low-quality scans. This leads to:
- Misdiagnosis due to poor image clarity
- Repeated X-rays causing excess radiation exposure
- Delayed treatment in critical cases

**Radio-X solves this by enhancing noisy X-rays using AI — for free, locally, with no data leaving the clinic.**

---

## 🏗️ Architecture

**Denoising Convolutional Autoencoder**
```
Input (noisy X-ray 128x128)
        ↓
   Encoder (CNN + BatchNorm + MaxPooling)
        ↓
   Bottleneck (compressed representation)
        ↓
   Decoder (CNN + BatchNorm + UpSampling)
        ↓
Output (clean X-ray 128x128)
```

### Key Design Choices
| Component | Choice | Reason |
|---|---|---|
| Loss Function | MSE | Better pixel reconstruction than BCE |
| Optimizer | Adam | Adaptive learning rate |
| Regularization | Batch Normalization + Early Stopping | Prevents overfitting |
| Framework | TensorFlow / Keras | Industry standard |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Average PSNR | 25.43 dB |
| Average SSIM | 0.7132 |
| Val Loss (MSE) | 0.0029 |
| Training Images | 3000 |
| Image Resolution | 128 × 128 |

---

## 🌐 Web Application

Built with **Streamlit** — runs entirely in the browser locally.

**Features:**
- Upload any chest X-ray image
- View Original → Simulated Noisy → AI Enhanced side by side
- Live PSNR & SSIM quality scores
- Auto Diagnostic Report with grade (Excellent / Good / Poor)
- Enhancement History table across the session
- Training loss curve visualization

---

## 🚀 Run Locally
```bash
# Clone the repo
git clone https://github.com/aftab-humongosaur/Radio-X.git
cd Radio-X

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle  
[kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- Streamlit
- NumPy, Pillow, scikit-image, Plotly

---

## 📁 Project Structure
```
Radio-X/
├── app.py                  ← Streamlit web application
├── radiox_model.h5         ← Trained autoencoder model
├── training_history.json   ← Loss curve data
├── requirements.txt        ← Python dependencies
└── README.md               ← Project documentation
```

---

## 👨‍💻 Author

**Aftab** — MSAIM, PG III Trimester  
Christ (Deemed to be University), Bangalore  
MAI417-3 Deep Learning | NeuralHack 2026
