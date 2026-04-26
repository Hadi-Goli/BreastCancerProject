
# 🎗️ Breast Cancer Detection System (Full-Stack)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project is a comprehensive **Full-Stack Breast Cancer Detection System** utilizing Deep Learning on mammogram images (**CBIS-DDSM** dataset).

Unlike traditional monolithic scripts, this project features a modern **Client-Server architecture**:

- **Backend:** A high-performance **FastAPI** server that handles AI inference, preprocessing, and Explainable AI (XAI) generation.
- **Frontend:** A responsive, interactive web dashboard built with **React**, **Vite**, and **Shadcn UI** for a seamless user experience.

The system classifies microcalcifications/masses as **Benign** or **Malignant** and provides **Grad-CAM heatmaps** to visualize the model's focus area.

---

## 🚀 Key Features

### 🧠 AI & Backend Core
- **Smart Preprocessing Pipeline:**
  - **Heuristic Data Selection:** Automatically distinguishes between "cropped tissue images" and "binary ROI masks" using color histogram analysis.
  - **CLAHE Enhancement:** Applies *Contrast Limited Adaptive Histogram Equalization* to reveal hidden details in dense breast tissue.
- **Robust Training Pipeline:** Implements **Class Weighting** to handle dataset imbalance and logs comprehensive metrics (**AUC, F1-Score, Precision, Recall**) for scientific evaluation.
- **Flexible Models:** Supports **ResNet18**, **ResNet34**, and **EfficientNet** backbones via Transfer Learning.
- **Explainable AI (XAI):** Generates **Grad-CAM heatmaps** to make the "black box" decisions interpretable for medical professionals.

### 💻 Frontend & UI
- **Modern Dashboard:** Built with **React** and **Tailwind CSS**.
- **Interactive Visualization:** Real-time upload, confidence score display, and toggleable heatmap overlays.
- **Decoupled Architecture:** Communicates with the backend via a RESTful API (`/predict`).

---

## 📂 Project Structure

```text
root/
│
├── api.py                   # 🧠 Backend: FastAPI entry point & endpoints
├── train_script.py          # 🏋️ Training: CLI script for model training
├── requirements.txt         # 📋 Dependencies (Python)
│
├── core/                    # ⚙️ Core Logic
│   ├── config.py            # Configuration & Constants
│   ├── model.py             # PyTorch Model Definitions
│   ├── preprocessing.py     # CLAHE, Transforms, & Data Loading
│   └── inference.py         # Prediction Logic & Grad-CAM Wrapper
│
├── frontend/                # 🎨 Frontend: React Application
│   ├── src/                 # Source code (Components, Pages, Hooks)
│   ├── package.json         # JS Dependencies
│   └── ...
│
├── checkpoints/             # 💾 Saved Models (.pth) & Training Graphs
└── data/                    # 📁 Dataset (Images & CSVs)

```

---

## 📂 Dataset Details

This project uses the **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) dataset.

* **Input:** Mammogram patches (grayscale).
* **Classes:** Benign (0) vs. Malignant (1).
* **Structure:** The core data loader automatically handles the complex directory structure (images organized by patient case) and metadata CSVs.

---

## 🛠️ Installation & Run Guide

To run the full system locally, you need to set up two separate terminals: one for the Backend and one for the Frontend.

### 1️⃣ Backend Setup (Terminal 1)

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Start the FastAPI server
uvicorn api:app --reload --host 0.0.0.0 --port 8000

```

✅ *The API will be live at `http://localhost:8000` (Docs at `/docs`).*

### 2️⃣ Frontend Setup (Terminal 2)

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install JavaScript packages
npm install
# Note: If you use bun, run: bun install

# 3. Start the development server
npm run dev

```

✅ *The web app will launch at `http://localhost:5173`.*

---

## 🏋️ Training the Model

### Option A: Local Training

If you want to retrain the model from scratch on your local machine:

```bash
# Train with ResNet34 and CLAHE enabled
python train_script.py --backbone resnet34 --use-clahe --epochs 10

```

**Training Arguments:**

* `--backbone`: `resnet18`, `resnet34`, `efficientnet_b0`
* `--no-clahe`: Disable CLAHE preprocessing.
* `--batch-size`: Set batch size (default: 32).

### Option B: ☁️ Training on Google Colab

Since the full-stack app requires a browser environment, use Colab only for **model training** (GPU acceleration recommended):

```python
# 1. Clone Repo & Install Deps
!git clone [https://github.com/yourusername/breast-cancer-project.git](https://github.com/yourusername/breast-cancer-project.git)
%cd breast-cancer-project
!pip install -r requirements.txt

# 2. Run training script
!python train_script.py --backbone resnet34 --epochs 10

```

> **Note:** Ensure your dataset is uploaded to `data/` or mounted via Google Drive.

---

## 📊 Methodology & Workflow

1. **Client-Side Processing (Frontend):**
* The user selects an image via the **React** interface.
* The image is converted to Base64 and sent asynchronously using `Axios` to the FastAPI backend.


2. **Data Validation (Backend):**
* The system validates the file format and checks for "mask images" vs "tissue scans" using the **`count_unique_colors`** heuristic algorithm.


3. **Advanced Preprocessing:**
* **CLAHE Enhancement:** A Contrast Limited Adaptive Histogram Equalization (ClipLimit=2.0) is applied to emphasize tissue structures.
* **Normalization:** The image is resized to `224x224` and normalized using ImageNet mean/std standards.


4. **AI Inference Engine:**
* The image is passed through the loaded **PyTorch Backbone** (ResNet18/34 or EfficientNet).
* The model acts as a binary classifier, calculating the logits for "Benign" vs "Malignant".


5. **Uncertainty Quantification & Safety:**
* The system evaluates the confidence score. If the probability falls within the **ambiguity range** (e.g., near the decision boundary), a **Warning Flag** is triggered.
* This alerts the user that the model is uncertain, prompting manual medical review.


6. **Explainability (XAI):**
* **Grad-CAM** hooks into the final convolutional layer to capture gradients.
* A heatmap is generated and superimposed on the original image to visualize the region of interest (ROI).


7. **Response Construction:**
* The backend constructs a JSON payload containing the **Prediction Class**, **Confidence Score (%)**, **Uncertainty Warning**, and the **Base64-encoded Heatmap** for rendering.



---

## 👥 Contributors & Academic Context

This project was developed by a dedicated team of students from **Kharazmi University** as part of the **Artificial Intelligence** course.

**Supervisor:**

* 🎓 **Dr. Hamidreza Bolhasani**

*We express our gratitude for the guidance and scientific supervision provided throughout the development of this research project.*

