# Machine-Learning-Basics
This repository contains a collection of Jupyter Notebook labs that progressively cover key **machine learning concepts and models**, from classical regression and tree-based methods to neural networks and diffusion-based generative modeling.
Each lab focuses on **hands-on implementation, evaluation, and analysis** of real ML workflows.

---

## 📂 Lab Overview

### 🔹 `Lab1.ipynb` — Regression (Used Car Pricing)

Supervised learning on tabular data to **predict used car prices** using:

- **Linear Regression**
- **Polynomial Regression**

Includes:

- preprocessing & feature selection  
- train / validation splits  
- model performance evaluation  
- exporting prediction CSV files  

---

### 🔹 `Lab2.ipynb` — Tree-Based Classification (Healthcare Outcome)

Binary **classification** using tree-based models:

- **Decision Tree Classifier**
- **Random Forest Classifier**

Focus areas:

- handling missing data with **imputation**
- model training & validation
- evaluation using **accuracy & F1-score**
- generating prediction CSVs  

---

### 🔹 `Lab3.ipynb` — LDA & SVM for Human Activity Recognition

**Multi-class classification** using wearable sensor data, applying:

- **Linear Discriminant Analysis (LDA)**
- **Support Vector Machines (SVM)**

Topics covered:

- binary vs multi-class classification
- feature projection & visualization
- confusion matrix & metric analysis

> ⚠️ **Data Notice**  
> The training and testing dataset files (e.g., `train.csv` and `test.csv`) are **NOT included in this repository due to file size limits**.  
> Download the dataset manually and place it in the `Lab3/` directory before running the notebook.

---

### 🔹 `Lab4.ipynb` — Neural Network Classification (Price Level Prediction)

Tabular classification using a **Feedforward Neural Network (PyTorch)**.

Includes:

- **Min-Max normalization**
- **One-Hot Encoding**
- custom PyTorch `Dataset` & `DataLoader`
- model training, validation, and testing pipelines  

Goal: predict **price level categories**.

---

### 🔹 `Lab5.ipynb` — Generative Modeling with Diffusion (Conditional DDPM)

Deep **generative modeling** on image data using a **Conditional Denoising Diffusion Probabilistic Model (DDPM)**.

Key components:

- diffusion noise scheduling  
- forward noise process  
- **conditional UNet with time embeddings**  
- label-conditioned image generation (digits 0–9)

Outputs: **generate images conditioned on labels**.

> ⚠️ **Data Notice (Lab 5 — MNIST)**  
> The MNIST dataset is **not downloaded automatically by the notebook**.  
> Download MNIST manually and place it in the required directory before running the diffusion model.

---

## 🚀 Getting Started

### 🔧 Requirements

You’ll need:

- Python **3.x**
- **Jupyter Notebook / JupyterLab**
- Common ML libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `torch` *(for Lab4 & Lab5)*

Install Jupyter with:

```bash
pip install jupyter
```

--- 
### ▶️ How To Use
- Open any notebook (e.g., Lab1.ipynb) in Jupyter.
- Run cells top-to-bottom (Shift + Enter).
- Review outputs, tweak parameters, and experiment freely.

---
### 🛠 Technologies Used
- Python 3
- Jupyter Notebook
- scikit-learn
- PyTorch
- NumPy
- Pandas
- Matplotlib

