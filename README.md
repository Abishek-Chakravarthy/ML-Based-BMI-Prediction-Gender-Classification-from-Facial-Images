# Gender Classification and BMI Prediction from Facial Features

This project aims to classify gender and predict Body Mass Index (BMI) from facial images using deep learning techniques and regression models. It involves preprocessing facial images, extracting features using pre-trained CNNs (VGG16 and FaceNet), and applying machine learning models for prediction.

## 👥 Author
- Abishek Chakravarthy (CS22B2054)
---

## 📌 Project Overview

We develop two pipelines:
1. **Gender Classification** – Binary classification using logistic regression.
2. **BMI Prediction** – Regression using linear regression.

Facial images (front and side) are processed to extract high-level features using VGG16 and FaceNet, which are then used for model training.

---

## 🗃️ Dataset & Preprocessing

### Preprocessing Steps
- Removed corrupted/invalid images and extreme BMI outliers (BMI < 10 or > 50).
- Resized all images to **128x128** and converted to **grayscale**.
- Computed BMI using the formula:  
  \[
  	ext{BMI} = rac{	ext{weight (kg)}}{	ext{height}^2 (	ext{m}^2)}
  \]
- Labels:
  - **Gender**: 0 or 1
  - **BMI**: Computed numerical value

---

## 🧠 Feature Extraction

### VGG16
- Pre-trained on ImageNet.
- Extracted ~30,000 features from front and side images.

### FaceNet
- Used for high-quality facial embeddings.
- Produced 2,000 features (1,000 from front and 1,000 from side).

---

## 🔍 Model Training

### Gender Classification
- **Model:** Logistic Regression
- **Train/Test Split:** 80/20
- **Dataset Size:** 61,110 (48,888 train, 12,222 test)

### BMI Prediction
- **Model:** Linear Regression
- **Train/Test Split:** 80/20
- **Dataset Size:** 59,985 (47,988 train, 11,997 test)

---

## 📈 Results

### Gender Classification
- **Accuracy:** High on both train and test sets
- **Model Metrics:**
  - MAE: ~1.24
  - MSE: ~2.79
  - R²: ~0.89
  - Pearson Correlation: ~0.94

### BMI Prediction
- **Testing Accuracy:** 89.33%
- **Model Metrics:**
  - MAE: 0.0741
  - MSE: 0.0325
  - R² Score: 0.87
  - Pearson Correlation: 0.93

---

## 📓 Notebooks

- [`gender_classification.ipynb`](./gender_classification.ipynb): End-to-end implementation of the gender classification pipeline.
- [`bmi_prediction.ipynb`](./bmi_prediction.ipynb): End-to-end implementation of BMI prediction pipeline.

---

## 🛠️ Requirements

- Python 3.x
- NumPy
- OpenCV
- TensorFlow / Keras
- Scikit-learn
- Matplotlib


---

