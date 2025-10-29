# 🧠 Handwritten Digit Recognition using CNN

This project builds a **Convolutional Neural Network (CNN)** to recognize handwritten digits using the **MNIST** dataset. It’s designed as a beginner-friendly deep learning project to understand image classification, CNN architecture, and model deployment basics.

---

## 🎯 Goal
To classify handwritten digits (0–9) from grayscale images using a CNN model trained on the MNIST dataset.

---

## 🧩 Step-by-Step Plan

### 1️⃣ Import and Prepare Data
- Load the MNIST dataset using TensorFlow/Keras  
- Normalize pixel values (0–255 → 0–1)  
- Reshape images to fit CNN input format `(28, 28, 1)`

### 2️⃣ Build and Train CNN
- Architecture:  
  **Conv2D → MaxPooling2D → Conv2D → Flatten → Dense → Dropout → Softmax**
- Compile with `adam` optimizer and `sparse_categorical_crossentropy` loss  
- Train and evaluate on test data  

### 3️⃣ Model Improvement
- Add **Dropout** for regularization  
- Add **Batch Normalization** for stability  
- Compare 2–3 architectures to find best accuracy  

### 4️⃣ Save and Use Model
- Export trained model as `model.h5`  
- Load model for predictions using `load_model()`  

### 5️⃣ Optional: Deployment
- Create a simple **Flask** or **Streamlit** web app to draw digits and recognize them in real time.

---

## 🧠 Skills Gained
- CNN design and tuning  
- Image preprocessing and normalization  
- Model evaluation and saving/loading  
- Basic model deployment using Flask/Streamlit  

---

## 📂 Folder Structure

