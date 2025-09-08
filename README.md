# 🧠 Neural Network on MNIST Dataset

## 📌 Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) from the **MNIST dataset** using **TensorFlow/Keras**.  
The model achieves **~98% accuracy** on the test set with optimized hyperparameters.

---

## 📂 Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (28×28 pixels).
- Split:
  - Training: 51,000
  - Validation: 9,000
  - Test: 10,000

---

## ⚙️ Features
- Data preprocessing and augmentation  
- CNN architecture with Conv2D, MaxPooling2D, Dropout, Dense layers  
- Hyperparameter tuning using **Keras Tuner (RandomSearch)**  
- Early stopping and checkpoint saving  
- Evaluation with accuracy, precision, recall, F1-score, and confusion matrix  

---

## 📊 Results
- **Training Accuracy:** ~99%  
- **Validation Accuracy:** ~98%  
- **Test Accuracy:** ~98%  
- Misclassifications occur mostly between visually similar digits (e.g., 4 vs 9, 3 vs 5).

---

## 🚀 Future Work
- Experiment with deeper architectures (ResNet, EfficientNet).  
- Explore advanced optimizers and learning rate schedules.  
- Deploy the trained model in a real-time digit recognition app.  

---

## 📦 Requirements
- Python 3.8+  
- TensorFlow / Keras  
- scikit-learn  
- matplotlib  
- keras-tuner  

---

## 📜 License

This project is licensed under the MIT License.


