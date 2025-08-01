# Galaxy Classification with Machine Learning

This project focuses on automatic galaxy classification using both traditional machine learning and deep learning techniques.  
The dataset contains 11,000 color galaxy images evenly split into 4 classes:

- Barred Spiral
- Elliptical
- Irregular
- Spiral

The project was implemented in PyTorch and explores both Logistic Regression with handcrafted features and ResNet18 transfer learning.

---

## Project Overview

1. **Dataset**
   - 11,000 RGB images resized to 224×224
   - 4 balanced galaxy classes
   - Train/Test split provided
   - Example preprocessing:
     ```python
     transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
     ])
     ```
   **Dataset Availability:**  
   The dataset used in this project cannot be uploaded to this repository due to licensing restrictions.  

2. **Feature Extraction for Logistic Regression**
   - **Color Histograms (64 bins per channel)**
   - **Edge Direction Histogram (Sobel filter, 64 bins)**
   - **RGB Co-occurrence Matrix (3-bit quantization, 10px offset)**
   - Combined into a **985-dimensional feature vector**

3. **Deep Learning: ResNet18 Transfer Learning**
   - Pretrained on ImageNet
   - Final FC layer replaced with 4 output neurons
   - Random rotation up to 180° for data augmentation
   - Color vs. Grayscale experiments conducted

---

## Results

| Model                          | Test Accuracy |
|--------------------------------|---------------|
| Logistic Regression (Color)     | 70.0%         |
| ResNet18 (Grayscale, No Rot.)   | 87.4%         |
| ResNet18 (Color, No Rot.)       | 88.2%         |
| ResNet18 (Grayscale + Rotation) | 92.0%         |
| **ResNet18 (Color + Rotation)** | **93.7%**     |

- **Rotation augmentation** significantly boosts accuracy  
- **Color images** provide a small but consistent improvement  
