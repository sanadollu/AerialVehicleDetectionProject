# Aerial Vehicle Detection

This project addresses an image classification task using both traditional machine learning algorithms (HOG + ML) and a deep learning approach (CNN). 

## Objective

The goal is to build and compare multiple models for vehicle detection using:
- HOG features with SVM, XGBoost, and MLP classifiers
- A CNN trained directly on raw image data without manual feature extraction

## Dataset

- **Source**: [Kaggle aerial imagery dataset](https://www.kaggle.com/datasets/javiersanchezsoriano/traffic-images-captured-from-uavs)
- **Total Images**: 4,500 (1920x1080 resolution)
- **Classes**:
  - Cars: 11,650 instances
  - Motorcycles: 8,037 instances
  - No vehicle: 36,000 instances
    
Annotations were provided in `(class_id, x_center, y_center, width, height)` format.

## Methods Used

### Traditional ML Pipeline:
- Image cropping and resizing to 40x30
- Feature extraction via Histogram of Oriented Gradients (HOG)
- Classifiers:
  - **SVM** – good with high-dimensional data
  - **MLP** – learns nonlinear relationships
  - **XGBoost** – robust gradient boosting method
- K-Fold Cross-Validation applied for tuning

### CNN Pipeline:
- Model trained on full image crops without manual features
- Data normalized and resized
- Architecture includes convolution, pooling, and dense layers
- Trained using categorical cross-entropy loss and SGD

## File Descriptions

| File Name                    | Description                                                 |
|-----------------------------|-------------------------------------------------------------|
| `Dataset_creation.ipynb`    | Preprocessing, annotation parsing, image visualization      |
| `HOG_and_ML_Train.ipynb`    | HOG feature extraction and ML model training                |
| `HOG_and_ML_Test.ipynb`     | Model evaluation on test set using confusion matrix, F1, etc.|
| `CNN_Train_and_Test.ipynb`  | CNN implementation and performance tracking                 |
| `svm_model.joblib`          | Trained SVM model                                           |
| `mlp_model.joblib`          | Trained MLP model                                           |
| `xgb_model.joblib`          | Trained XGBoost model                                       |

## Metrics Used

- Accuracy
- F1 Score
- Precision & Recall
- Intersection over Union (IoU)
- Confusion Matrix

## Requirements
Below are the main Python packages required to run the project:
- numpy
- pandas
- matplotlib
- plotly
- opencv-python
- scikit-image
- scikit-learn
- xgboost
- joblib
- torch
- torchvision
- Pillow

## Developers
Sena Ezgi Anadollu

## About the Project
This project was developed as part of a Computer Vision course. The development was completed in April 2024.
