# MNIST and CIFAR-10 Classification using KNN, MLP, and CNN

**Author**: Arman Gökalp

## Project Description
This project focuses on the implementation of three distinct machine learning models - K-Nearest Neighbors (KNN), Multi-Layer Perceptron (MLP), and Convolutional Neural Network (CNN), for the classification of images from the MNIST and CIFAR-10 datasets.

### MNIST Classification
- **Objective**: Classify handwritten digits.
- **Models Used**: KNN, MLP, and CNN.
- **Libraries Utilized**: PyTorch, scikit-learn, numpy, matplotlib.
- **Key Techniques**:
  - Data normalization.
  - Model tuning with grid search (for KNN).
  - Neural network architecture design.
  - Hyperparameter optimization using Optuna.
  - Comprehensive model evaluation.

### CIFAR-10 Classification
- **Objective**: Classify images into ten different classes.
- **Models Used**: KNN, MLP, and CNN.
- **Key Techniques**:
  - PCA for dimensionality reduction (for KNN).
  - Designing neural network architectures suitable for image data.
  - Hyperparameter optimization.
  - Detailed model evaluation.

## Key Findings
- The **CNN model** outperformed others in both datasets, achieving a test accuracy of **98.69% for MNIST** and **75.14% for CIFAR-10**.
- KNN classifiers exhibited the lowest accuracy, underscoring the superiority of deep learning models in image classification tasks.
- Error analysis revealed distinct classification challenges, like the misclassification between classes 7 and 6 in MNIST, and between 'cat' and 'dog' in CIFAR-10.

## Repository Structure
1. **Data Preprocessing**: Scripts and instructions for loading and preprocessing the datasets.
2. **Model Implementation**: Separate Jupyter notebooks detailing the implementation of KNN, MLP, and CNN for both datasets.
3. **Model Evaluation**: Includes accuracy assessments, confusion matrices, and visualizations.
4. **Error Analysis**: Discusses common misclassifications and includes visualizations of misclassified images.
