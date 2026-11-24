# Melanoma Tumor Classification

A deep-learning image classification project that distinguishes benign and malignant melanoma lesions from dermoscopic images.

## Overview

Melanoma is a potentially fatal form of skin cancer. Because benign and malignant lesions can look similar, early and accurate classification can help support medical diagnosis.

This notebook uses transfer learning with MobileNetV2 and TensorFlow/Keras to perform binary image classification. It includes data preprocessing and augmentation, model training, validation, hyperparameter tuning, threshold optimization, and model interpretability.

## Dataset

The project uses the **Melanoma Skin Cancer Dataset - Benign vs Malignant**:

https://www.kaggle.com/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant

- 11,879 training images
  - 6,289 benign
  - 5,590 malignant
- 2,000 test images
  - 1,000 benign
  - 1,000 malignant

## Project Contents

- `MelanomaProject.ipynb`: complete training and evaluation workflow
- `LICENSE`: MIT license

## Model Approach

1. Load and normalize dermoscopic images.
2. Apply data augmentation to the training set.
3. Use MobileNetV2 with pre-trained ImageNet weights as a feature extractor.
4. Train a binary classification head.
5. Tune the learning rate, malignant class weight, and classification threshold.
6. Evaluate the model with accuracy, recall, AUC, and a confusion matrix.
7. Inspect predictions and visualize Grad-CAM explanations.

The hyperparameter-tuning stage uses a 1e-3 learning rate, a malignant class weight of 2.0, and a 0.4 malignant classification threshold. The notebook also describes reducing training to three epochs after the model begins to overfit.

## Evaluation

The notebook reports:

- Training and validation loss
- Training and validation accuracy
- Training and validation AUC
- ROC-AUC
- Benign and malignant recall
- Confusion matrix
- Examples of correct and incorrect predictions
- Grad-CAM heatmaps for model interpretation

## Requirements

The notebook uses:

- Python
- TensorFlow/Keras
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn
- OpenCV
- PIL

Install the packages in a Python environment, open `MelanomaProject.ipynb`, and run the notebook cells from top to bottom.

## Running on Kaggle

The notebook is configured for Kaggle paths in two places:

- Training data: `/kaggle/input/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant/train`
- Test data: `/kaggle/input/datasets/ailearner-researchlab/melanoma-skin-cancer-dataset-benign-vs-malignant/test`

To run elsewhere, update `TRAINING_DIR` and `TESTING_DIR` to point to local copies of the training and test folders.

## Related Project

This project was inspired by a previous COVID-19 chest X-ray image classification project:

https://github.com/gupta-reyansh/Covid_Detection_From_Chext_Xrays

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
