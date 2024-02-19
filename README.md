# Brain Tumor Detection using Pre-Trained DenseNet-121 Model

## Introduction
This repository contains a Jupyter Notebook demonstrating the detection of brain tumors using computer vision techniques. The task is accomplished through the utilization of a pre-trained DenseNet-121 model with ImageNet weights. The notebook is implemented using the Keras API and TensorFlow framework.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this notebook, ensure you have the following dependencies installed:
- Python 3
- TensorFlow
- Keras
- NumPy
- OpenCV
- PIL
- SciPy
- scikit-learn
- imutils
- matplotlib
- seaborn
- tqdm
- plotly
- tree (for directory structure visualization)

You can install these dependencies using the command: `pip install <dependency name>`.

## Usage
1. Clone this repository to your local machine or download the notebook file.
2. Install the required dependencies mentioned in the installation section.
3. Open the notebook using Jupyter Notebook or any compatible environment.
4. Run the notebook cells sequentially to execute the code and observe the results.

## Dataset
The dataset used in this notebook comprises MRI images of the brain, both with and without tumors. The images are categorized into three sets: train, test, and validation. Each set consists of two subdirectories: "YES" for images with tumors and "NO" for images without tumors.Dateset is provided on this link https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri

## Data Preprocessing
The notebook applies a cropping technique to the MRI images to focus on the brain region, eliminating unnecessary information and enhancing the tumor detection model's performance. The cropped images are then resized to a standard size for compatibility with the pre-trained DenseNet-121 model.

## Model Training and Evaluation
The notebook employs the pre-trained DenseNet-121 model from the Keras library. The model undergoes fine-tuning on the cropped and resized brain tumor dataset using the training set. The validation set is utilized for model evaluation and hyperparameter tuning if necessary. An appropriate optimizer and loss function are employed during model training.

## Results
The notebook provides visualizations and metrics to assess the performance of the brain tumor detection model, including accuracy, precision, recall, F1 score, and a confusion matrix. The achieved accuracy on the test set is reported.

## Contributing
Contributions to this project are welcomed. If you encounter any issues or have suggestions for improvements, please submit a pull request or open an issue.


