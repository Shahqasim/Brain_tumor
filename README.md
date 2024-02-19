Dateset is provided on this link https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri
Brain Tumor Detection using Pre-Trained DenseNet-121 model
This notebook demonstrates the detection of brain tumors using computer vision techniques. The task is performed using a pre-trained DenseNet-121 model with ImageNet weights. The notebook utilizes the Keras API and TensorFlow framework.

Table of Contents
Introduction
Installation
Usage
Dataset
Data Preprocessing
Model Training and Evaluation
Results
Contributing
License
Introduction
The goal of this notebook is to detect brain tumors in MRI images using a pre-trained DenseNet-121 model. The DenseNet-121 architecture is a deep convolutional neural network that has been trained on the ImageNet dataset, which contains a large variety of images from various categories. By utilizing transfer learning, we can leverage the pre-trained weights of DenseNet-121 to achieve accurate tumor detection.

Installation
To run this notebook, you need to have the following dependencies installed:

Python 3
TensorFlow
Keras
NumPy
OpenCV
PIL
SciPy
scikit-learn
imutils
matplotlib
seaborn
tqdm
plotly
tree (for directory structure visualization)
You can install these dependencies using pip install <dependency name>.

Usage
Clone this repository to your local machine or download the notebook file.
Install the required dependencies mentioned in the installation section.
Open the notebook using Jupyter Notebook or any compatible environment.
Run the notebook cells sequentially to execute the code and see the results.
Dataset
The dataset used in this notebook consists of MRI images of the brain, with and without tumors. The images are divided into three sets: train, test, and validation. Each set contains two subdirectories: "YES" for images with tumors and "NO" for images without tumors.

Data Preprocessing
The notebook applies a cropping technique to the MRI images to focus on the brain region. This technique helps remove unnecessary information from the images and improves the performance of the tumor detection model. The cropped images are then resized to a standard size for compatibility with the pre-trained DenseNet-121 model.

Model Training and Evaluation
The notebook utilizes the pre-trained DenseNet-121 model from the Keras library. The model is fine-tuned on the cropped and resized brain tumor dataset using the training set. The validation set is used for model evaluation and to tune hyperparameters if needed. The model is trained using an appropriate optimizer and loss function.

Results
The notebook provides visualizations and metrics to evaluate the performance of the brain tumor detection model. These include accuracy, precision, recall, F1 score, and a confusion matrix. The achieved accuracy on the test set is reported.

Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.
