## Image Classification Project
This repository contains Jupyter Notebooks for training, analyzing, and inferencing on an Resnet18 model on CIFAR 10 dataset. All the code is done in pytorch.

## Notebooks:

Train.ipynb: This notebook defines the model architecture, training pipeline, and hyperparameters. It loads the training data, preprocesses it, trains the model, and saves the trained model weights.

Train_Analysis.ipynb: This notebook analyzes the training process. It includes visualizations of the training and validation loss curves, accuracy plots and learning rate plots.

Inference_testlabels.ipynb: This notebook loads our pretrained model saved on files folder and tests it on a labeled cifar10 test dataset. It generates predictions and analyzes the results, computing metrics like accuracy, precision, recall, and F1-score.

Inference_test_nolabels.ipynb: This notebook loads a pre-trained model and tests it on an unlabeled dataset. It uses the model to predict class labels for the unseen images and generates a CSV file containing the image IDs and predicted labels in the required format for submission to a competition.

## Getting Started:


### Install dependencies:

Create a new virtual environment and install the required libraries using a package manager like pip. All the requirements are on requirements.txt

### Run the notebooks:

Open the Jupyter Notebook application and navigate to the repository directory. You can then open and run each notebook individually.


## Model Architecture Summary

The following table details the architecture of the convolutional neural network (CNN) model used for image classification:

| Layer (type)               | Output Shape         | Param # |
|----------------------------|----------------------|---------|
| Conv2d-1                   | [-1, 64, 32, 32]     | 1,728   |
| BatchNorm2d-2               | [-1, 64, 32, 32]     | 128     |
| Conv2d-3                   | [-1, 64, 32, 32]     | 36,864  |
| BatchNorm2d-4               | [-1, 64, 32, 32]     | 128     |
| Dropout-5                   | [-1, 64, 32, 32]     | 0       |
| Conv2d-6                   | [-1, 64, 32, 32]     | 36,864  |
| BatchNorm2d-7               | [-1, 64, 32, 32]     | 128     |
| Dropout-8                   | [-1, 64, 32, 32]     | 0       |
| BasicBlock-9                | [-1, 64, 32, 32]     | 0       |
| ... (repeat for layers 10-46) | ...                   | ...     |
| Linear-47                   | [-1, 10]             | 5,130   |
| **Total Params:**           | **4,977,226**        |         |
| **Trainable Params:**        | **4,977,226**        |         |
| **Non-trainable Params:**   | **0**                 |         |

**Additional Information:**

* Input size (MB): 0.01
* Forward/backward pass size (MB): 12.38
* Params size (MB): 18.99
* Estimated Total Size (MB): 31.37

**Note:** The repeated layer information (...) is omitted for brevity. 
