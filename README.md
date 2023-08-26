# Image Classification with CNNs on CIFAR-10 Dataset

This project demonstrates the process of training Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this project is to experiment with different CNN architectures, measure training times, and evaluate model performance using various metrics.

## Overview

The provided Jupyter Notebook demonstrates the following steps:

1. Loading and preprocessing the CIFAR-10 dataset.
2. Building a CNN model architecture using TensorFlow and Keras.
3. Compiling the model with suitable loss, optimizer, and evaluation metrics.
4. Training the model on the training dataset and validating it on the validation dataset.
5. Evaluating the trained model on the test dataset.
6. Calculating precision, recall, and accuracy scores for the test predictions.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. It is divided into 50,000 training images and 10,000 test images. 

## Results

The notebook includes visualizations of training and validation loss/accuracy trends over epochs, as well as a summary of the trained model's architecture. Additionally, it provides precision, recall, and accuracy scores for the model's performance on the test dataset.

## Notes

1. The notebook includes commented code blocks that can be used to either return only final CPU and wall times or per-epoch CPU and wall times. Uncomment the appropriate block based on your preference.
2. The model architecture and training parameters can be adjusted to experiment with different configurations and improve performance.
   
