# Automate Identification and Recognition of Handwritten Text from an Image

![CRNN Model](Images/AnimationHTR.gif)

This repository contains the implementation of a CRNN (Convolutional Recurrent Neural Network) model designed to detect and recognize handwritten text from images. The CRNN combines convolutional layers for feature extraction with recurrent layers for sequence modeling, making it well-suited for tasks involving sequential data like text.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Demo](#demo)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Handwritten text recognition is a challenging task due to the variability in handwriting styles, orientations, and the presence of noise in images. This project leverages a CRNN architecture to accurately detect and recognize handwritten text from images, making it suitable for applications such as digitizing handwritten documents, automated form processing, and more.

## Features

- **End-to-end text detection and recognition**: Automatically detect and recognize text from input images.
- **CRNN architecture**: Combines CNNs for feature extraction with RNNs for sequence modeling.
- **CTC loss function**: Utilizes Connectionist Temporal Classification (CTC) for sequence prediction without pre-segmented labels.
- **Preprocessing utilities**: Includes image preprocessing and augmentation utilities.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/VMD7/Automate-identification-and-recognition-of-handwritten-text-from-an-image
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Model Architecture

The CRNN model architecture consists of the following components:

- Convolutional Layers: Extract spatial features from input images.

- Recurrent Layers (Bidirectional LSTM): Model the sequential nature of the text.

- CTC Loss Function: Handles the alignment between predicted and actual sequences.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 32, 128, 1)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 128, 64)       640       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 64, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 64, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 32, 128)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 32, 256)        295168    
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 32, 256)        590080    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 32, 256)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 32, 512)        1180160   
_________________________________________________________________
batch_normalization_1 (Batch (None, 4, 32, 512)        2048      
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 4, 32, 512)        2359808   
_________________________________________________________________
batch_normalization_2 (Batch (None, 4, 32, 512)        2048      
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 32, 512)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 1, 31, 512)        1049088   
_________________________________________________________________
lambda_1 (Lambda)            (None, 31, 512)           0         
_________________________________________________________________
bidirectional_1 (Bidirection (None, 31, 512)           1574912   
_________________________________________________________________
bidirectional_2 (Bidirection (None, 31, 512)           1574912   
_________________________________________________________________
dense_1 (Dense)              (None, 31, 79)            40527     
=================================================================
Total params: 8,743,247
Trainable params: 8,741,199
Non-trainable params: 2,048
_________________________________________________________________
```
   





