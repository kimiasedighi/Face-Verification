
# Siamese Neural Network for Image Similarity

This repository contains an implementation of a **Siamese Neural Network** for image similarity tasks. The model is trained using **contrastive loss** to distinguish between images of the same class and images from different classes.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [References](#references)

## Overview

The Siamese Network consists of two identical neural networks, both sharing the same weights and architecture. Each network takes an image as input, processes it through convolutional layers, and returns a feature vector. The Euclidean distance between the feature vectors of the two images is then computed to determine their similarity.

The model is trained using a contrastive loss function, which minimizes the distance between feature vectors of similar images and maximizes the distance between those of different images.

## Dataset

We use a **Faces** dataset provided in this project, containing grayscale images of human faces. The dataset is split into training and testing sets.

### Dataset Download

The dataset is automatically downloaded and extracted using the following commands:

```bash
!wget https://github.com/maticvl/dataHacker/raw/master/DATA/at%26t.zip
!unzip "at&t.zip" -d .
```

## Model Architecture

The Siamese Network has two major components:

1. **Convolutional Layers**: Used for feature extraction.
   - Conv2D layers with ReLU activations and MaxPooling for downsampling.
   
2. **Fully Connected Layers**: Used for comparison of feature vectors.
   - A series of linear layers with ReLU activations that output a 2-dimensional vector.

The network takes two images as input, processes each image through the network, and computes the Euclidean distance between the output feature vectors.

## Installation

To run this project, make sure you have the following dependencies installed:

- `torch`
- `torchvision`
- `PIL`
- `matplotlib`
- `numpy`

You can install the required dependencies using pip:

```bash
pip install torch torchvision pillow matplotlib numpy
```

## Training

The Siamese Network is trained on the provided faces dataset using **contrastive loss**. To begin training, follow these steps:

1. **Data Preprocessing**: The images are resized to (100x100) and converted to grayscale.
2. **Training Loop**: 
   - The model is trained over 100 epochs.
   - A batch size of 64 is used.
   - The Adam optimizer is used with a learning rate of 0.0005.

### Key Training Code:

```python
for epoch in range(100):
    for i, (img0, img1, label) in enumerate(train_dataloader, 0):
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f"Epoch number {epoch}
 Current loss {loss_contrastive.item()}
")
```

## Testing

After training, the network can be tested on the **testing dataset** to verify its performance. The network is provided with pairs of images, and the Euclidean distance between their feature vectors is calculated to determine their similarity.

### Key Testing Code:

```python
for i in range(10):
    _, x1, label2 = next(data_iter)
    output1, output2 = net(x0.cuda(), x1.cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.3f}')
```

## Results

- The **Euclidean distance** between two images is used to determine their similarity. A lower distance indicates a higher similarity between the images, and vice versa.
- The model can visually display the similarity/dissimilarity between the images.

## References

This project is based on the tutorial and dataset provided by DataHacker:

- [Siamese Neural Network Tutorial](https://github.com/maticvl/dataHacker)
