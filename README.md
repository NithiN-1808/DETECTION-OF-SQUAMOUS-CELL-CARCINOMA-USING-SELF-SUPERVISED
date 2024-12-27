# DETECTION-OF-SQUAMOUS-CELL-CARCINOMA-USING-SELF-SUPERVISED-LEARNING
## SimCLR - Contrastive Learning of Visual Representations in TensorFlow

This repository contains an implementation of **SimCLR** (Simple Contrastive Learning of Representations) using **TensorFlow 2.x**. SimCLR is a self-supervised learning method for learning visual representations without using labeled data. The primary goal of SimCLR is to learn effective image feature representations by contrasting positive and negative image pairs. These representations can be used for downstream tasks such as image classification, object detection, etc.

SimCLR utilizes a ResNet-based architecture and a projection head to map the features learned by the backbone into a latent space where contrastive loss (NT-Xent) is used to train the model.

## Project Overview

### Objective:
The main objective of this project is to implement and train the **SimCLR** model for self-supervised learning on image data. The model learns useful feature representations by contrasting augmented views of the same image against views from different images. Once pre-trained, these representations can be fine-tuned for specific downstream tasks like image classification.

### Key Components:
1. **Data Augmentation**: Apply a series of augmentations like random cropping, color jitter, and flipping to create positive pairs.
2. **SimCLR Model**: The backbone of the model is a ResNet architecture (e.g., ResNet-50 or ResNet-18), followed by a projection head that maps the learned features to a lower-dimensional space.
3. **Contrastive Loss**: **NT-Xent Loss** is used to ensure that the representations of positive pairs are close together in the embedding space, while negative pairs are far apart.
4. **Fine-tuning**: After pretraining on unlabeled data, the model is fine-tuned on labeled data for tasks like image classification.

---

## Installation

This project uses **TensorFlow 2.x**. The dependencies can be installed using `pip`.

### Requirements:
- **Python** >= 3.7
- **TensorFlow** >= 2.4
- **NumPy** >= 1.19.0
- **Matplotlib** (for visualization)
- **Pandas** (for data manipulation)
- **TensorFlow Datasets** (for loading datasets)

### Install the dependencies:

To install all the required dependencies, run the following command:

```bash
pip install tensorflow==2.9 numpy pandas matplotlib tensorflow-datasets
