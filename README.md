# CIFAR-10 Classification with EWC and Adaptive EWC Optimizers

## Project Overview

This project explores the application of Elastic Weight Consolidation (EWC) and Adaptive EWC optimizers in the context of training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset. EWC is a technique used to mitigate catastrophic forgetting in continual learning scenarios by penalizing changes to important model parameters. Adaptive EWC extends this concept by dynamically adjusting the regularization strength based on gradient magnitudes. This project demonstrates the implementation and comparison of these techniques to improve model performance and stability over multiple tasks.

## Table of Contents

- [Project Description](#project-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [License](#license)

## Project Description

### Goals

1. **Train a CNN on CIFAR-10 Dataset:** Implement and train a CNN model on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes.
2. **Implement EWC and Adaptive EWC Optimizers:** Develop and apply EWC and Adaptive EWC optimizers to the CNN to assess their effectiveness in reducing catastrophic forgetting.
3. **Visualize Loss Landscape:** Generate and visualize the loss landscape of the model to understand the impact of different optimizers on the loss surface.

### Approach

- **CNN Model:** The model consists of two convolutional layers followed by two fully connected layers. This architecture is suitable for image classification tasks and provides a good balance between complexity and performance.
- **EWC Optimization:** EWC penalizes changes to model parameters that are important for previously learned tasks, helping to retain knowledge across different tasks.
- **Adaptive EWC Optimization:** This approach dynamically adjusts the EWC regularization strength based on the magnitude of gradients, potentially leading to better performance and stability.

## Requirements

To run this project, you will need:

- Python 3.6 or higher
- PyTorch 1.7.0 or higher
- torchvision 0.8.0 or higher
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install torch torchvision numpy matplotlib


### `IMPLEMENTATION_DETAILS.md`

```markdown
# Implementation Details

## CNN Model

The model architecture used in this project is a Convolutional Neural Network (CNN) designed for image classification on the CIFAR-10 dataset. The model is defined in the `CNN` class and includes:

- **Convolutional Layers:**
  - **Layer 1:** `nn.Conv2d(3, 32, kernel_size=3, padding=1)` followed by `nn.ReLU()` and `nn.MaxPool2d(kernel_size=2, stride=2)`.
  - **Layer 2:** `nn.Conv2d(32, 64, kernel_size=3, padding=1)` followed by `nn.ReLU()` and `nn.MaxPool2d(kernel_size=2, stride=2)`.

- **Fully Connected Layers:**
  - **Layer 1:** `nn.Linear(64 * 8 * 8, 128)` followed by `nn.ReLU()`.
  - **Layer 2:** `nn.Linear(128, 10)` to output the class probabilities.

## Optimizers

### EWCOptimizer

- **Purpose:** Incorporates Elastic Weight Consolidation (EWC) regularization to prevent catastrophic forgetting by penalizing significant changes to important parameters.
- **Implementation:** Extends `torch.optim.Adam` to include EWC regularization in the `step` function. The regularization term is added to the loss function based on Fisher information and original parameters.

### AdaptiveEWCOptimizer

- **Purpose:** An extension of `EWCOptimizer` that dynamically adjusts the EWC regularization strength (`ewc_lambda`) based on the average gradient magnitude.
- **Implementation:** Adjusts `ewc_lambda` after each optimizer step to increase or decrease the regularization based on the gradient norms.

## Training Procedure

1. **Training:** 
   - The model is trained using standard Adam optimization for a fixed number of epochs.
   - For EWC and Adaptive EWC, additional regularization terms are incorporated into the loss function.

2. **Computing Fisher Information:**
   - Fisher information is computed using the training data and is used to determine the importance of each parameter for the learned tasks.

3. **Applying EWC and Adaptive EWC:**
   - EWC regularization is applied to penalize deviations from important parameters.
   - Adaptive EWC adjusts the regularization strength based on gradient magnitudes, aiming for improved stability and performance.

## Visualization

### Loss Landscape

- **Purpose:** To visualize how the model's loss changes with perturbations in the parameter space.
- **Implementation:**
  - Generates random directions in the parameter space.
  - Perturbs the model parameters along these directions.
  - Computes and plots the loss over a grid of perturbations to visualize the loss landscape.

Feel free to adjust or expand these details to fit the specifics of your implementation or preferences.
