# Deep Learning

## Overview

This module introduces the fundamental concepts of **Deep Learning**, focusing on **Artificial Neural Networks (ANN)** and **Generative Adversarial Networks (GAN)**. It provides the theoretical foundation behind neural networks and demonstrates how deep learning models can perform both classification and image generation tasks.

---

## Topics Covered

### Artificial Neural Networks (ANN)

- Biological inspiration of neural networks
- Artificial neuron 
- Input, Hidden and Output Layers
- Forward Propagation
- Activation Functions
- Loss Function
- Backpropagation
- Gradient Descent
- Model Training

### Generative Adversarial Networks (GAN)

- Introduction to GANs
- Generator Network
- Discriminator Network
- Adversarial Training
- Binary Cross Entropy (BCE) Loss
- Adam Optimizer
- Image Generation using GAN

---

## Files

| File | Description |
|------|-------------|
| **ANN_and_GAN_Overview.pdf** | Learning notes covering ANN and GAN concepts. |
| **gan_cifar10.py** | GAN implementation using PyTorch and the CIFAR-10 dataset. |
| **GAN_Results.pdf** | Training logs, generated images, and experimental observations. |

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

## GAN Workflow

```text
Random Noise 
            ↓
        Generator
            ↓
     Fake Images Generated
            ↓
      Discriminator
      ↙             ↘
Real Images     Fake Images
      ↓              ↓
  Classification (Real / Fake)
            ↓
      Loss Calculation
            ↓
    Backpropagation
            ↓
    Update Generator &
    Discriminator Weights
            ↺
      Repeat for Epochs
```

---

## Dataset

The GAN implementation uses the **CIFAR-10** dataset.

**Dataset Summary**

| Attribute | Value |
|-----------|------:|
| Images | 60,000 |
| Training Images | 50,000 |
| Test Images | 10,000 |
| Image Size | 32 × 32 |
| Channels | RGB |
| Classes | 10 |

---

## Experimental Setup

| Parameter | Value |
|----------|------:|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 0.0002 |
| Loss Function | Binary Cross Entropy (BCELoss) |
| Latent Dimension | 100 |
| Batch Size | 32 |
| Epochs | 10 |

---

## Results

The Generator and Discriminator were trained adversarially for **10 epochs** on the CIFAR-10 dataset.

The generated images demonstrate that the Generator successfully learned the basic data distribution. Although the generated images remain slightly blurry after 10 epochs, recognizable visual patterns begin to emerge, indicating successful adversarial learning.

The detailed training logs and generated sample images are provided in **GAN_Results.pdf**.

---

## Learning Outcomes

After completing this module, I gained practical experience in:

- Understanding Artificial Neural Networks.
- Understanding the complete deep learning training pipeline.
- Implementing a Generative Adversarial Network (GAN).
- Training Generator and Discriminator networks.
- Generating synthetic images using random latent vectors.
- Evaluating GAN training through Generator and Discriminator losses.

---

