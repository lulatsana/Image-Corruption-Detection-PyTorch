# Image-Corruption-Detection-PyTorch
An end-to-end Deep Learning Computer vision pipelines designed to classify clean vs. corrupted images(Gaussian blur and synthetic sensor noise) using PyTorch.
## Key Features
* **Custom Dataset Architecture:** Extends PyTorch `Dataset` to generate clean (Class 0) and corrupted (Class 1) samples dynamically.
* **Data Preprocessing & Augmentation:** Uses `torchvision.transforms` and `PIL.ImageFilter` to resize, normalize, and introduce controlled Gaussian noise and blur anomalies.
* **Convolutional Neural Network (CNN):** Custom 2D CNN architecture optimized with ReLU activation, Max Pooling, and linear classification layers.
* **GPU-Accelerated Performance:** Trained on an NVIDIA T4 GPU via Google Colab using `Adam` optimizer and `CrossEntropyLoss`.

## Performance Metrics
* **Training Dataset:** 50,000 CIFAR-10 baseline samples (80/20 train/test split)
* **Final Test Accuracy:** 99.95% on unseen validation batches

## Technical Stack
* **Language:** Python
* **Framework:** PyTorch (`torch`, `torchvision`, `torch.nn`)
* **Libraries:** NumPy, Pillow (PIL)
* **Environment:** Google Colab (CUDA GPU Acceleration)
