# 👁️ Computer Vision Tutorials: From Pixels to Deep Representations

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Armstrong66/computer-vision-tutorials/blob/main/tutorials/01_how_neural_networks_see/01_how_neural_networks_see_images.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

An intuitive, pedagogical, and visual tutorial series exploring how neural networks perceive, transform, and understand visual data. Built for students, researchers, and engineers seeking visual intuition alongside mathematical rigor.

---

## 📚 Curriculum Roadmap

| # | Tutorial | Topic & Highlights | Cloud Runner |
| :---: | :--- | :--- | :---: |
| **01** | [**How Neural Networks "See" Images**](tutorials/01_how_neural_networks_see/README.md) | Channels (1-ch grayscale, 2-ch vector fields, 3-ch RGB), pairwise channel comparison, convolutions from scratch in NumPy vs TensorFlow, patches, padding, pooling, and feature hierarchies. | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Armstrong66/computer-vision-tutorials/blob/main/tutorials/01_how_neural_networks_see/01_how_neural_networks_see_images.ipynb) |

---

## 🌟 Tutorial 01 Highlights: How Neural Networks "See" Images

- **The Anatomy of Channels**: Understand how images are represented as multidimensional numerical tensors ($\mathbb{R}^{H \times W \times C}$) across 1, 2, 3, and $N$ channels.
- **Dedicated Pairwise Channel Comparison**: Inspect isolated color channels in pairs (Red vs. Green, Red vs. Blue, Green vs. Blue) with dedicated heatmaps and intensity histograms, followed by complete additive RGB synthesis.
- **Preprocessing Transforms**: Min-Max scaling ($[0, 255] \to [0.0, 1.0]$), Z-score standardization, and perceptual luminance conversion ($Y = 0.299R + 0.587G + 0.114B$).
- **Convolutions from Scratch**: Step-by-step 2D sliding window implementation in pure NumPy, verified with machine-precision equivalence against TensorFlow `tf.nn.conv2d`.
- **Spatial Mechanics**: Receptive field patches, padding mechanics (`valid` vs. `same`), and pooling operations (Max vs. Average vs. Global Average).
- **The Power of Depth & Non-Linearity**: Why deep linear networks collapse into a single linear regression, how ReLU provides feature gating and sparsity, and visualization of intermediate feature maps across layers.
- **Interactive Student TODOs**: Hands-on exercises including custom diagonal edge kernels, feature map dimension formula implementation, channel gating filters, and manual max pooling.

---

## 🚀 Running the Tutorials

### Option 1: In the Cloud via Google Colab (Recommended — Zero Setup)
Click any **"Open in Colab"** badge above to launch the notebooks directly in Google Colab with free GPU acceleration.

### Option 2: Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/Armstrong66/computer-vision-tutorials.git
cd computer-vision-tutorials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook
```

---

## 📜 Acknowledgments

Inspired by foundational concepts and pedagogy in deep learning and computer vision education (including MIT 6.S191 and Stanford CS231n).

Licensed under the [MIT License](LICENSE.md).
