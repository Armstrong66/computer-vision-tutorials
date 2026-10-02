# Tutorial 01: How Neural Networks "See" Images

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Armstrong66/computer-vision-tutorials/blob/main/tutorials/01_how_neural_networks_see/TF_Visual_Deep_Dive_How_NNs_See.ipynb)

In this foundational deep-dive tutorial, we peel back the high-level abstractions of deep learning vision libraries (`Conv2D`, `MaxPool2D`) to examine the underlying mathematical and visual mechanisms of computer vision.

---

## 🎯 Learning Objectives

1. **Images as Numerical Tensors**: Understand how images are represented in computer memory as 3D arrays of shape $(H, W, C)$.
2. **Channel Dimensionality**:
   - **1-Channel**: Grayscale intensity maps (e.g., MNIST).
   - **2-Channel**: Spatial vector fields and directional gradients $(\partial I / \partial x, \partial I / \partial y)$.
   - **3-Channel**: Standard RGB color reflectance.
   - **4+ Channels**: Multispectral imagery and latent neural network feature maps.
3. **Channel Decomposition & Pairwise Comparison**:
   - Compare color channels in isolated pairs (Red vs. Green, Red vs. Blue, Green vs. Blue) with dedicated colorbars and pixel intensity histograms.
   - Reconstruct the complete color image via additive synthesis: $\mathbf{I}_{\text{RGB}} = [R, 0, 0] + [0, G, 0] + [0, 0, B]$.
4. **Preprocessing Transforms**:
   - Min-Max normalization ($[0, 255] \to [0.0, 1.0]$) and standardization ($\frac{x - \mu}{\sigma}$).
   - Perceptual luminance weighting ($Y = 0.299R + 0.587G + 0.114B$) vs. naive mean.
5. **The Mechanics of Convolutions**:
   - Handcrafted spatial filters: Horizontal/Vertical Sobel, Laplacian, Box Blur, Sharpen.
   - Pure NumPy implementation of 2D sliding window cross-correlation.
   - Mathematical proof of equivalence with TensorFlow `tf.nn.conv2d`.
   - Multi-channel 3D convolution: How a $(K_h, K_w, C_{\text{in}})$ filter collapses all channels into a 2D feature map.
6. **Spatial Mechanics**:
   - **Patches & Local Receptive Fields**: Visualizing sliding windows across scanlines.
   - **Padding**: Why 'valid' padding shrinks borders vs. how 'same' padding preserves resolution.
   - **Pooling**: Max Pooling (salient feature preservation) vs. Average Pooling (smoothing) vs. Global Average Pooling.
7. **Feature Hierarchies & Non-Linearity**:
   - **The Collapse Theorem**: Why deep linear networks without non-linearities collapse to a single linear regression.
   - **ReLU**: Thresholding, gating negative responses, and introducing activation sparsity.
   - **Layer Progression**: Visualizing Layer 1 (edges/gradients) vs. Layer 2 (motifs/textures) vs. Deep Layers (semantic abstractions).

---

## 💻 How to Run

### Run Online via Google Colab
Click the badge below to run the notebook on a cloud GPU with zero installation required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Armstrong66/computer-vision-tutorials/blob/main/tutorials/01_how_neural_networks_see/TF_Visual_Deep_Dive_How_NNs_See.ipynb)

### Run Locally
```bash
# Clone the repository
git clone https://github.com/Armstrong66/computer-vision-tutorials.git
cd computer-vision-tutorials

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook tutorials/01_how_neural_networks_see/TF_Visual_Deep_Dive_How_NNs_See.ipynb
```
