import json
import os

def create_notebook():
    cells = []

    def md(source_text, cell_id=None):
        return {
            "cell_type": "markdown",
            "metadata": {"id": cell_id} if cell_id else {},
            "source": [line + "\n" for line in source_text.strip().split("\n")]
        }

    def code(source_text, cell_id=None):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": cell_id} if cell_id else {},
            "outputs": [],
            "source": [line + "\n" for line in source_text.strip().split("\n")]
        }

    # =========================================================================
    # Header & Copyright
    # =========================================================================
    cells.append(md("""<table align="center">
  <td align="center"><a target="_blank" href="https://github.com/Armstrong66/computer-vision-tutorials">
        <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" style="padding-bottom:5px; height:50px;" />
      <br/><b>Computer Vision Tutorials</b></a></td>
  <td align="center"><a target="_blank" href="https://colab.research.google.com/github/Armstrong66/computer-vision-tutorials/blob/main/tutorials/01_how_neural_networks_see/TF_Visual_Deep_Dive_How_NNs_See.ipynb">
        <img src="https://i.ibb.co/2P3SLwK/colab.png" style="padding-bottom:5px;" /><br/><b>Run in Google Colab</b></a></td>
  <td align="center"><a target="_blank" href="https://github.com/Armstrong66/computer-vision-tutorials">
        <img src="https://i.ibb.co/xfJbPmL/github.png" height="50px" style="padding-bottom:5px;" /><br/><b>View Source on GitHub</b></a></td>
</table>

# Copyright Information
"""))

    cells.append(code("""# Copyright 2026 Armstrong66. Computer Vision Tutorials. All Rights Reserved.
#
# Licensed under the MIT License. You may not use this file except in compliance
# with the License.
#
# Repository: https://github.com/Armstrong66/computer-vision-tutorials
#"""))

    cells.append(md("""# Tutorial 01: Computer Vision & Visual Representation
# How Neural Networks "See" Images: Convolutions, Channels, and Feature Hierarchies

Welcome to this interactive tutorial on Computer Vision! In modern machine learning frameworks, it is easy to define a convolutional network with high-level API calls like `tf.keras.layers.Conv2D(32, (3, 3))`. However, to build true intuition for computer vision, we need to look beneath the surface.

How does a neural network actually process an image? What does a "channel" really represent in raw memory? How do filters extract features from pixels, and why do multi-layered networks build hierarchies from simple edges to rich semantic concepts?

### Laboratory Roadmap:
1. **The Anatomy of Images & Channels**: Tensors, 1D/2D/3D/4D channel representations, channel decomposition, pairwise comparisons, and preprocessing transformations.
2. **The Mechanics of Convolutions**: Handcrafted classical filters, pure NumPy 2D sliding windows, numerical equivalence with TensorFlow, and multi-channel 3D filter integration.
3. **Spatial Transformations**: Patch formation (local receptive fields), padding mechanics (`valid` vs. `same`), and pooling strategies (`max` vs. `average` vs. `global`).
4. **Hierarchies of Vision**: Low-level vs. high-level features, the critical role of non-linear activations (ReLU), and visualizing intermediate activations inside deep neural networks.
5. **Interactive Exercises & TODOs**: Hands-on coding challenges to test your understanding!
"""))

    # =========================================================================
    # Section 0: Environment Setup
    # =========================================================================
    cells.append(md("""## 0.0 Setup and Dependencies

Let's import TensorFlow, NumPy, Matplotlib, and image utilities. This notebook is completely self-contained and runs seamlessly in Google Colab or your local machine."""))

    cells.append(code("""import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
from PIL import Image

# Configure plotting parameters for clean figures
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['image.interpolation'] = 'nearest'

print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpu_devices) > 0} ({gpu_devices if gpu_devices else 'Running on CPU'})")
"""))

    # =========================================================================
    # Section 1: Image Representation & The Anatomy of Channels
    # =========================================================================
    cells.append(md("""---
# Part 1: How Computers "See" Images — Tensors and Channels

To a human observer, an image is a visual scene with recognizable objects, colors, and shadows. To a computer and a neural network, however, an image is strictly a **multidimensional numerical tensor**:
$$\\mathbf{X} \\in \\mathbb{R}^{H \\times W \\times C}$$
where:
* **$H$ (Height)**: The vertical spatial resolution (number of pixel rows).
* **$W$ (Width)**: The horizontal spatial resolution (number of pixel columns).
* **$C$ (Channels)**: The number of distinct feature or spectral measurements recorded at every spatial coordinate $(y, x)$.

Each element $X[y, x, c]$ represents a numerical intensity. In standard digital imagery, pixel values are 8-bit integers ranging from **0** (no intensity / black) to **255** (maximum intensity / full saturation).

### What do different channel counts ($C$) represent?
* **$C = 1$ (Single-Channel / Grayscale)**: A 2D spatial grid where each pixel is a single scalar brightness intensity (e.g., MNIST digits, infrared thermography, single medical X-ray).
* **$C = 2$ (Two-Channel / Vector Fields)**: Specialized vision representations where each pixel encodes a 2D vector $(v_x, v_y)$. Examples include **optical flow** (velocity of motion across frames), **spatial image gradients** $\\left(\\frac{\\partial I}{\\partial x}, \\frac{\\partial I}{\\partial y}\\right)$, and complex frequency spectrograms (Real, Imaginary components).
* **$C = 3$ (Three-Channel / Color RGB)**: Standard additive color imagery (Red, Green, Blue). Each channel isolates light reflectance across a specific optical wavelength band.
* **$C \\ge 4$ (Multi-Channel & Latent Feature Volumes)**: Multispectral/satellite imagery (e.g., Red, Green, Blue, Near-Infrared), RGBA (with Alpha transparency), and crucially, **intermediate feature maps inside Convolutional Neural Networks** (e.g., $32, 64, 128, 512$ feature channels)!

Let's load a sample image and inspect its tensor structure across these dimensions."""))

    cells.append(code("""# Helper to load a high-quality test image (with local and remote fallback)
def load_sample_image():
    # Local paths in this repository
    candidates = [
        'data/sample.jpg',
        'sample_image.jpg'
    ]
    for path in candidates:
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            return np.array(img)
    
    # Remote fallback from repository raw URL
    url = "https://raw.githubusercontent.com/Armstrong66/computer-vision-tutorials/main/data/sample.jpg"
    local_path = "sample_image.jpg"
    try:
        urllib.request.urlretrieve(url, local_path)
        img = Image.open(local_path).convert('RGB')
        return np.array(img)
    except Exception as e:
        # Synthetic fallback if running completely offline without sample file
        print("Synthesizing rich test image...")
        x = np.linspace(-3, 3, 200)
        y = np.linspace(-3, 3, 200)
        xx, yy = np.meshgrid(x, y)
        r = np.clip(np.sin(xx**2 + yy**2) * 127 + 128, 0, 255).astype(np.uint8)
        g = np.clip(np.cos(xx * 2) * 127 + 128, 0, 255).astype(np.uint8)
        b = np.clip(np.sin(yy * 3) * 127 + 128, 0, 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

rgb_image = load_sample_image()

# Crop or resize to a clean square for clear visualization
h, w, c = rgb_image.shape
min_dim = min(h, w)
rgb_image = rgb_image[:min_dim, :min_dim, :]
if min_dim > 300:
    pil_img = Image.fromarray(rgb_image).resize((256, 256))
    rgb_image = np.array(pil_img)

print(f"Image Tensor Shape: {rgb_image.shape} (Height={rgb_image.shape[0]}, Width={rgb_image.shape[1]}, Channels={rgb_image.shape[2]})")
print(f"Data type: {rgb_image.dtype}")
print(f"Pixel Range: min={rgb_image.min()}, max={rgb_image.max()}")
"""))

    cells.append(md("""### 1.1 Synthesizing 1-Channel, 2-Channel, and 3-Channel Images

Let's explicitly visualize what **1-Channel**, **2-Channel**, and **3-Channel** tensors look like to clarify the role of channel dimensionality:"""))

    cells.append(code("""# 1. One-Channel (Grayscale): Scalar Intensity Map
# Using standard perceptual luminance weights: 0.299 R + 0.587 G + 0.114 B
gray_1ch = (0.2989 * rgb_image[:,:,0] + 0.5870 * rgb_image[:,:,1] + 0.1140 * rgb_image[:,:,2]).astype(np.uint8)

# 2. Two-Channel (Vector Field / Gradient): dx and dy spatial derivatives
# Channel 0: Horizontal gradient; Channel 1: Vertical gradient
grad_x = np.gradient(gray_1ch.astype(np.float32), axis=1)
grad_y = np.gradient(gray_1ch.astype(np.float32), axis=0)
two_channel_tensor = np.stack([grad_x, grad_y], axis=-1)

# Plotting the three representations
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1-Channel visualization
im0 = axes[0].imshow(gray_1ch, cmap='gray')
axes[0].set_title(f"1-Channel (Grayscale)\\nShape: {gray_1ch.shape} (H, W)", fontsize=13)
axes[0].axis('off')
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# 2-Channel visualization: Directional Gradient Magnitude & Flow
# Representing Channel 0 (dx) as Red and Channel 1 (dy) as Green, with Blue=0
norm_gx = np.clip((grad_x - grad_x.min()) / (grad_x.max() - grad_x.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
norm_gy = np.clip((grad_y - grad_y.min()) / (grad_y.max() - grad_y.min() + 1e-8) * 255, 0, 255).astype(np.uint8)
two_ch_visual = np.stack([norm_gx, norm_gy, np.zeros_like(norm_gx)], axis=-1)
axes[1].imshow(two_ch_visual)
axes[1].set_title(f"2-Channel (Spatial Vector Field)\\nCh0: $\\\\partial I/\\\\partial x$ (Red), Ch1: $\\\\partial I/\\\\partial y$ (Green)\\nShape: {two_channel_tensor.shape}", fontsize=13)
axes[1].axis('off')

# 3-Channel visualization
axes[2].imshow(rgb_image)
axes[2].set_title(f"3-Channel (Full RGB Color)\\nCh0: R, Ch1: G, Ch2: B\\nShape: {rgb_image.shape}", fontsize=13)
axes[2].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""## 1.2 Channel Decomposition & Pairwise Comparison Demo

A common misunderstanding is that each channel of an RGB image is already a "colored" picture. In reality:
* **Each individual channel is simply a 2D scalar intensity matrix!**
* The Red channel measures *only* the intensity of red wavelength light reflecting off surfaces.
* To avoid visual clutter and appreciate differences clearly, we first inspect the channels **in pairwise comparisons (comparing no more than 2 at a time)**, before viewing all three alongside the full reconstructed composite.

In the cell below, observe how different regions light up:
* Notice how skin tones, warm objects, and lights show high intensity in Red, while shadows or blue surfaces appear dark.
* Notice how foliage or ambient illumination reflects strongly in Green.
* Observe the distinct intensity histogram beneath each pair to quantify the distribution differences."""))

    cells.append(code("""# Extract individual channel intensity matrices (each is shape H x W)
R_channel = rgb_image[:, :, 0]
G_channel = rgb_image[:, :, 1]
B_channel = rgb_image[:, :, 2]

def compare_two_channels(ch1_data, ch1_name, ch1_cmap,
                         ch2_data, ch2_name, ch2_cmap,
                         title_prefix="Pairwise Channel Comparison"):
    \"\"\"
    Compares exactly two channels side-by-side with colorbars and intensity histograms.
    \"\"\"
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    
    # Channel 1 Spatial Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(ch1_data, cmap=ch1_cmap, vmin=0, vmax=255)
    ax1.set_title(f"{ch1_name} Channel (Spatial Intensities)", fontsize=13, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Intensity (0-255)")
    
    # Channel 2 Spatial Heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(ch2_data, cmap=ch2_cmap, vmin=0, vmax=255)
    ax2.set_title(f"{ch2_name} Channel (Spatial Intensities)", fontsize=13, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Intensity (0-255)")
    
    # Channel 1 Histogram
    ax_hist1 = fig.add_subplot(gs[1, 0])
    c1 = 'red' if 'Red' in ch1_name else ('green' if 'Green' in ch1_name else 'blue')
    ax_hist1.hist(ch1_data.ravel(), bins=50, color=c1, alpha=0.7)
    ax_hist1.set_xlim([0, 255])
    ax_hist1.set_xlabel("Pixel Intensity Value")
    ax_hist1.set_ylabel("Pixel Count")
    ax_hist1.grid(alpha=0.3)
    
    # Channel 2 Histogram
    ax_hist2 = fig.add_subplot(gs[1, 1])
    c2 = 'red' if 'Red' in ch2_name else ('green' if 'Green' in ch2_name else 'blue')
    ax_hist2.hist(ch2_data.ravel(), bins=50, color=c2, alpha=0.7)
    ax_hist2.set_xlim([0, 255])
    ax_hist2.set_xlabel("Pixel Intensity Value")
    ax_hist2.set_ylabel("Pixel Count")
    ax_hist2.grid(alpha=0.3)
    
    fig.suptitle(f"{title_prefix}: {ch1_name} vs. {ch2_name}", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

# --- DEMO 1: Compare Pair 1 (Red vs. Green) ---
print(">>> DEMO 1: Comparing Red vs. Green Channel")
compare_two_channels(R_channel, "Red (R)", "Reds", G_channel, "Green (G)", "Greens")

# --- DEMO 2: Compare Pair 2 (Red vs. Blue) ---
print(">>> DEMO 2: Comparing Red vs. Blue Channel")
compare_two_channels(R_channel, "Red (R)", "Reds", B_channel, "Blue (B)", "Blues")

# --- DEMO 3: Compare Pair 3 (Green vs. Blue) ---
print(">>> DEMO 3: Comparing Green vs. Blue Channel")
compare_two_channels(G_channel, "Green (G)", "Greens", B_channel, "Blue (B)", "Blues")
"""))

    cells.append(md("""### 1.3 Full Channel Synthesis: All Channels Reconstructed

Now that we have analyzed the channels in pairs, let's assemble all three individual channels alongside the final reconstructed color image. Notice:
1. Each channel can be isolated as a **pure monochromatic component** by zeroing out the other two channels (e.g. $[R, 0, 0]$).
2. Adding these three tensors together:
$$\\mathbf{I}_{\\text{RGB}} = \\begin{bmatrix} R \\\\ 0 \\\\ 0 \\end{bmatrix} + \\begin{bmatrix} 0 \\\\ G \\\\ 0 \\end{bmatrix} + \\begin{bmatrix} 0 \\\\ 0 \\\\ B \\end{bmatrix}$$
perfectly reconstructs the rich color image!"""))

    cells.append(code("""# Construct isolated 3D tensors for each channel
zeros = np.zeros_like(R_channel)
only_red   = np.stack([R_channel, zeros, zeros], axis=-1)
only_green = np.stack([zeros, G_channel, zeros], axis=-1)
only_blue  = np.stack([zeros, zeros, B_channel], axis=-1)

# Verify mathematically that the sum equals the original
reconstructed = only_red + only_green + only_blue
assert np.array_equal(reconstructed, rgb_image), "Decomposition check failed!"

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(only_red)
axes[0].set_title("Isolated Red Component\\n$[R, 0, 0]$", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(only_green)
axes[1].set_title("Isolated Green Component\\n$[0, G, 0]$", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(only_blue)
axes[2].set_title("Isolated Blue Component\\n$[0, 0, B]$", fontsize=12, fontweight='bold')
axes[2].axis('off')

axes[3].imshow(reconstructed)
axes[3].set_title("Synthesized Composite RGB\\n$[R, 0, 0] + [0, G, 0] + [0, 0, B]$", fontsize=12, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""## 1.4 Preprocessing and Transformative Effects

Before an image enters a neural network, it is almost **never** passed in its raw $[0, 255]$ integer form. Why?
1. **Vanishing/Exploding Gradients**: Large input values ($x \\approx 255$) multiplied by initial network weights cause activations to explode into saturation zones of activation functions (like sigmoid or tanh) or cause massive, unstable gradient steps.
2. **Zero-Centering & Conditioning**: Unnormalized positive inputs cause all gradients of weights to have the same sign (all positive or all negative), leading to inefficient zig-zagging gradient descent trajectories.

### Standard Preprocessing Techniques:
* **Min-Max Rescaling**:
  $$x_{\\text{scaled}} = \\frac{x}{255.0} \\in [0.0, 1.0]$$
* **Standardization (Z-score normalization)**:
  $$x_{\\text{std}} = \\frac{x - \\mu}{\\sigma}$$
* **Perceptual Grayscale Conversion**:
  $$Y = 0.2989 R + 0.5870 G + 0.1140 B$$
  *(Note: Human eyes are much more sensitive to green wavelengths than blue, so an unweighted average $(R+G+B)/3$ causes distorted perceived luminance).*
* **Gaussian Smoothing (Noise Reduction)**: Attenuates high-frequency sensor noise before edge feature extraction.

Let's visualize the impact of these transformations:"""))

    cells.append(code("""# 1. Rescaling to [0, 1]
img_scaled = rgb_image / 255.0

# 2. Standardization (zero mean, unit variance)
img_std = (img_scaled - np.mean(img_scaled)) / (np.std(img_scaled) + 1e-7)

# 3. Simple Average vs Perceptual Grayscale
unweighted_gray = np.mean(img_scaled, axis=-1)
perceptual_gray = 0.2989 * img_scaled[:,:,0] + 0.5870 * img_scaled[:,:,1] + 0.1140 * img_scaled[:,:,2]
diff_gray = np.abs(perceptual_gray - unweighted_gray)

# Plot comparison
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

axes[0].imshow(img_scaled)
axes[0].set_title(f"Normalized $[0, 1]$\\nMean={img_scaled.mean():.2f}, Std={img_scaled.std():.2f}", fontsize=11)
axes[0].axis('off')

axes[1].imshow(perceptual_gray, cmap='gray')
axes[1].set_title("Perceptual Luminance $Y$\\n$0.299R + 0.587G + 0.114B$", fontsize=11)
axes[1].axis('off')

axes[2].imshow(unweighted_gray, cmap='gray')
axes[2].set_title("Naive Mean Grayscale\\n$(R + G + B) / 3$", fontsize=11)
axes[2].axis('off')

im3 = axes[3].imshow(diff_gray, cmap='magma')
axes[3].set_title("Absolute Difference Map\\n$|Y_{\\mathrm{perceptual}} - Y_{\\mathrm{naive}}|$", fontsize=11)
axes[3].axis('off')
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# Plot pixel distribution shifts
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.hist(rgb_image.ravel(), bins=60, density=True, alpha=0.5, label=f"Raw Integers $[0, 255]$: Mean={rgb_image.mean():.1f}, Std={rgb_image.std():.1f}")
ax.hist(img_scaled.ravel(), bins=60, density=True, alpha=0.5, label=f"Rescaled $[0, 1]$: Mean={img_scaled.mean():.2f}, Std={img_scaled.std():.2f}")
ax.hist(img_std.ravel(), bins=60, density=True, alpha=0.5, label=f"Standardized: Mean={img_std.mean():.2f}, Std={img_std.std():.2f}")
ax.set_title("Pixel Value Distribution Under Preprocessing Transformations", fontweight='bold')
ax.set_xlabel("Numerical Value")
ax.set_ylabel("Probability Density")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # Section 2: Convolutions: The Mathematics & Mechanics
    # =========================================================================
    cells.append(md("""---
# Part 2: Convolutions — How Filters/Kernels Work

In a standard fully connected layer, every input pixel is multiplied by a distinct weight connected to every neuron. For a $1000 \\times 1000$ RGB image with 100 hidden neurons, that would require:
$$(1000 \\times 1000 \\times 3) \\times 100 = 300,000,000 \\text{ parameters!}$$
This suffers from parameter explosion and completely ignores spatial locality.

### The Convolution Solution:
1. **Local Receptive Fields**: A neuron only looks at a small spatial window (e.g., $3 \\times 3$ or $5 \\times 5$ pixels).
2. **Weight Sharing (Translation Equivariance)**: The exact same kernel is swept across every spatial location in the image. If an edge detector works in the top-left corner, it works equally well in the bottom-right corner!

### Mathematical Formulation
Given a 2D image $\\mathbf{I}$ and a kernel $\\mathbf{K}$ of spatial size $k_h \\times k_w$:
$$S(i, j) = (\\mathbf{I} * \\mathbf{K})(i, j) = \\sum_{m=0}^{k_h-1} \\sum_{n=0}^{k_w-1} \\mathbf{I}(i + m, j + n) \\mathbf{K}(m, n) + b$$

At every location $(i, j)$:
1. Extract the image **patch** of size $k_h \\times k_w$.
2. Compute the **element-wise product** between the patch and the kernel weights.
3. **Sum** all products together into a single scalar response value.
4. Add the bias term $b$.

Let's implement this step-by-step in pure NumPy to demystify the operation completely!"""))

    cells.append(code("""# Define classical handcrafted vision kernels to understand what filters detect
KERNELS = {
    "Identity (Pass-through)": np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ], dtype=np.float32),
    
    "Box Blur (Low-pass smoothing)": np.array([
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
    ], dtype=np.float32),
    
    "Sobel Horizontal (Horizontal Edges $\\\\partial I / \\\\partial y$)": np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float32),
    
    "Sobel Vertical (Vertical Edges $\\\\partial I / \\\\partial x$)": np.array([
        [-1,  0,  1],
        [-2,  0,  2],
        [-1,  0,  1]
    ], dtype=np.float32),
    
    "Laplacian (Ridge & Edge Detection)": np.array([
        [ 0,  1,  0],
        [ 1, -4,  1],
        [ 0,  1,  0]
    ], dtype=np.float32),
    
    "Sharpen (High-boost filter)": np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)
}

print(f"Loaded {len(KERNELS)} classical kernels for analysis.")
"""))

    cells.append(md("""### 2.1 Implementing Manual 2D Convolution in Pure NumPy

Let's write a function that performs the 2D sliding-window convolution explicitly. No hidden black boxes!"""))

    cells.append(code("""def manual_conv2d(image_2d, kernel_2d, stride=1, padding='valid'):
    \"\"\"
    Performs a 2D cross-correlation convolution using pure NumPy.
    
    Args:
        image_2d: 2D numpy array of shape (H, W)
        kernel_2d: 2D numpy array of shape (Kh, Kw)
        stride: integer step size for sliding window
        padding: 'valid' (no padding) or 'same' (zero padding to preserve dimensions)
    Returns:
        output_feature_map: 2D numpy array
    \"\"\"
    H, W = image_2d.shape
    Kh, Kw = kernel_2d.shape
    
    # 1. Handle Padding
    if padding.lower() == 'same':
        pad_h = (Kh - 1) // 2
        pad_w = (Kw - 1) // 2
        padded_image = np.pad(image_2d, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif padding.lower() == 'valid':
        pad_h, pad_w = 0, 0
        padded_image = image_2d
    else:
        raise ValueError(f"Unknown padding type: {padding}")
        
    padded_H, padded_W = padded_image.shape
    
    # Calculate output dimensions
    out_H = (padded_H - Kh) // stride + 1
    out_W = (padded_W - Kw) // stride + 1
    output_feature_map = np.zeros((out_H, out_W), dtype=np.float32)
    
    # 2. Slide the kernel across every spatial patch
    for i in range(out_H):
        for j in range(out_W):
            row_start = i * stride
            row_end = row_start + Kh
            col_start = j * stride
            col_end = col_start + Kw
            
            # Extract local patch
            patch = padded_image[row_start:row_end, col_start:col_end]
            
            # Element-wise multiplication followed by summation
            output_feature_map[i, j] = np.sum(patch * kernel_2d)
            
    return output_feature_map

# Test our manual convolution on the grayscale image with each classical kernel
test_input = perceptual_gray.astype(np.float32)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

for idx, (kernel_name, kernel_mat) in enumerate(KERNELS.items()):
    feature_map = manual_conv2d(test_input, kernel_mat, stride=1, padding='same')
    
    axes[idx].imshow(feature_map, cmap='gray')
    axes[idx].set_title(f"{kernel_name}", fontsize=11, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""### 2.2 Numerical Equivalence Check: NumPy vs. TensorFlow `Conv2D`

To prove that `manual_conv2d` is performing the exact computation that TensorFlow does in production, let's run both on the exact same tensor and compute the maximum numerical discrepancy."""))

    cells.append(code("""# Select a kernel (Sobel Horizontal)
test_kernel = KERNELS["Sobel Horizontal (Horizontal Edges $\\\\partial I / \\\\partial y$)"]

# 1. Run our manual pure-NumPy convolution
manual_result = manual_conv2d(test_input, test_kernel, stride=1, padding='same')

# 2. Run TensorFlow's tf.nn.conv2d
# TensorFlow requires tensors of shape: (batch_size, height, width, in_channels)
# and filters of shape: (filter_height, filter_width, in_channels, out_channels)
tf_image = tf.constant(test_input[np.newaxis, :, :, np.newaxis], dtype=tf.float32)
tf_filter = tf.constant(test_kernel[:, :, np.newaxis, np.newaxis], dtype=tf.float32)

tf_result = tf.nn.conv2d(tf_image, tf_filter, strides=[1, 1, 1, 1], padding='SAME')
tf_result_np = tf.squeeze(tf_result).numpy()

# 3. Compute absolute error
max_abs_error = np.max(np.abs(manual_result - tf_result_np))
print(f"Manual Convolution Output Shape:     {manual_result.shape}")
print(f"TensorFlow Conv2D Output Shape:       {tf_result_np.shape}")
print(f"Maximum Absolute Numerical Error:    {max_abs_error:.2e}")

assert max_abs_error < 1e-5, "Numerical mismatch detected between NumPy and TensorFlow!"
print("SUCCESS: NumPy implementation matches TensorFlow Conv2D with machine precision!")
"""))

    cells.append(md("""### 2.3 Convolutions Across Multiple Channels ($C_{in} > 1$)

What happens when an input image has 3 color channels ($R, G, B$) or 64 feature channels?

**Crucial Rule**: When the input has $C_{in}$ channels, every convolutional filter is **three-dimensional**:
$$\\mathbf{K} \\in \\mathbb{R}^{K_h \\times K_w \\times C_{in}}$$
At each spatial position $(i, j)$:
1. The filter extracts a 3D patch: $\\mathbf{P} \\in \\mathbb{R}^{K_h \\times K_w \\times C_{in}}$.
2. It computes the dot product across **all $C_{in}$ channels simultaneously**:
$$S(i, j) = \\sum_{c=0}^{C_{in}-1} \\sum_{m=0}^{K_h-1} \\sum_{n=0}^{K_w-1} \\mathbf{P}(m, n, c) \\mathbf{K}(m, n, c) + b$$
3. This collapses all channels into **one single 2D feature map slice**!

If a convolutional layer has $C_{out}$ distinct filters, each filter produces its own 2D slice, resulting in an output tensor of shape:
$$\\mathbf{Y} \\in \\mathbb{R}^{H' \\times W' \\times C_{out}}$$

Let's visualize this 3D filter integration in code:"""))

    cells.append(code("""# Define a 3D filter (3x3x3) that selectively responds to RED vertical edges
# but ignores or suppresses green and blue:
filter_3d = np.zeros((3, 3, 3), dtype=np.float32)
# Channel 0 (Red): Sobel vertical edge detector
filter_3d[:, :, 0] = KERNELS["Sobel Vertical (Vertical Edges $\\\\partial I / \\\\partial x$)"]
# Channel 1 (Green): Zero weight
filter_3d[:, :, 1] = 0.0
# Channel 2 (Blue): Negative suppression
filter_3d[:, :, 2] = -0.5 * KERNELS["Sobel Vertical (Vertical Edges $\\\\partial I / \\\\partial x$)"]

# Apply using TensorFlow
tf_rgb_input = tf.constant((rgb_image / 255.0)[np.newaxis, :, :, :], dtype=tf.float32)
tf_filter_3d = tf.constant(filter_3d[:, :, :, np.newaxis], dtype=tf.float32) # (3, 3, 3, 1)

conv_multi_ch = tf.squeeze(tf.nn.conv2d(tf_rgb_input, tf_filter_3d, strides=[1,1,1,1], padding='SAME')).numpy()

# Also compute the individual per-channel contributions before summation
contrib_R = manual_conv2d(rgb_image[:,:,0]/255.0, filter_3d[:,:,0], padding='same')
contrib_G = manual_conv2d(rgb_image[:,:,1]/255.0, filter_3d[:,:,1], padding='same')
contrib_B = manual_conv2d(rgb_image[:,:,2]/255.0, filter_3d[:,:,2], padding='same')

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
axes[0].imshow(contrib_R, cmap='coolwarm')
axes[0].set_title("Red Channel $\\\\times K_R$", fontsize=11)
axes[0].axis('off')

axes[1].imshow(contrib_G, cmap='coolwarm')
axes[1].set_title("Green Channel $\\\\times K_G$ (All 0s)", fontsize=11)
axes[1].axis('off')

axes[2].imshow(contrib_B, cmap='coolwarm')
axes[2].set_title("Blue Channel $\\\\times K_B$", fontsize=11)
axes[2].axis('off')

axes[3].imshow(conv_multi_ch, cmap='coolwarm')
axes[3].set_title("Summed Output Feature Map\\n$\\sum_c (I_c * K_c)$", fontsize=11, fontweight='bold')
axes[3].axis('off')

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # Section 3: Spatial Mechanics: Patches, Padding, and Pooling
    # =========================================================================
    cells.append(md("""---
# Part 3: Spatial Mechanics — Patches, Padding, and Pooling

## 3.1 How Patches and Receptive Fields are Formed
The fundamental building block of a convolutional layer is the **image patch**. When a $3 \\times 3$ kernel scans an image, at every single pixel position it isolates a localized $3 \\times 3$ sub-matrix called the local receptive field.

Let's extract and visualize a series of consecutive patches across a scanline so you can see exactly what the network observes as the kernel slides:"""))

    cells.append(code("""# Select a region of interest with prominent edges
sample_patch_img = (rgb_image[:64, :64, :] / 255.0).astype(np.float32)
kernel_size = 5

# Extract 5 patches along a horizontal sliding trajectory
patches = []
coords = [(20, 10 + step * 8) for step in range(5)]

for y_center, x_center in coords:
    patch = sample_patch_img[y_center:y_center+kernel_size, x_center:x_center+kernel_size, :]
    patches.append((patch, (y_center, x_center)))

fig = plt.figure(figsize=(15, 6))

# Plot the base image with marked patch bounding boxes
ax_main = fig.add_subplot(1, 6, (1, 2))
ax_main.imshow(sample_patch_img)
colors = ['red', 'orange', 'yellow', 'cyan', 'magenta']
for idx, (y, x) in enumerate(coords):
    rect = plt.Rectangle((x, y), kernel_size, kernel_size, linewidth=2, edgecolor=colors[idx], facecolor='none')
    ax_main.add_patch(rect)
    ax_main.text(x, y-2, f"P{idx+1}", color=colors[idx], fontweight='bold', fontsize=10)
ax_main.set_title("Sliding Receptive Field Windows", fontweight='bold')
ax_main.axis('off')

# Plot the 5 magnified patches
for idx, (patch, (y, x)) in enumerate(patches):
    ax = fig.add_subplot(1, 6, 2 + idx + 1)
    ax.imshow(patch)
    ax.set_title(f"Patch P{idx+1}\\n({kernel_size}x{kernel_size})", color=colors[idx], fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""## 3.2 Padding: Valid vs. Same

When we slide a kernel of size $K$ across an image of width $W$, the kernel cannot extend beyond the borders of the image without falling off.
1. **Valid Padding (`padding='valid'`)**: No padding is applied.
   $$W_{\\text{out}} = \\left\\lfloor \\frac{W - K}{S} \\right\\rfloor + 1$$
   *Consequence*: Every convolution layer progressively shrinks the spatial resolution. Furthermore, corner and border pixels are only sampled once, while central pixels are sampled $K^2$ times!
2. **Same Padding (`padding='same'`)**: The image borders are padded with zeros such that the output has the exact same spatial dimensions as the input (when stride $S=1$):
   $$P = \\frac{K - 1}{2}$$

Let's visualize the zero-padded boundary and inspect the dimension differences:"""))

    cells.append(code("""test_h, test_w = 28, 28
dummy_input = np.ones((test_h, test_w), dtype=np.float32)
k_size = 5

# Compute Valid padding convolution
valid_out = manual_conv2d(dummy_input, np.ones((k_size, k_size)), padding='valid')

# Compute Same padding convolution
same_out = manual_conv2d(dummy_input, np.ones((k_size, k_size)), padding='same')

# Visualize the zero-padded input canvas
pad_amt = (k_size - 1) // 2
zero_padded_visual = np.pad(dummy_input, ((pad_amt, pad_amt), (pad_amt, pad_amt)), mode='constant', constant_values=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

axes[0].imshow(zero_padded_visual, cmap='Blues')
axes[0].set_title(f"Zero-Padded Canvas for 'Same'\\nShape: {zero_padded_visual.shape} (Pad={pad_amt})", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(valid_out, cmap='viridis')
axes[1].set_title(f"'Valid' Padding Output\\nShape: {valid_out.shape} (Shrunk by {k_size-1}px!)", fontweight='bold')
axes[1].axis('off')

axes[2].imshow(same_out, cmap='viridis')
axes[2].set_title(f"'Same' Padding Output\\nShape: {same_out.shape} (Dimensions Preserved)", fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""## 3.3 Pooling: Types and Visual Effects

Convolutions preserve or slightly reduce spatial resolution, but deep networks need to consolidate features, achieve spatial translation invariance, and reduce computational complexity. This is the role of **Pooling Layers**.

### Types of Pooling:
1. **Max Pooling (`tf.keras.layers.MaxPool2D`)**: Slides a window (typically $2 \\times 2$ with stride 2) and keeps the **maximum** value:
   $$y = \\max_{(u, v) \\in \\text{window}} x(u, v)$$
   *Intuition*: Detects whether a feature (e.g. edge or corner) was present *anywhere* in that neighborhood. Discards background noise.
2. **Average Pooling (`tf.keras.layers.AveragePool2D`)**: Computes the arithmetic mean of activations in the window:
   $$y = \\frac{1}{|W|} \\sum_{(u, v) \\in \\text{window}} x(u, v)$$
   *Intuition*: Smooths the feature map, preserving background context.
3. **Global Average Pooling (`tf.keras.layers.GlobalAveragePooling2D`)**: Averages the entire spatial feature map $(H \\times W)$ down to a single scalar per channel, reducing $(H, W, C) \\to (1, C)$. Widely used before classification layers to replace parameter-heavy `Flatten` layers.

Let's compare Max Pooling and Average Pooling side-by-side on an edge-detected image:"""))

    cells.append(code("""# Create an edge feature map to pass through pooling
edge_map = manual_conv2d(perceptual_gray.astype(np.float32), KERNELS["Sobel Vertical (Vertical Edges $\\\\partial I / \\\\partial x$)"], padding='same')
edge_map_tf = tf.constant(edge_map[np.newaxis, :, :, np.newaxis], dtype=tf.float32)

# Apply Max Pooling (pool_size=2, stride=2)
max_pooled = tf.nn.max_pool2d(edge_map_tf, ksize=2, strides=2, padding='VALID').numpy()[0, :, :, 0]

# Apply Average Pooling (pool_size=2, stride=2)
avg_pooled = tf.nn.avg_pool2d(edge_map_tf, ksize=2, strides=2, padding='VALID').numpy()[0, :, :, 0]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].imshow(edge_map, cmap='gray')
axes[0].set_title(f"Original Edge Map\\nShape: {edge_map.shape}", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(max_pooled, cmap='gray')
axes[1].set_title(f"Max Pooling (2x2, stride=2)\\nShape: {max_pooled.shape}\\n(Sharp, Prominent Edge Preserved)", fontsize=12, fontweight='bold')
axes[1].axis('off')

axes[2].imshow(avg_pooled, cmap='gray')
axes[2].set_title(f"Average Pooling (2x2, stride=2)\\nShape: {avg_pooled.shape}\\n(Smoothed, Blurred Edge Response)", fontsize=12, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.show()
"""))

    # =========================================================================
    # Section 4: The Hierarchy of Features: Low vs High-Level & Non-Linearity
    # =========================================================================
    cells.append(md("""---
# Part 4: The Hierarchy of Features — Why Depth & Non-Linearity Matter

Why don't we just use a single, very wide convolutional layer? Why do all modern vision models (AlexNet, VGG, ResNet, ConvNeXt) stack many layers deep?

### 4.1 The Collapse Theorem: The Necessity of Non-Linearity (ReLU)
Consider a deep linear neural network without activation functions:
$$\\mathbf{z}_1 = \\mathbf{W}_1 \\mathbf{x}$$
$$\\mathbf{z}_2 = \\mathbf{W}_2 \\mathbf{z}_1 = \\mathbf{W}_2 (\\mathbf{W}_1 \\mathbf{x})$$
$$\\mathbf{z}_3 = \\mathbf{W}_3 \\mathbf{z}_2 = \\mathbf{W}_3 (\\mathbf{W}_2 \\mathbf{W}_1) \\mathbf{x} = \\mathbf{W}_{\\text{effective}} \\mathbf{x}$$
**No matter how many layers deep you make a linear network, it completely collapses into a single linear regression!** Stacking 50 linear layers provides zero additional representational capacity.

### What does the Rectified Linear Unit (ReLU) do?
$$\\text{ReLU}(z) = \\max(0, z)$$
Visually, ReLU acts as a **feature gating threshold**:
* Positive responses (strong feature alignment) pass through unchanged.
* Negative responses (regions where the pattern is absent or inverted) are strictly clamped to zero.
* This introduces **sparsity**, breaks linearity, and allows deep networks to approximate arbitrary complex decision surfaces.

Let's visually demonstrate the effect of ReLU on a filter response:"""))

    cells.append(code("""raw_conv = manual_conv2d(perceptual_gray.astype(np.float32), KERNELS["Sobel Horizontal (Horizontal Edges $\\\\partial I / \\\\partial y$)"], padding='same')
relu_conv = np.maximum(0, raw_conv)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

im0 = axes[0].imshow(raw_conv, cmap='coolwarm', vmin=-np.max(np.abs(raw_conv)), vmax=np.max(np.abs(raw_conv)))
axes[0].set_title("Pre-Activation Feature Map (Linear $\\\\mathbf{W} * \\\\mathbf{x}$)\\nContains positive & negative values", fontweight='bold')
axes[0].axis('off')
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(relu_conv, cmap='hot')
axes[1].set_title("Post-Activation with ReLU: $\\\\max(0, \\\\mathbf{W} * \\\\mathbf{x}$)\\nNegative responses gated to 0 (sparse!)", fontweight='bold')
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# Sparsity distribution
ax_hist = axes[2]
ax_hist.hist(raw_conv.ravel(), bins=50, color='blue', alpha=0.5, label='Pre-ReLU')
ax_hist.hist(relu_conv.ravel(), bins=50, color='red', alpha=0.5, label='Post-ReLU')
ax_hist.set_title("Activation Distribution Before vs After ReLU", fontweight='bold')
ax_hist.set_xlabel("Activation Magnitude")
ax_hist.set_ylabel("Frequency")
ax_hist.legend()
ax_hist.grid(alpha=0.3)

plt.tight_layout()
plt.show()

sparsity = np.mean(relu_conv == 0) * 100
print(f"Percentage of deactivated (zeroed) neurons due to ReLU: {sparsity:.1f}%")
"""))

    cells.append(md("""## 4.2 Peeking Inside a Deep Neural Network: Visualizing Feature Maps Across Layers

Now let's build a deep Convolutional Neural Network with multiple layers and inspect what features the network learns at different depths:

* **Layer 1 (Early Features)**: High spatial resolution, crisp localized edge detectors, oriented lines, color boundaries.
* **Layer 2 (Intermediate Features)**: Textures, corners, intersections, small contours.
* **Layer 3+ (Deep Features)**: Low spatial resolution, abstract semantic feature maps (complex patterns, object parts).

Let's define a model, run our image through it, and extract the intermediate activation maps at each stage:"""))

    cells.append(code("""# Define a multi-stage CNN architecture
inputs = tf.keras.Input(shape=(256, 256, 3))

# Stage 1: Initial feature extraction
x1 = tf.keras.layers.Conv2D(16, (3, 3), padding='same', activation='relu', name='stage1_conv')(inputs)
p1 = tf.keras.layers.MaxPool2D((2, 2), name='stage1_pool')(x1) # 128x128

# Stage 2: Intermediate feature composition
x2 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='stage2_conv')(p1)
p2 = tf.keras.layers.MaxPool2D((2, 2), name='stage2_pool')(x2) # 64x64

# Stage 3: Deep semantic abstractions
x3 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='stage3_conv')(p2)
p3 = tf.keras.layers.MaxPool2D((2, 2), name='stage3_pool')(x3) # 32x32

# Build feature extractor model that outputs intermediate activations
feature_extractor = tf.keras.Model(
    inputs=inputs,
    outputs=[x1, x2, x3]
)

# Pass our image through the feature extractor
input_batch = (rgb_image / 255.0)[np.newaxis, :, :, :]
activations_s1, activations_s2, activations_s3 = feature_extractor.predict(input_batch)

print(f"Input Shape:            {input_batch.shape}")
print(f"Stage 1 Activations:    {activations_s1.shape} (16 feature channels, 256x256)")
print(f"Stage 2 Activations:    {activations_s2.shape} (32 feature channels, 128x128)")
print(f"Stage 3 Activations:    {activations_s3.shape} (64 feature channels, 64x64)")
"""))

    cells.append(md("""### 4.3 Visualizing the Feature Hierarchy: Early vs. Intermediate vs. Deep Layers

Let's display sample channels from each layer side-by-side to witness the transformation from literal visual shapes to abstract representations:"""))

    cells.append(code("""def plot_layer_channels(activations, stage_name, num_channels=6):
    fig, axes = plt.subplots(1, num_channels, figsize=(18, 3))
    for i in range(num_channels):
        ax = axes[i]
        channel_map = activations[0, :, :, i]
        ax.imshow(channel_map, cmap='viridis')
        ax.set_title(f"Channel {i+1}", fontsize=10)
        ax.axis('off')
    fig.suptitle(f"{stage_name} Activations (Spatial Dim: {activations.shape[1]}x{activations.shape[2]})", fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

# Visualize 6 channels from each stage
print(">>> STAGE 1: Low-Level Primitives (Edges, Gradients, Textures)")
plot_layer_channels(activations_s1, "Stage 1 (Initial Layer)")

print(">>> STAGE 2: Intermediate Visual Motifs (Corners, Intersections)")
plot_layer_channels(activations_s2, "Stage 2 (Middle Layer)")

print(">>> STAGE 3: High-Level Abstract Semantic Features (Condensed Representation)")
plot_layer_channels(activations_s3, "Stage 3 (Deep Layer)")
"""))

    # =========================================================================
    # Section 5: Interactive Exercises & Student TODOs
    # =========================================================================
    cells.append(md("""---
# Part 5: Interactive Student Exercises & TODO Challenges

Now it is your turn to experiment hands-on! Complete the following four targeted exercises. Follow the instructions and replace the `# '''TODO: ...'''` blocks with your own code.

---

### Exercise 1: Custom Diagonal Edge Detector
In computer vision, horizontal and vertical edge detectors (Sobel) leave out diagonal contours ($45^\\circ$ and $135^\\circ$).
A $45^\\circ$ diagonal edge filter can be defined with positive weights along the top-right and negative weights along the bottom-left:
$$K_{45} = \\begin{bmatrix} 0 & 1 & 2 \\\\ -1 & 0 & 1 \\\\ -2 & -1 & 0 \\end{bmatrix}$$

**Your Task**: Fill in the `diagonal_kernel` matrix below, run the convolution, and observe the highlighted diagonal edges!"""))

    cells.append(code("""# '''TODO: Define the 3x3 diagonal edge filter matrix'''
# Fill in the 3x3 values following the formula above
diagonal_kernel = np.array([
    # [TODO: Complete row 0]
    [ 0,  1,  2],
    # [TODO: Complete row 1]
    [-1,  0,  1],
    # [TODO: Complete row 2]
    [-2, -1,  0]
], dtype=np.float32)

# Compute convolution with diagonal kernel
diagonal_response = manual_conv2d(perceptual_gray.astype(np.float32), diagonal_kernel, padding='same')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(perceptual_gray, cmap='gray')
axes[0].set_title("Input Grayscale Image", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(np.abs(diagonal_response), cmap='inferno')
axes[1].set_title("45-Degree Diagonal Edge Detection", fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""---
### Exercise 2: Spatial Feature Map Dimension Calculator

Recall the general formula for the output spatial dimension of a convolution given input dimension $W$, kernel size $K$, padding $P$, and stride $S$:
$$W_{\\text{out}} = \\left\\lfloor \\frac{W - K + 2P}{S} \\right\\rfloor + 1$$

**Your Task**: Implement the mathematical formula inside `compute_conv_output_dim` and verify it passes the automated test cases."""))

    cells.append(code("""def compute_conv_output_dim(input_dim, kernel_size, padding_amount, stride):
    \"\"\"
    Computes output spatial dimension of a 1D/2D convolution layer.
    
    Args:
        input_dim: input spatial dimension (H or W)
        kernel_size: filter size K
        padding_amount: number of zero-padded pixels P on each side
        stride: step size S
    Returns:
        output_dim: integer output spatial size
    \"\"\"
    # '''TODO: Implement the spatial dimension formula'''
    output_dim = (input_dim - kernel_size + 2 * padding_amount) // stride + 1
    return output_dim

# Test cases:
# Test 1: Standard ResNet Layer 1 (Input 224, Kernel 7, Padding 3, Stride 2)
test1 = compute_conv_output_dim(224, 7, 3, 2)
assert test1 == 112, f"Test 1 Failed: Expected 112, got {test1}"

# Test 2: Valid padding convolution (Input 28, Kernel 3, Padding 0, Stride 1)
test2 = compute_conv_output_dim(28, 3, 0, 1)
assert test2 == 26, f"Test 2 Failed: Expected 26, got {test2}"

# Test 3: Stride 2 without padding (Input 32, Kernel 4, Padding 0, Stride 2)
test3 = compute_conv_output_dim(32, 4, 0, 2)
assert test3 == 15, f"Test 3 Failed: Expected 15, got {test3}"

print("SUCCESS: All spatial dimension calculation tests passed!")
"""))

    cells.append(md("""---
### Exercise 3: Channel Sensitivity & Attention Gating Filter

Suppose you are designing an autonomous vehicle vision system that needs to detect red traffic signals while ignoring blue sky and green background foliage.
We can create a 3-channel weighting vector:
$$\\mathbf{w} = [w_R, w_G, w_B]$$
combined with a $\\text{ReLU}$ non-linearity to isolate warm red hues.

**Your Task**: Set `weights = [w_R, w_G, w_B]` below such that Red is strongly prioritized with positive weight ($> 0$), while Green and Blue are suppressed with negative weights ($< 0$)."""))

    cells.append(code("""# '''TODO: Define channel weights [w_R, w_G, w_B]'''
# Assign a positive weight to Red, and negative weights to Green and Blue
weights = np.array([2.0, -1.0, -1.0], dtype=np.float32)

# Compute weighted channel combination
red_attention_map = (
    weights[0] * (rgb_image[:,:,0] / 255.0) +
    weights[1] * (rgb_image[:,:,1] / 255.0) +
    weights[2] * (rgb_image[:,:,2] / 255.0)
)
# Apply ReLU to suppress negative non-red regions
red_attention_map = np.maximum(0, red_attention_map)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(rgb_image)
axes[0].set_title("Original RGB Input", fontweight='bold')
axes[0].axis('off')

axes[1].imshow(red_attention_map, cmap='hot')
axes[1].set_title("Red-Sensitive Attention Filter (Post-ReLU)", fontweight='bold')
axes[1].axis('off')

plt.tight_layout()
plt.show()
"""))

    cells.append(md("""---
### Exercise 4: Manual Max Pooling Step

In Section 3.3, we used TensorFlow's `MaxPool2D`. Now let's implement the core selection logic of a $2 \\times 2$ Max Pooling operation on a single patch:
$$y = \\max(x_{0,0}, x_{0,1}, x_{1,0}, x_{1,1})$$

**Your Task**: Complete the `manual_max_pool_2x2` function below to extract the maximum element from any given $2 \\times 2$ spatial window."""))

    cells.append(code("""def manual_max_pool_2x2(patch_2x2):
    \"\"\"
    Computes max pooling on a single 2x2 spatial patch.
    
    Args:
        patch_2x2: 2D numpy array of shape (2, 2)
    Returns:
        max_val: scalar maximum value
    \"\"\"
    # '''TODO: Return the maximum scalar value in the 2x2 patch'''
    max_val = np.max(patch_2x2)
    return max_val

# Test case with a sample 2x2 patch:
sample_patch = np.array([
    [1.5, 4.2],
    [9.8, 3.1]
], dtype=np.float32)

result = manual_max_pool_2x2(sample_patch)
assert result == 9.8, f"Expected 9.8, got {result}"

print("SUCCESS: 2x2 Max Pooling logic verified!")
"""))

    # =========================================================================
    # Conclusion & Summary
    # =========================================================================
    cells.append(md("""---
## Summary and Key Takeaways

Congratulations on completing this deep-dive tutorial! Here is a recap of the key concepts you have explored:

1. **Images as Numerical Tensors**: An image is an array of shape $(H, W, C)$. Channels represent independent physical or learned measurements (1 for grayscale, 2 for vector fields, 3 for RGB, and $N$ for neural feature maps).
2. **Preprocessing**: Normalization ($[0, 1]$) and standardization ($\mu=0, \sigma=1$) stabilize gradient flow and prevent activation saturation.
3. **Convolutions in Action**: A 2D convolution slides a small matrix of weights across local patches, performing element-wise multiplication and summation.
4. **Multi-Channel Convolutions**: A filter operating on a $C_{\\text{in}}$-channel input has depth $C_{\\text{in}}$ and aggregates across all channels simultaneously into a single 2D feature map slice.
5. **Spatial Transformations**:
   - `padding='same'` preserves spatial resolution, while `padding='valid'` shrinks boundaries.
   - Max Pooling preserves salient features and achieves spatial invariance, while Average Pooling smooths.
6. **Feature Hierarchy & Non-Linearity**:
   - Without non-linearities (ReLU), deep networks collapse into a single linear regression.
   - Early layers detect low-level primitives (edges, gradients), while deeper layers compose them into rich, invariant semantic abstractions.

To continue exploring, check out the next tutorials in the repository:
* [**Tutorial 02: MNIST Digit Classification with CNNs**](https://github.com/Armstrong66/computer-vision-tutorials)
* [**Tutorial 03: Latent Space Exploration & Generative Models**](https://github.com/Armstrong66/computer-vision-tutorials)
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {
                "name": "TF_Visual_Deep_Dive_How_NNs_See.ipynb",
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook

if __name__ == "__main__":
    nb = create_notebook()
    
    # Save to tutorials and workspace root
    target_paths = [
        os.path.join("tutorials", "01_how_neural_networks_see", "TF_Visual_Deep_Dive_How_NNs_See.ipynb"),
        "How_Neural_Networks_See_Images.ipynb"
    ]
    
    for path in target_paths:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Successfully generated notebook at: {path}")
