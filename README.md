# Neural Network PointCloud Compression


## Introduction

Efficient compression is essential for the storage and transmission of data. While traditional methods like JPEG for images and MPEG for videos are effective for 2D media, they struggle with 3D data formats like point clouds. This project aims to enhance point cloud compression using Non-Linear Transform Coding (NTC), which can better handle the complexities of 3D data.

Our goal is to develop a deep learning-based Neural Network PointCloud Compressor capable of achieving lossy compression while maintaining high-quality reconstructions.cloud data.

## Project Objectives
The project focuses on designing a Non-Linear Transform Coder (NTC) for lossy point cloud compression. The architecture leverages neural networks to capture spatial patterns in point cloud data and eliminates redundancy, leading to efficient compression.

Key Aspects:

* Developing a neural network architecture that can encode 3D point cloud data.
* Balancing compression size (code rate) and reconstruction quality.

## Installation

To set up the project environment, make sure you have **Python 3.10** and **CUDA V12.2.140** installed. Then, run the following command to install dependencies:

```sh
pip install -r requirements.txt
```

## Dataset Setup

Two datasets were used in the project were

- ShapenetCore.v2
- ModelNet40

Datasets can be downloaded from this [repository](https://github.com/antao97/PointCloudDatasets)

## Training Model

Run main.py with the training arguments

```shell
python -m main --dataset "modelnet40" --batch_size 32 --lmbda 0.000001 --latent_dim 512 --model_name "ModelxRunx"
```

Access the real time results and pointcloud reconstruction visualization via Tensorboard

```shell
tensorboard --logdir="<path to logs/fit>"
```

## Folder Structure

```python
    tensorflow_project/
    │
    ├── data/                   # Data handling
    │ ├── ModelNet40            # ModelNet40 Dataset Folder
    │ ├── ShapeNet              # ShapeNet Dataset Folder
    │ └── data_loader.py        # Script to load data for training/evaluation
    │
    ├── models/                 # Model architectures
    │ ├── encoder.py            # Model Encoder definition
    │ ├── decoder.py            # Model Decoder definition
    │ └── model.py              # Complete AutoEncoder
    │
    ├── training/               # Training
    │ └── train_utils.py        # Training utilities (callbacks, etc.)
    │
    │
    ├── config/                 # Configuration files
    │ ├── configx.json          # Model Training hyperparameters
    │ └── default.yaml          # Default Model Training hyperparameters
    │
    │
    ├── notebooks/              # Jupyter notebooks
    │ ├── PCN.ipynb             # Notebook for testing
    │ └── NTC.ipynb             # Simple NTC Outline Demo
    │
    ├── README.md               # Project overview and setup instructions
    └── requirements.txt        # Project dependencies for pip
```

# Background

### Non-Linear Transform Coder (NTC)s

The following is a high level overview of a NTC:

![alt text](https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/NTC3.png?raw=true)

Non-Linear Transform Coding (NTC) is a method designed for efficient data compression. It consists of:

* **Non-Linear Transforms**: Maps high-dimensional data into a compressed low-dimensional latent space, capturing the essential features more compactly than linear methods.
* **Quantizer**: Discretizes the latent space, introducing a small amount of reconstruction error but enabling efficient digital storage and transmission.
* **Lossless Encoder-Decoder**: Encodes the quantized data into a compressed format for storage or transmission.

In training the NTC, a trade-off must be made between <span style="color:#fce303">reconstruction quality</span> and <span style="color:#5da4d7">compression size</span> (size in bits). This trade-off is controlled via a regularization parameter $λ$ in the loss function.

### Network Architectures 

The literature on point cloud classification and segmentation is extensive. While distinct from compression, these studies offer valuable insights on the training methodologies and network architectures that can serve as the foundation for the development of an effective NTC transform architecture. Most notable is FoldingNet (FN), an autoencoder for point clouds that aligns closely with the goal of compression. It consists of two main components: the encoder, and decoder.

<figure align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/fnarch.png?raw=true" alt="FoldingNet Architecture"/>
  <figcaption style="text-align: center;">
    <i>FoldingNet Architecture</i>
    <br>
    <cite>Source: https://arxiv.org/pdf/1712.07262.pdf</cite>
  </figcaption>
</figure>

#### FN Encoder

In contrast to the ordered and structured nature of pixel-based images, point cloud data is unstructured and inherently unordered. Consequently, some methods attempt to voxelize the point cloud to impose a structure suitable for traditional convolution operations. However, such voxelization becomes computational unfeasible at high resolutions necessitating a better approach.

The FoldingNet encoder can process point cloud’s directly, mitigating the computational overhead introduced by voxelization. Nonetheless, this approach introduces its own set of challenges. A key network requirement when processing directly is permutation invariance, that is, if two identical point clouds are evaluated by the model, one with a different order, they should produce an identical latent vector. To achieve this, FoldingNet employs shared weights across the MLP layers and processes points independently.

This approach, however, raises the challenge of capturing the local geometry accurately since the relational information between points is diminished when they are processed separately. FoldingNet addresses this in two ways: firstly, by concatenating a local covariance matrix to accompany each point so local geometric information in not lost when processing points individually in MLP blocks, and secondly, by employing K-Nearest Neighbour (K-NN) graph layers to effectively aggregate local features.

By hierarchically stacking graph layers, the "resolution" of the representation is progressively reduced allowing each subsequent layer to learn larger more abstract features.

The output of the encoding layer is a 1x512 latent vector that contains the most salient features from the original object.

#### FN Decoder

The FoldingNet decoder is designed to reconstruct the original point cloud by "folding" a 2D grid back into the original 3D shape from information contained in the latent vector.

The universal approximation theorem suggests that a sufficiently deep MLP can approximate any non-linear function. In this case, the MLP is used to approximate a function that maps information from 2D -> 3D, allowing the grid to be transformed into the target 3D point cloud.

By concatenating the latent vector on a 2D grid before passing it through the MLP, the decoder can reconstruct the orignal sample.


<p align="center">
  <img width="460" height="300" src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/example_fold.gif?raw=true">
</p>
<p align="center">
  <i>FoldingNet reconstructing a 3D couch from an inital 2D grid and couch latent vector</i>
</p>

#### PCN Decoder 

<figure align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/pcnarch.png?raw=true" alt="FoldingNet Architecture"/>
  <figcaption style="text-align: center;">
    <i>PCN Decoder Architecture</i>
    <br>
    <cite>Source: https://arxiv.org/pdf/1808.00671</cite>
  </figcaption>
</figure>

(chat gpt work here. make the bullet points below into a small coheernet paragraph)
*Multiple grids addresses the pitfalls of FoldingNet
*𝐿𝑜𝑠𝑠=𝑑_𝐶𝐷 (𝑥_𝑖, 𝑧 ̂_𝑖 )+𝛼∙𝑑_𝐶𝐷 (𝑥_𝑖, 𝑥 ̂_𝑖 )
*While training 𝛼 starts small then increases to prioritize final reconstruction



## Results
The final NTC was implemented as follows:

![alt text](https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/final_design.png?raw=true)










The model was trained on $N$ point clouds, $\{x_i\}_{i=1}^N$, where each point cloud $x_i$ consists of $n$ points in $\mathbb{R}^3$, i.e., $x_i = \{\vec{p}_1, \vec{p}_2, \ldots, \vec{p}_n\}$ with $\vec{p}_j \in \mathbb{R}^3$ for $j = 1, \ldots, n$ and $i = 1, \ldots, N$.  The loss function is given as


```math
L_{NTC} =  \frac{1}{N} \sum_{i=1}^{N} \left(  d_{CD}(x_i, \hat{z}_i) + \alpha \cdot d_{CD}(x_i, \hat{x}_i) + \lambda \sum_{j=1}^{m} -\log\big(q((\hat{y}_i)_j)\big) \right)
```


$y_i \in \mathbb{R}^m$ and $\hat{y}_i \in \mathbb{R}^m$ denote the latent vector and quantized latent vector respectively for the $i$-th point cloud. $\hat{z}_i$ is the coarse reconstruction, and $\hat{x}_i$ is the fine reconstruction. $\hat{y}_i = g_a(x_i) + u$, $\hat{z}_i = g_{sc}(g_a(x_i) + u)$, $\hat{x}_i = g_{sf}(g_a(x_i) + u)$. $N = 13000, n = 2048, m = 1024$.

$u \in \mathbb{R}^m$ is a vector where each component $u_i$ is independently drawn from a uniform distribution, specifically, for each $i$ in $1, \ldots, m$, we have $u_i \sim \text{Uniform}\left(-\frac{1}{2}, \frac{1}{2}\right)$. Uniform noise is used to emulate the distortion introdced by quantization while remainign differentiable during backpropagation. 

The $q$ function denotes the pdf of a normal distribution used to estimate the <span style="color:#5da4d7">code rate</span> of the quantized latent vector $\hat{y}_i$. Each component $(\hat{y}_i)_j$ of the vector is assumed to be normally distributed with mean $\mu_j$ and variance $\sigma_j^2$. $\mu = [\mu_1, \mu_2, \ldots, \mu_m]$ and $\sigma^2 = [\sigma_1^2, \sigma_2^2, \ldots, \sigma_m^2]$ are determined by calculating the mean and variance of the quantized latent vector every 5 training epochs.


The following plots shows the reconstruction performance at distinct values of $\lambda$. Each reconstruction places a different importance on reconstruction quality and compression size.

<p align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/rd_curve.png?raw=true">
</p>
<p align="center">
  <i> Rate-Distortion Curve</i>
</p>

<p align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/final_reconstruction.png?raw=true">
</p>
<p align="center">
  <i> Reconstruction performance</i>
</p>


As expected, reconstructions with low distortion come at the cost of high code rates (size in bits), while high distortion reconstructions can be represented by a smaller code rate.
<!-- 

### FoldingNet


The literature on point cloud classification and segmentation is extensive. While distinct from compression, these studies offer valuable insights on the training methodologies and network architectures that can serve as the foundation for the development of an effective NTC transform architecture. Most notable is FoldingNet, an autoencoder for point clouds that aligns closely with the goal of compression. It consists of two main components: the encoder, and decoder.

<figure align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/FoldingNet.jpg?raw=true" alt="FoldingNet Architecture"/>
  <figcaption style="text-align: center;">
    <i>FoldingNet Architecture</i>
    <br>
    <cite>Source: https://arxiv.org/pdf/1712.07262.pdf</cite>
  </figcaption>
</figure>


#### Encoder

In contrast to the ordered and structured nature of pixel-based images, point cloud data is unstructured and inherently unordered. Consequently, some methods attempt to voxelize the point cloud to impose a structure suitable for traditional convolution operations. However, such voxelization becomes computational unfeasible at high resolutions necessitating a better approach.

The FoldingNet encoder can process point cloud’s directly, mitigating the computational overhead introduced by voxelization. Nonetheless, this approach introduces its own set of challenges. A key network requirement when processing directly is permutation invariance, that is, if two identical point clouds are evaluated by the model, one with a different order, they should produce an identical latent vector. To achieve this, FoldingNet employs shared weights across the MLP layers and processes points independently.

This approach, however, raises the challenge of capturing the local geometry accurately since the relational information between points is diminished when they are processed separately. FoldingNet addresses this in two ways: firstly, by concatenating a local covariance matrix to accompany each point so local geometric information in not lost when processing points individually in MLP blocks, and secondly, by employing K-Nearest Neighbour (K-NN) graph layers to effectively aggregate local features.

By hierarchically stacking graph layers, the "resolution" of the representation is progressively reduced allowing each subsequent layer to learn larger more abstract features.

The output of the encoding layer is a 1x512 latent vector that contains the most salient abstract features from the original object.

#### Decoder

The FoldingNet decoder is designed to reconstruct the original point cloud by "folding" a 2D grid back into the original 3D shape from information contained in the latent vector.

The universal approximation theorem suggests that a sufficiently deep MLP can approximate any non-linear function. In this case, the MLP is used to approximate a function that maps information from 2D -> 3D, allowing the grid to be transformed into the target 3D point cloud.

By concatenating the latent vector on a 2D grid before passing it through the MLP, the decoder can reconstruct the orignal sample.

<p align="center">
  <img width="460" height="300" src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/example_fold.gif?raw=true">
</p>
<p align="center">
  <i>FoldingNet reconstructing a 3D couch from an inital 2D grid and couch latent vector</i>
</p>


### Integration

Encoder and Decoder network architecture from FoldingNet will serve as a starting point for the forward and inverse transform respectively in the NTC. Following their successful implementation, network modification and hyperparameter tuning will ensue to explore what changes facilitate effective compression. -->

### Reference Repositories

- [Network Architechtures](https://github.com/lynetcha/completion3d)
- [Pointcloud Rendering](https://github.com/zekunhao1995/PointFlowRenderer)
