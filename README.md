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

Point cloud data is inherently unordered and unstructured, unlike pixel-based images. Some approaches try to impose structure by voxelizing the point cloud, but this method becomes computationally expensive at high resolutions. FoldingNet bypasses this issue by processing point clouds directly, which eliminates the need for voxelization but introduces its own challenges.

One key challenge is achieving **permutation invariance**—ensuring that different permutations of the same point cloud produce the same latent vector. FoldingNet accomplishes this by applying **shared weights** across multi-layer perceptron (MLP) layers, allowing independent point processing while maintaining consistent representations.

However, processing points independently risks losing local geometric relationships. FoldingNet addresses this with two mechanisms:

1. Concatenating a **local covariance matrix** to each point, preserving local geometric information.
2. Using **K-Nearest Neighbor** (K-NN) graph layers to aggregate local features effectively.

By stacking graph layers hierarchically, the network progressively learns abstract features, while the final encoding produces a 1x512 latent vector capturing the most salient features of the original point cloud.

#### FN Decoder

The FoldingNet decoder is designed to reconstruct the original point cloud by "folding" a 2D grid back into the original 3D shape from information contained in the latent vector.

The universal approximation theorem suggests that a sufficiently deep MLP can approximate any non-linear function. In this case, the MLP is used to approximate a function that maps information from 2D -> 3D, allowing the grid to be transformed into the target 3D point cloud.

By concatenating the latent vector on a 2D grid before passing it through the MLP, the decoder can reconstruct the original sample.


<p align="center">
  <img width="460" height="300" src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/example_fold.gif?raw=true">
</p>
<p align="center">
  <i>FoldingNet reconstructing a 3D couch from an initial 2D grid and couch latent vector</i>
</p>

#### PCN Decoder 

The Point Completion Network (PCN) decoder improves on FoldingNet by addressing some of its limitations. Instead of relying on a single grid, PCN employs multiple grids to handle complex point cloud structures more effectively. This approach allows for better reconstructions, especially for intricate shapes.

The PCN decoder’s loss function balances coarse and fine reconstructions:
```math
L_{PCN} = d_{CD}(x_i, \hat{z}_i) + \alpha \cdot d_{CD}(x_i, \hat{x}_i)

```

During training, the parameter $\alpha$ starts small, allowing the network to prioritize the coarse structure initially. As training progresses, $\alpha$ increases, enabling the network to focus on refining the final reconstruction quality.

<figure align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/pcnarch.png?raw=true" alt="FoldingNet Architecture"/>
  <figcaption style="text-align: center;">
    <i>PCN Decoder Architecture</i>
    <br>
    <cite>Source: https://arxiv.org/pdf/1808.00671</cite>
  </figcaption>
</figure>



## Results
The final NTC was implemented as follows:

![alt text](https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/final_design.png?raw=true)


The model was trained on $N = 13,000$ point clouds, where each point cloud $x_i$ consists of $n = 2048$ points in $\mathbb{R}^3$. The points were transformed into a latent representation of size $m = 1024$, before being quantized for transmission or storage.

The loss function guiding this process is given by:

```math
L_{NTC} =  \frac{1}{N} \sum_{i=1}^{N} \left(  d_{CD}(x_i, \hat{z}_i) + \alpha \cdot d_{CD}(x_i, \hat{x}_i) + \lambda \sum_{j=1}^{m} -\log\big(q(\hat{y}_{ij})\big) \right)
```
Here:
- ${x}_i$ refers to the original pointcloud
- $\hat{x}_i$ refers to the fine reconstruction
- $\hat{z}_i$ represents a coarse reconstruction of the input
- $d_{CD}$ is the Chamfer distance (a distance measure of similiarity)
- The term involving $q(\hat{y}_{ij})$ calculates the code rate by modeling each component of the quantized latent vector $\hat{y}_i$ using a Gaussian distribution.


To compute the code rate, the quantized latent representation $\hat{y}_i$ is assumed to follow a Gaussian distribution, where each component $\hat{y}_{ij}$ has a mean $\mu_j$ and variance $\sigma_j^2$. The rate is calculated by summing the negative log-likelihood of the probability density function (pdf) of the Gaussian distribution over all components:

```math
-\log q(\hat{y}_{ij}) \quad \text{where} \quad q(\hat{y}_{ij}) \sim \mathcal{N}(\mu_j, \sigma_j^2)
```

The parameters $\mu_j$ and $\sigma_j^2$ are updated every 5 epochs to ensure that the model’s estimate of the data distribution is accurate. This regular update mechanism helps maintain a high degree of compression efficiency by continually refining the latent representation to match the actual distribution of the training data.



The NTC is tasked with balancing the trade-off between compression rate (measured in bits) and reconstruction quality. This balance is controlled by the regularization parameter $\lambda$ in the loss function. A smaller $\lambda$ prioritizes high reconstruction accuracy at the expense of larger code rates, while a larger $\lambda$ leads to smaller code rates but with lower reconstruction quality.

As depicted in the Rate-Distortion (RD) curve below, as 
$\lambda$ increases, we observe an increase in code efficiency at the cost of some degradation in point cloud reconstruction:

<p align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/rd_curve.png?raw=true">
</p>
<p align="center">
  <i> Rate-Distortion Curve</i>
</p>

The following visualization illustrates how different values of $\lambda$ affect the reconstruction of a sample point cloud. As $\lambda$ increases, the visual fidelity of the reconstruction decreases, but the code rate is significantly reduced:

<p align="center">
  <img src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/final_reconstruction.png?raw=true">
</p>
<p align="center">
  <i> Reconstruction performance</i>
</p>

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
