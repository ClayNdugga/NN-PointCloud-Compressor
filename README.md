# Neural Network PointCloud Compression

Note this project is ongoing: Sept 2023 - April 2024

# Introduction
Compression is crucial for efficient storage and transmission of media. Current compression methods, such as JPEG for images, and MPEG for videos rely on **Linear Transform Coding**. This works well for 2D media where spatial relationships are consistent and predictable. However, as we move towards more complex mediums that incorporate the third dimension, like point clouds, the limitations of linear transform coding become apparent.

**Non-Linear Transform Coding** does not assume any fixed linear relationship among data points. It allows for a more flexible and adaptive representation that can capture the complexity of point clouds. This approach can lead to more accurate reconstructions and more efficient compression.

## Project Goal

This project follows the development of a **Non-Linear Transform Coder** for lossy data compression to enhance point cloud compression efficiency. By capturing underlying patterns in data distributions, neural networks can eliminate redundancy and create smaller datasets that contain the most salient features from the original data. The project objective is to devise a neural network architecture (Non-linear Transform) capable of encoding the complex spatial information inherent in point cloud data. 


## Requirements
* Python 3.10
* CUDA V12.2.140

Run
```sh
pip install -r requirements.txt
```

## Download Datasets
Two datasets were used in the project were
* ShapenetCore.v2
* ModelNet40 

Datasets can be downloaded from this [repository](https://github.com/antao97/PointCloudDatasets)


## Training Model

Run main.py with the training arguments

```shell
python -m main --dataset "modelnet40" --batch_size 32 --latent_dim 512 --model_name "ModelxRunx"
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

### Non-Linear Transform Coder (NTC)

The following is diagram of the NTC implemented in the project:

![alt text](https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/final_design.png?raw=true)

A NTC is a system designed to efficiently compress data. The system is composed of three primary components: the Forward and Inverse **Transforms**, the **Quantizer**, and the **Encoder-Decoder** pair.

The **Transforms** map the high dimensional input data into a compressed low dimensional latent space. These transforms are non-linear, facilitating a more compact representation than linear methods typically allow.

The **Quantizer** discretizes the continuous values from the latent space creating a finite set, which is essential for digital storage and transmission. This step is lossy and introduces reconstruction errors, but ultimately allows the data to be represented with fewer bits.

The **Lossless Encoder** creates the compressed code representation whose rate, measured in bits, is size of the resultant encoded object. This code is what will be stored or transmitted and is the point cloud in its most compressed representation. 

While training the NTC, a balance must be struck between two inherently contradictory goals: minimizing the <span style="color:#7030A0">reconstruction distortion</span>, and minimizing the <span style="color:#5da4d7">code rate</span> (size in bits). To accommodate for this, the loss function is augmented with a training parameter λ that controls the trade-off between rate and distortion.



## Results 


<p align="center">
  <img width="460" height="300" src="https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/final_reconstruction.png?raw=true">
</p>
<p align="center">
  <i>FoldingNet reconstructing a 3D couch from an inital 2D grid and couch latent vector</i>
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
* [Network Architechtures](https://github.com/lynetcha/completion3d)
* [Pointcloud Rendering](https://github.com/zekunhao1995/PointFlowRenderer)

