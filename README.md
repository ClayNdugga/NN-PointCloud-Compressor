# NN-PointCloud-Compressor

## Introduction
Compression is crucial for efficient storage and transmission of media. Current compression methods, such as JPEG for images, and MPEG for videos rely on linear transform coding. This works well for 2D media where spatial relationships are consistent and predictable. However, as we move towards more complex mediums that incorporate the third dimension, like point clouds, the limitations of linear transform coding become apparent.

Non-linear transform coding does not assume any fixed linear relationship among data points. It allows for a more flexible and adaptive representation that can capture the complexity of point clouds. This approach can lead to more accurate reconstructions and more efficient compression.

## Project Goal

This project follows the development of a Non-Linear Transform Coder for lossy data compression, leveraging Neural Networks to enhance point cloud data compression efficiency. The core objective is to devise a neural network architecture (Non-linear Transform) capable of encoding the complex spatial information inherent in point cloud data. Due to the high computational demands of processing point clouds, this endeavor necessitated an in-depth exploration of the interplay between hardware capabilities and software optimization within the TensorFlow framework. Databricks was used for compute to access NVIDIA GPU clusters for training the model. 

## Background

The following is diagram of a Non-linear Transform coder.

![alt text](https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/NTC.png?raw=true)

Starting on the left we have our PointCloud to be compressed which is then fed through a neural network. This network is the forward transform, serving as the non-linear function that maps our input data to a latent compressed representation. 

At the heart of the non-linear transform coder is the N level quantizer. This component discretizes the continuous output of the forward transform into N buckets, destroying information in the processes, but enabling the data to be represented with fewer bits which is the main goal. 

Next the data is passed through a lossless encoder-decoder pair, which seems redundant, but it allows us to obtain a code that can be used in the loss function when training the network.

Following this process, the inverse transform attempts to reconstruct the original PointCloud using the quantized values.

The loss function is a critical component of achieving effective results. It is defined as the sum of the Reconstructed Distortion and λ multiplied by the Compressed Code rate. The Reconstructed Distortion is calculated using Chamfer Distance, which measures the average closest point distance between the original and reconstructed point clouds. The λ parameter helps to balance the importance between the fidelity of the reconstruction (low distortion) and the efficiency of the compression (low code rate). This ensure that the reconstructed point clouds are as close as possible to the original while maintaining a compact code representation.
