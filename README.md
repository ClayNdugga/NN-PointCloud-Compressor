# NN-PointCloud-Compressor

## Introduction
Compression is crucial for efficient storage and transmission of media. Current compression methods, such as JPEG for images, and MPEG for videos rely on linear transform coding. This works well for 2D media where spatial relationships are consistent and predictable. However, as we move towards more complex mediums that incorporate the third dimension, like point clouds, the limitations of linear transform coding become apparent.

Non-linear transform coding does not assume any fixed linear relationship among data points. It allows for a more flexible and adaptive representation that can capture the complexity of point clouds. This approach can lead to more accurate reconstructions and more efficient compression.

## Project Goal

This project follows the development of a Non-Linear Transform Coder for lossy data compression, leveraging Neural Networks to enhance point cloud data compression efficiency. The core objective is to devise a neural network architecture (Non-linear Transform) capable of encoding the complex spatial information inherent in point cloud data. Due to the high computational demands of processing point clouds, this endeavor necessitated an in-depth exploration of the interplay between hardware capabilities and software optimization within the TensorFlow framework. Databricks was used for compute to access NVIDIA GPU clusters for training the model. 

## Background

The following is diagram of a Non-linear Transform coder.

![alt text](https://github.com/ClayNdugga/NN-PointCloud-Compressor/blob/main/assets/NTC.png?raw=true)
