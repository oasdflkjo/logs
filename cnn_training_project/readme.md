# Log Number Detection from Point Cloud

## Overview
This project aims to train a machine learning model to detect log numbers from point cloud data. We utilize a high-performance GPU and modern deep learning frameworks to efficiently process and analyze the data.

## Setup
### Requirements
- **GPU**: AMD RX 6800
- **Python Version**: 3.11
- **Deep Learning Framework**: PyTorch (Windows Backend)

## Data
- The dataset consists of **1000 captures**, each containing:
  - **Metadata**: `capture_YYYYMMDD_HHMMSS_mmm_metadata.json`
  - **Point Cloud Data**: `capture_YYYYMMDD_HHMMSS_mmm_pointcloud.npy`
  - **Rendered Image**: `capture_YYYYMMDD_HHMMSS_mmm_render.png`
  
- Files are stored in `C:\output` and each capture has a **unique timestamp**.

## Next Steps
- Decide on the **model architecture** for log number detection from point cloud data.
- Preprocess the dataset for optimal performance.
- Implement training and evaluation pipelines.

## Possible Model Choices
1. **PointNet** - A lightweight neural network designed for point cloud classification and segmentation.
2. **PointNet++** - An improved version of PointNet that captures local context hierarchically.
3. **DGCNN (Dynamic Graph CNN)** - A model leveraging graph-based representation learning.
4. **VoxelNet** - Converts point clouds into voxels and applies 3D convolutions.
5. **Custom Transformer-based Model** - Leveraging attention mechanisms for structured point cloud understanding.

## Contribution & Feedback
If you have any suggestions for the model choice or improvements, feel free to contribute to this project!

