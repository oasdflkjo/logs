# Tree Log Detection Project

This project aims to develop a machine learning model for detecting and counting tree logs using simulated LiDAR point cloud data.

## Project Checklist
- [x] Initial project setup
- [x] Basic Blender scene generation
- [x] Single log type implementation
- [x] Physics simulation tuning
- [x] LiDAR sensor simulation
- [x] Dataset generation pipeline
- [x] Initial dataset creation (10,000 scenes)
- [x] Neural network architecture design
- [x] Model training pipeline
- [x] Model evaluation and testing

## Project Phases

### Phase 1: Data Generation using Blender
- Automated scene generation using Blender's Python API
- Random placement of tree log objects in the scene
- Physics simulation for realistic log placement

### Phase 2: LiDAR Data Generation
- Simulate LiDAR sensor to generate point cloud data
- Generate large-scale dataset (target: 10,000 scenes)
- Dataset will contain:
  - Point cloud data for each scene
  - Ground truth log count per scene

### Phase 3: Machine Learning Model
- Develop neural network architecture for log detection
- Train model using generated dataset
- Goal: Accurate prediction of log count from point cloud data
