import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=1024, device='cpu'):
        self.data_dir = data_dir
        self.device = device
        self.num_points = num_points
        self.data_by_log_count = defaultdict(list)
        
        # Load empty scene point cloud first
        empty_scene_file = None
        for file in os.listdir(data_dir):
            if 'pointcloud.npy' in file:
                timetag = '_'.join(file.split('_')[:4])
                metadata_file = f"{timetag}_metadata.json"
                
                if os.path.exists(os.path.join(data_dir, metadata_file)):
                    with open(os.path.join(data_dir, metadata_file), 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('num_logs', 0) == 0:
                        empty_scene_file = os.path.join(data_dir, file)
                        break
        
        if empty_scene_file is None:
            raise ValueError("No empty scene (0 logs) found in dataset!")
            
        # Load empty scene and store as numpy array
        self.empty_scene = np.load(empty_scene_file)
        self.empty_scene = self.empty_scene.reshape(-1, 3).astype(np.float32)
        
        # Organize data by log count, skipping the empty scene
        for file in os.listdir(data_dir):
            if 'pointcloud.npy' in file:
                timetag = '_'.join(file.split('_')[:4])
                metadata_file = f"{timetag}_metadata.json"
                
                if os.path.exists(os.path.join(data_dir, metadata_file)):
                    with open(os.path.join(data_dir, metadata_file), 'r') as f:
                        metadata = json.load(f)
                    
                    num_logs = metadata.get('num_logs', 0)
                    if num_logs > 0:  # Skip empty scenes
                        self.data_by_log_count[num_logs].append({
                            'point_cloud': os.path.join(data_dir, file),
                            'metadata': metadata
                        })
        
        # Create dataset
        self.data_samples = []
        for log_count in sorted(self.data_by_log_count.keys()):
            samples = self.data_by_log_count[log_count]
            print(f"Found {len(samples)} samples for {log_count} logs")
            self.data_samples.extend(samples)
        
        print(f"Total samples: {len(self.data_samples)}")

    def __len__(self):
        return len(self.data_samples)

    def find_different_points(self, point_cloud, threshold=0.05):
        """Fast filtering of points that differ from empty scene"""
        # Ensure both arrays are float32
        point_cloud = point_cloud.astype(np.float32)
        
        # First filter by height (Z coordinate) - much faster initial filter
        height_threshold = 0.02
        above_ground = point_cloud[:, 2] > height_threshold
        point_cloud = point_cloud[above_ground]
        
        # Then do a simple distance check from empty scene mean
        empty_mean = np.mean(self.empty_scene, axis=0)
        distances = np.abs(point_cloud - empty_mean)
        
        # Point is different if it's far enough from empty scene mean in any dimension
        mask = np.any(distances > threshold, axis=1)
        
        return point_cloud[mask]

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Load point cloud
        point_cloud = np.load(sample['point_cloud'])
        point_cloud = point_cloud.reshape(-1, 3)
        
        # Apply fast filtering
        point_cloud = self.find_different_points(point_cloud)
        
        # Center the remaining points
        if point_cloud.shape[0] > 0:
            center = np.mean(point_cloud, axis=0)
            point_cloud = point_cloud - center
        
        # Create array of exact size
        final_points = np.zeros((self.num_points, 3))
        
        if point_cloud.shape[0] >= self.num_points:
            # If we have more points, randomly sample
            idx = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            final_points = point_cloud[idx]
        elif point_cloud.shape[0] > 0:
            # If we have fewer points but not zero, use all points and pad with zeros
            final_points[:point_cloud.shape[0]] = point_cloud
        
        # Convert to tensor
        point_cloud = torch.FloatTensor(final_points)
        point_cloud = point_cloud.transpose(0, 1)  # Shape: (3, N)
        
        # Get number of logs
        num_logs = sample['metadata'].get('num_logs', 0)
        
        return point_cloud.to(self.device), torch.tensor(num_logs, dtype=torch.long).to(self.device)

    def get_log_counts(self):
        """Return sorted list of unique log counts"""
        return sorted(self.data_by_log_count.keys())

    def normalize_point_cloud(self, point_cloud):
        """Center and scale point cloud"""
        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid
        furthest_distance = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
        point_cloud = point_cloud / furthest_distance
        return point_cloud

    def visualize_preprocessing(self, point_cloud):
        """Debug visualization of preprocessing steps"""
        fig = plt.figure(figsize=(15, 5))
        
        # Original
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
        ax1.set_title("Original")
        
        # After filtering
        mask = np.abs(point_cloud[:, 2]) > 0.05  # Simple height-based filtering
        pc_filtered = point_cloud[mask]
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(pc_filtered[:, 0], pc_filtered[:, 1], pc_filtered[:, 2], s=1)
        ax2.set_title("After Filtering")
        
        # After normalization and padding/sampling
        final_points = np.zeros((self.num_points, 3))
        if pc_filtered.shape[0] >= self.num_points:
            idx = np.random.choice(pc_filtered.shape[0], self.num_points, replace=False)
            final_points = pc_filtered[idx]
        else:
            final_points[:pc_filtered.shape[0]] = pc_filtered
        
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(final_points[:, 0], final_points[:, 1], final_points[:, 2], s=1)
        ax3.set_title(f"Final ({self.num_points} points)")
        
        for ax in [ax1, ax2, ax3]:
            ax.view_init(elev=5, azim=315)
            ax.set_box_aspect([1, 1, 0.5])
        
        plt.tight_layout()
        plt.show()

    def visualize_sample(self, idx):
        """Visualize a sample before and after preprocessing"""
        sample = self.data_samples[idx]
        
        # Load original point cloud
        original = np.load(sample['point_cloud'])
        original = original.reshape(-1, 3)
        
        # Get processed point cloud
        processed, label = self[idx]
        processed = processed.cpu().numpy().T  # Convert back to (N, 3)
        
        fig = plt.figure(figsize=(15, 5))
        
        # Original
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1)
        ax1.set_title(f"Original - {sample['metadata'].get('num_logs', 0)} logs")
        
        # Processed
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(processed[:, 0], processed[:, 1], processed[:, 2], s=1)
        ax2.set_title("After Preprocessing")
        
        for ax in [ax1, ax2]:
            ax.view_init(elev=5, azim=315)
            ax.set_box_aspect([1, 1, 0.5])
        
        plt.tight_layout()
        plt.show()
