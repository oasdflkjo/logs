import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, num_points=1024, device='cpu'):
        self.data_dir = data_dir
        self.device = device
        self.num_points = num_points
        self.data_by_log_count = defaultdict(list)  # Organize by number of logs
        
        # Organize data by log count
        for file in os.listdir(data_dir):
            if 'pointcloud.npy' in file:
                timetag = '_'.join(file.split('_')[:4])
                metadata_file = f"{timetag}_metadata.json"
                
                if os.path.exists(os.path.join(data_dir, metadata_file)):
                    with open(os.path.join(data_dir, metadata_file), 'r') as f:
                        metadata = json.load(f)
                    
                    num_logs = metadata.get('num_logs', 0)
                    self.data_by_log_count[num_logs].append({
                        'point_cloud': os.path.join(data_dir, file),
                        'metadata': metadata
                    })
        
        # Create balanced dataset
        self.data_samples = []
        for log_count in sorted(self.data_by_log_count.keys()):
            samples = self.data_by_log_count[log_count]
            print(f"Found {len(samples)} samples for {log_count} logs")
            self.data_samples.extend(samples)
        
        print(f"Total samples: {len(self.data_samples)}")

    def __len__(self):
        return len(self.data_samples)

    def filter_ground_points(self, point_cloud, z_threshold=0.05):
        """Remove points close to ground level"""
        mask = np.abs(point_cloud[:, 2]) > z_threshold
        return point_cloud[mask]

    def sample_points(self, point_cloud, num_points):
        """Randomly sample a fixed number of points"""
        if point_cloud.shape[0] >= num_points:
            idx = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        else:
            idx = np.random.choice(point_cloud.shape[0], num_points, replace=True)
        return point_cloud[idx]

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Load point cloud
        point_cloud = np.load(sample['point_cloud'])
        point_cloud = point_cloud.reshape(-1, 3)
        
        # Only filter points that are exactly at z=0 (ground plane)
        mask = point_cloud[:, 2] != 0
        point_cloud = point_cloud[mask]
        
        # Simple centering (keep the scale)
        center = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - center
        
        # Sample points if needed
        if point_cloud.shape[0] > self.num_points:
            idx = np.random.choice(point_cloud.shape[0], self.num_points, replace=False)
            point_cloud = point_cloud[idx]
        
        # Convert to tensor
        point_cloud = torch.FloatTensor(point_cloud)
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
        fig = plt.figure(figsize=(20, 5))
        
        # Original
        ax1 = fig.add_subplot(141, projection='3d')
        ax1.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
        ax1.set_title("Original")
        
        # After ground removal
        pc_no_ground = self.filter_ground_points(point_cloud)
        ax2 = fig.add_subplot(142, projection='3d')
        ax2.scatter(pc_no_ground[:, 0], pc_no_ground[:, 1], pc_no_ground[:, 2], s=1)
        ax2.set_title("Ground Removed")
        
        # After normalization
        pc_normalized = self.normalize_point_cloud(pc_no_ground)
        ax3 = fig.add_subplot(143, projection='3d')
        ax3.scatter(pc_normalized[:, 0], pc_normalized[:, 1], pc_normalized[:, 2], s=1)
        ax3.set_title("Normalized")
        
        # After sampling
        pc_sampled = self.sample_points(pc_normalized, self.num_points)
        ax4 = fig.add_subplot(144, projection='3d')
        ax4.scatter(pc_sampled[:, 0], pc_sampled[:, 1], pc_sampled[:, 2], s=1)
        ax4.set_title(f"Sampled ({self.num_points} points)")
        
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
