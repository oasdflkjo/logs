import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from collections import defaultdict

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
        
        # Remove ground points
        point_cloud = self.filter_ground_points(point_cloud)
        
        # Normalize point cloud
        point_cloud = self.normalize_point_cloud(point_cloud)
        
        # Sample fixed number of points
        point_cloud = self.sample_points(point_cloud, self.num_points)
        
        # Convert to tensor
        point_cloud = torch.FloatTensor(point_cloud)
        
        # Ensure shape is (3, N) for PointNet
        point_cloud = point_cloud.transpose(0, 1)
        
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
