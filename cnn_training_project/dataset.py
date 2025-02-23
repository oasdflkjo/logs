import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree

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

        # Find and process empty scene for reference
        self.empty_scene_points = None
        for idx in range(len(self.data_samples)):
            sample = self.data_samples[idx]
            if sample['metadata'].get('num_logs', 0) == 0:
                point_cloud = np.load(sample['point_cloud'])
                point_cloud = point_cloud.reshape(-1, 3)
                mask = point_cloud[:, 2] != 0
                self.empty_scene_points = point_cloud[mask]
                break

        print(f"Empty scene reference points: {len(self.empty_scene_points) if self.empty_scene_points is not None else 'None'}")

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

    def normalize_point_cloud(self, point_cloud):
        """Normalize point cloud with PCA-based alignment and better scaling"""
        # Center the point cloud
        centroid = np.mean(point_cloud, axis=0)
        centered = point_cloud - centroid

        # Compute PCA
        try:
            # Get principal axes
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            
            # Sort by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Align to principal axes
            aligned = centered @ eigenvectors

            # Scale based on the largest dimension while preserving aspect ratios
            scale = np.max(np.abs(aligned))
            if scale > 0:
                aligned = aligned / scale * 0.9  # Leave some margin from [-1, 1] boundary
            
            return aligned
        except np.linalg.LinAlgError:
            # Fallback for degenerate cases
            print("Warning: PCA failed, using basic normalization")
            scale = np.max(np.abs(centered))
            if scale > 0:
                return centered / scale * 0.9
            return centered

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Load point cloud
        point_cloud = np.load(sample['point_cloud'])
        
        # Use the preprocessing function with empty scene reference
        point_cloud = self.preprocess_point_cloud(
            point_cloud, 
            self.num_points,
            self.empty_scene_points
        )
        
        # Get number of logs
        num_logs = sample['metadata'].get('num_logs', 0)
        
        return point_cloud.to(self.device), torch.tensor(num_logs, dtype=torch.long).to(self.device)

    def get_log_counts(self):
        """Return sorted list of unique log counts"""
        return sorted(self.data_by_log_count.keys())

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
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1)
        ax1.set_title(f"Original - {sample['metadata'].get('num_logs', 0)} logs")
        
        # After centering
        centered = original - np.mean(original, axis=0)
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(centered[:, 0], centered[:, 1], centered[:, 2], s=1)
        ax2.set_title("After Centering")
        
        # After PCA alignment and normalization
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(processed[:, 0], processed[:, 1], processed[:, 2], s=1)
        ax3.set_title("After PCA & Normalization")
        
        for ax in [ax1, ax2, ax3]:
            ax.view_init(elev=20, azim=45)
            ax.set_box_aspect([1,1,0.5])
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def preprocess_point_cloud(point_cloud, num_points=4096, empty_scene_points=None):
        """
        Optimized preprocessing function
        """
        # Reshape if needed
        point_cloud = point_cloud.reshape(-1, 3)
        
        # Remove ground points
        mask = point_cloud[:, 2] != 0
        point_cloud = point_cloud[mask]

        # If we have an empty scene reference, remove matching points
        if empty_scene_points is not None:
            # Downsample empty scene points for faster KDTree queries
            if len(empty_scene_points) > 1000:
                idx = np.random.choice(len(empty_scene_points), 1000, replace=False)
                empty_scene_query = empty_scene_points[idx]
            else:
                empty_scene_query = empty_scene_points
            
            # Create KD-tree for empty scene points
            empty_tree = cKDTree(empty_scene_query)
            
            # Process in batches to prevent memory issues
            batch_size = 10000
            unique_points = []
            
            for i in range(0, len(point_cloud), batch_size):
                batch = point_cloud[i:i + batch_size]
                distances, _ = empty_tree.query(batch, k=1)
                unique_mask = distances > 0.15
                unique_points.append(batch[unique_mask])
            
            point_cloud = np.concatenate(unique_points) if unique_points else point_cloud[:10]
            
            if len(point_cloud) < 10:  # If almost all points filtered out, it's probably an empty scene
                point_cloud = empty_scene_points[:10]  # Use a few reference points

        # PCA-based normalization
        centroid = np.mean(point_cloud, axis=0)
        centered = point_cloud - centroid

        try:
            # Get principal axes
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            
            # Sort by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Align to principal axes
            aligned = centered @ eigenvectors

            # Scale based on the largest dimension while preserving aspect ratios
            scale = np.max(np.abs(aligned))
            if scale > 0:
                aligned = aligned / scale * 0.9
            point_cloud = aligned
        except np.linalg.LinAlgError:
            # Fallback for degenerate cases
            scale = np.max(np.abs(centered))
            if scale > 0:
                point_cloud = centered / scale * 0.9
            else:
                point_cloud = centered

        # Sample points
        if point_cloud.shape[0] > num_points:
            idx = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        else:
            idx = np.random.choice(point_cloud.shape[0], num_points, replace=True)
        point_cloud = point_cloud[idx]
        
        # Convert to tensor and transpose
        point_cloud = torch.FloatTensor(point_cloud)
        point_cloud = point_cloud.transpose(0, 1)  # Shape: (3, N)
        
        return point_cloud
