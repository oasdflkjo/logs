import numpy as np
import matplotlib.pyplot as plt
from dataset import PointCloudDataset
from collections import defaultdict
import seaborn as sns
from scipy.spatial import cKDTree

def analyze_point_clouds(dataset):
    """Analyze point clouds and their characteristics"""
    
    # Statistics containers
    points_per_class = defaultdict(list)
    filtered_points_per_class = defaultdict(list)
    distance_distributions = []
    
    # Get empty scene reference
    empty_scene = dataset.empty_scene_points
    if empty_scene is None:
        print("No empty scene reference found!")
        return
    
    empty_tree = cKDTree(empty_scene)
    
    # Analyze each point cloud
    for idx in range(len(dataset)):
        sample = dataset.data_samples[idx]
        num_logs = sample['metadata'].get('num_logs', 0)
        
        # Load and preprocess point cloud
        point_cloud = np.load(sample['point_cloud'])
        point_cloud = point_cloud.reshape(-1, 3)
        mask = point_cloud[:, 2] != 0
        point_cloud = point_cloud[mask]
        
        # Store original point count
        points_per_class[num_logs].append(len(point_cloud))
        
        # Calculate distances to empty scene points
        distances, _ = empty_tree.query(point_cloud, k=1)
        unique_mask = distances > 0.15  # Increased from 0.05 to 0.15
        filtered_cloud = point_cloud[unique_mask]
        
        # Store filtered point count
        filtered_points_per_class[num_logs].append(len(filtered_cloud))
        
        # Store distance distribution for this sample
        distance_distributions.append(distances)
        
        # Progress indicator
        if idx % 100 == 0:
            print(f"Processed {idx}/{len(dataset)} point clouds")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # 1. Box plot of points per class
    plt.subplot(2, 2, 1)
    plot_data_original = [points_per_class[i] for i in sorted(points_per_class.keys())]
    plt.boxplot(plot_data_original)
    plt.title('Original Points per Log Count')
    plt.xlabel('Number of Logs')
    plt.ylabel('Number of Points')
    
    # 2. Box plot of filtered points per class
    plt.subplot(2, 2, 2)
    plot_data_filtered = [filtered_points_per_class[i] for i in sorted(filtered_points_per_class.keys())]
    plt.boxplot(plot_data_filtered)
    plt.title('Filtered Points per Log Count')
    plt.xlabel('Number of Logs')
    plt.ylabel('Number of Points')
    
    # 3. Distance distribution histogram
    plt.subplot(2, 2, 3)
    all_distances = np.concatenate(distance_distributions)
    plt.hist(all_distances, bins=50, density=True)
    plt.title('Distance Distribution to Empty Scene Points')
    plt.xlabel('Distance (m)')
    plt.ylabel('Density')
    
    # 4. Points reduction percentage
    plt.subplot(2, 2, 4)
    reduction_percents = []
    log_counts = []
    for log_count in sorted(points_per_class.keys()):
        orig = np.mean(points_per_class[log_count])
        filt = np.mean(filtered_points_per_class[log_count])
        reduction = (orig - filt) / orig * 100
        reduction_percents.append(reduction)
        log_counts.append(log_count)
    
    plt.bar(log_counts, reduction_percents)
    plt.title('Points Reduction Percentage by Log Count')
    plt.xlabel('Number of Logs')
    plt.ylabel('Reduction %')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    print("Original point cloud statistics:")
    for log_count in sorted(points_per_class.keys()):
        points = points_per_class[log_count]
        print(f"Log count {log_count:2d}: "
              f"mean={np.mean(points):8.1f}, "
              f"std={np.std(points):8.1f}, "
              f"min={np.min(points):6d}, "
              f"max={np.max(points):6d}")
    
    print("\nFiltered point cloud statistics:")
    for log_count in sorted(filtered_points_per_class.keys()):
        points = filtered_points_per_class[log_count]
        print(f"Log count {log_count:2d}: "
              f"mean={np.mean(points):8.1f}, "
              f"std={np.std(points):8.1f}, "
              f"min={np.min(points):6d}, "
              f"max={np.max(points):6d}")
    
    print("\nAverage reduction percentage by log count:")
    for log_count, reduction in zip(log_counts, reduction_percents):
        print(f"Log count {log_count:2d}: {reduction:5.1f}%")

def main():
    # Initialize dataset
    dataset = PointCloudDataset("C:/output", num_points=4096)
    
    # Run analysis
    analyze_point_clouds(dataset)

if __name__ == "__main__":
    main() 