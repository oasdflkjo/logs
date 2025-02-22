import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob

def load_and_plot_pointcloud(file_path):
    # Load the point cloud
    points = np.load(file_path)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title with file name
    ax.set_title(os.path.basename(file_path))
    
    # Add grid
    ax.grid(True)
    
    # Show plot
    plt.show()

def print_pointcloud_info(file_path):
    points = np.load(file_path)
    print(f"\nPoint Cloud File: {os.path.basename(file_path)}")
    print(f"Number of points: {len(points)}")
    print(f"Shape: {points.shape}")
    print("\nBounding Box:")
    print(f"X: min={points[:, 0].min():.2f}, max={points[:, 0].max():.2f}")
    print(f"Y: min={points[:, 1].min():.2f}, max={points[:, 1].max():.2f}")
    print(f"Z: min={points[:, 2].min():.2f}, max={points[:, 2].max():.2f}")

def main():
    # Path to output directory
    output_dir = "C:\\output"
    
    # Find all point cloud files
    pointcloud_files = glob.glob(os.path.join(output_dir, "*_pointcloud.npy"))
    
    if not pointcloud_files:
        print(f"No point cloud files found in {output_dir}")
        return
    
    # Sort files by timestamp (newest first)
    pointcloud_files.sort(reverse=True)
    
    # Print list of available files
    print("Available point cloud files:")
    for i, file in enumerate(pointcloud_files):
        print(f"{i+1}. {os.path.basename(file)}")
    
    # Let user choose which file to inspect
    while True:
        try:
            choice = input("\nEnter number of file to inspect (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
                
            file_index = int(choice) - 1
            if 0 <= file_index < len(pointcloud_files):
                chosen_file = pointcloud_files[file_index]
                print_pointcloud_info(chosen_file)
                load_and_plot_pointcloud(chosen_file)
            else:
                print("Invalid file number")
        except ValueError:
            print("Please enter a valid number")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 