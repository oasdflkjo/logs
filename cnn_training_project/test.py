import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import PointCloudDataset
from pointnet_modified import PointNet
import torch_directml
import os
from PIL import Image

class ModelTester:
    def __init__(self, model_path, data_dir, device):
        self.device = device
        self.data_dir = data_dir
        
        # Initialize dataset
        self.dataset = PointCloudDataset(data_dir, num_points=4096, device=device)
        
        # Load model
        self.model = PointNet(num_classes=21).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create persistent figure and axes
        self.fig = plt.figure(figsize=(15, 7))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        
    def get_png_path(self, pointcloud_path):
        """Convert pointcloud path to corresponding render image path"""
        base_name = os.path.basename(pointcloud_path)
        render_name = base_name.replace('_pointcloud.npy', '_render.png')
        return os.path.join(self.data_dir, render_name)
    
    def update_plot(self, idx):
        """Update the existing plot with new data"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Get sample data
        points, true_count = self.dataset[idx]
        sample = self.dataset.data_samples[idx]
        pointcloud_path = sample['point_cloud']
        image_path = self.get_png_path(pointcloud_path)
        
        # Get model prediction
        with torch.no_grad():
            points_batch = points.unsqueeze(0)
            pred_count = self.model(points_batch)
            pred_count = pred_count.squeeze().item()
        
        # Plot image if found
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            self.ax1.imshow(img)
            self.ax1.set_title("Original Image")
            self.ax1.axis('off')
        else:
            self.ax1.text(0.5, 0.5, f"Image not found\nPointcloud: {os.path.basename(pointcloud_path)}", 
                         ha='center', va='center')
            self.ax1.set_xlim(0, 1)
            self.ax1.set_ylim(0, 1)
        
        # Plot point cloud
        points_np = points.cpu().numpy().T
        self.ax2.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=1)
        self.ax2.view_init(elev=5, azim=315)
        self.ax2.set_box_aspect([1, 1, 0.5])
        
        # Add prediction information
        title = f"Point Cloud\nPredicted Logs: {pred_count:.1f}\nActual Logs: {true_count.item()}"
        self.ax2.set_title(title)
        
        # Update the display
        plt.tight_layout()
        self.fig.canvas.draw_idle()
        
        # Print detailed information
        print(f"\nSample {idx}:")
        print(f"Predicted log count: {pred_count:.1f}")
        print(f"Actual log count: {true_count.item()}")
        print(f"Absolute error: {abs(pred_count - true_count.item()):.1f}")
        print("\nPress right arrow for next random sample")
    
    def test_interactive(self):
        """Interactive testing loop"""
        num_samples = len(self.dataset)
        
        print("\nControls:")
        print("- Right arrow: Show next random sample")
        print("- Close window to quit")
        print("\nStarting with a random sample...")
        
        # Add key event handler
        def on_key(event):
            if event.key == 'right':
                next_idx = np.random.randint(num_samples)
                self.update_plot(next_idx)
        
        # Connect the key event handler
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Show first random sample
        idx = np.random.randint(num_samples)
        self.update_plot(idx)
        plt.show()

def main():
    # Setup
    device = torch_directml.device()
    print(f"Using device: {device}")
    
    # Configuration
    model_path = 'best_model.pth'
    data_dir = "C:/output"  # Update this to your data directory
    
    # Debug: List all files in directory
    print("\nFiles in data directory:")
    for file in os.listdir(data_dir):
        if file.endswith(('.png', '.jpg', '.npy')):
            print(f"  {file}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    # Create tester and run interactive testing
    tester = ModelTester(model_path, data_dir, device)
    tester.test_interactive()

if __name__ == "__main__":
    main() 