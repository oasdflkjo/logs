import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from matplotlib.widgets import Button
from matplotlib.image import imread
import torch
import torch_directml
from pointnet_modified import PointNet
from dataset import PointCloudDataset
import json
from collections import defaultdict

class PointCloudViewer:
    def __init__(self, output_dir="C:\\output"):
        self.output_dir = output_dir
        
        # Setup model and device
        self.device = torch_directml.device()
        self.model = self.load_model()
        self.dataset = PointCloudDataset(output_dir, device=self.device)
        
        # Organize files by log count and sort them
        self.files_by_log_count = defaultdict(list)
        self.max_files_per_count = 0  # Track max files for any log count
        
        # Load and organize all files
        for file in glob.glob(os.path.join(output_dir, "*_pointcloud.npy")):
            timetag = '_'.join(file.split('_')[:4])
            metadata_file = os.path.join(output_dir, f"{timetag}_metadata.json")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                num_logs = metadata.get('num_logs', 0)
                self.files_by_log_count[num_logs].append(file)
        
        # Sort files within each log count
        for log_count in self.files_by_log_count:
            self.files_by_log_count[log_count].sort()
            self.max_files_per_count = max(self.max_files_per_count, 
                                         len(self.files_by_log_count[log_count]))
        
        # Initialize position
        self.current_log_count = 1
        self.file_position = 0  # Position in the sequence
        
        if not self.files_by_log_count:
            raise Exception(f"No point cloud files found in {output_dir}")
        
        # Create figure with adjusted proportions
        self.fig = plt.figure(figsize=(20, 8))
        
        # Create gridspec with minimal spacing
        gs = self.fig.add_gridspec(1, 2, 
                                  width_ratios=[1.2, 1], 
                                  wspace=0.05,
                                  left=0.05,
                                  right=0.95)
        
        self.ax_pc = self.fig.add_subplot(gs[0], projection='3d')
        self.ax_img = self.fig.add_subplot(gs[1])
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial plot
        self.plot_current_data()
    
    def load_model(self):
        """Load the trained model"""
        model = PointNet(num_classes=21).to(self.device)
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model with accuracy: {checkpoint['accuracy']:.2f}%")
        return model

    def get_prediction(self, points):
        """Get model prediction for point cloud"""
        # Preprocess point cloud same as training
        points = points.reshape(-1, 3)
        mask = points[:, 2] != 0
        points = points[mask]
        
        # Center points
        center = np.mean(points, axis=0)
        points = points - center
        
        # Sample points if needed
        if points.shape[0] > self.dataset.num_points:
            idx = np.random.choice(points.shape[0], self.dataset.num_points, replace=False)
            points = points[idx]
        
        # Convert to tensor
        points = torch.FloatTensor(points).to(self.device)
        points = points.transpose(0, 1).unsqueeze(0)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = self.model(points)
            pred = output.argmax(1).item()
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            confidence = probs[pred].item() * 100
            
            # Get top 3 predictions
            top3_values, top3_indices = torch.topk(probs, 3)
            top3_preds = [(idx.item(), val.item() * 100) for idx, val in zip(top3_indices, top3_values)]
        
        return pred, confidence, top3_preds

    def get_current_file(self):
        """Get current file based on position"""
        files = self.files_by_log_count[self.current_log_count]
        # Use modulo to wrap around if we reach the end of files for this log count
        return files[self.file_position % len(files)]
    
    def next_file(self):
        """Move to next file, incrementing log count when needed"""
        self.file_position += 1
        if self.file_position >= self.max_files_per_count:
            # Move to next log count
            self.current_log_count = (self.current_log_count % 20) + 1
            self.file_position = 0
    
    def prev_file(self):
        """Move to previous file, decrementing log count when needed"""
        self.file_position -= 1
        if self.file_position < 0:
            # Move to previous log count
            self.current_log_count = ((self.current_log_count - 2) % 20) + 1
            self.file_position = self.max_files_per_count - 1

    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            self.next_file()
            self.plot_current_data()
        elif event.key == 'left':
            self.prev_file()
            self.plot_current_data()
        elif event.key == 'r':  # Reset view
            self.ax_pc.view_init(elev=5, azim=315)
            self.fig.canvas.draw_idle()
        elif event.key == '+' or event.key == '=':  # Zoom in
            current_xlim = self.ax_pc.get_xlim()
            current_ylim = self.ax_pc.get_ylim()
            current_zlim = self.ax_pc.get_zlim()
            self.ax_pc.set_xlim(current_xlim[0] * 0.9, current_xlim[1] * 0.9)
            self.ax_pc.set_ylim(current_ylim[0] * 0.9, current_ylim[1] * 0.9)
            self.ax_pc.set_zlim(current_zlim[0] * 0.9, current_zlim[1] * 0.9)
            self.fig.canvas.draw_idle()
        elif event.key == '-':  # Zoom out
            current_xlim = self.ax_pc.get_xlim()
            current_ylim = self.ax_pc.get_ylim()
            current_zlim = self.ax_pc.get_zlim()
            self.ax_pc.set_xlim(current_xlim[0] * 1.1, current_xlim[1] * 1.1)
            self.ax_pc.set_ylim(current_ylim[0] * 1.1, current_ylim[1] * 1.1)
            self.ax_pc.set_zlim(current_zlim[0] * 1.1, current_zlim[1] * 1.1)
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)

    def show(self):
        """Display usage instructions and show the plot"""
        print("\nUsage:")
        print("- Right Arrow: Next log count")
        print("- Left Arrow: Previous log count")
        print("- R: Reset view to match render")
        print("- +/-: Zoom in/out")
        print("- Q: Quit")
        plt.show()

    def plot_current_data(self):
        """Plot point cloud and render"""
        self.ax_pc.clear()
        self.ax_img.clear()
        
        current_file = self.get_current_file()
        render_file = current_file.replace('_pointcloud.npy', '_render.png')
        
        # Load and plot point cloud
        points = np.load(current_file)
        
        # Get model prediction
        pred, confidence, top3_preds = self.get_prediction(points)
        
        # Calculate height-based colors and transparency
        z_values = points[:, 2]
        min_z = np.percentile(z_values, 5)
        max_z = np.percentile(z_values, 95)
        
        # Normalize z values for coloring
        normalized_z = (z_values - min_z) / (max_z - min_z)
        normalized_z = np.clip(normalized_z, 0, 1)
        
        # Calculate alpha based on z value
        ground_threshold = 0.1
        alphas = np.clip((np.abs(z_values) - ground_threshold) / 0.5, 0, 1) * 0.8 + 0.1
        
        # Create color array with varying transparency
        colors = np.zeros((len(points), 4))
        colors[:, 2] = normalized_z  # Blue component
        colors[:, 1] = normalized_z * 0.5  # Green component
        colors[:, 0] = normalized_z * 0.2  # Red component
        colors[:, 3] = alphas  # Alpha channel
        
        # Plot points
        scatter = self.ax_pc.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors,
            marker='.',
            s=2,
        )
        
        # Set view angle
        self.ax_pc.view_init(elev=5, azim=315)
        
        # Set equal aspect ratio
        self.ax_pc.set_box_aspect([1, 1, 0.5])
        
        # Set axis labels
        self.ax_pc.set_xlabel('X')
        self.ax_pc.set_ylabel('Y')
        self.ax_pc.set_zlabel('Z')
        
        # Auto-scale axes
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        zoom = 0.4
        self.ax_pc.set_xlim(mid_x - max_range * zoom, mid_x + max_range * zoom)
        self.ax_pc.set_ylim(mid_y - max_range * zoom, mid_y + max_range * zoom)
        self.ax_pc.set_zlim(mid_z - max_range/2 * zoom, mid_z + max_range/2 * zoom)
        
        # Plot render image
        if os.path.exists(render_file):
            img = imread(render_file)
            self.ax_img.imshow(img)
            self.ax_img.axis('off')
        else:
            self.ax_img.text(0.5, 0.5, 'No render found', 
                            ha='center', va='center')
        
        # Update title with prediction info
        total_files = len(self.files_by_log_count[self.current_log_count])
        title = f"Actual: {self.current_log_count} logs | Predicted: {pred} logs (Confidence: {confidence:.1f}%)\n"
        title += f"Top 3 predictions: "
        title += ", ".join([f"{p} logs ({c:.1f}%)" for p, c in top3_preds])
        title += f"\nFile {self.file_position + 1}/{total_files}: {os.path.basename(current_file)}"
        
        self.fig.suptitle(title, fontsize=12, y=0.98)
        
        plt.tight_layout(rect=[0.05, 0.02, 0.95, 0.92])
        self.fig.canvas.draw_idle()

def main():
    try:
        viewer = PointCloudViewer()
        viewer.show()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 