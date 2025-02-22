import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
from matplotlib.widgets import Button
from matplotlib.image import imread

class PointCloudViewer:
    def __init__(self, output_dir="C:\\output"):
        self.output_dir = output_dir
        self.pointcloud_files = self.get_pointcloud_files()
        self.current_index = 0
        
        if not self.pointcloud_files:
            raise Exception(f"No point cloud files found in {output_dir}")
        
        # Create figure with adjusted proportions
        self.fig = plt.figure(figsize=(20, 8))  # Reduced width from 24 to 20
        
        # Create gridspec with minimal spacing
        gs = self.fig.add_gridspec(1, 2, 
                                  width_ratios=[1.2, 1], 
                                  wspace=0.05,  # Reduced space between subplots
                                  left=0.05,    # Less margin on left
                                  right=0.95)   # Less margin on right
        
        # Point cloud subplot
        self.ax_pc = self.fig.add_subplot(gs[0], projection='3d')
        
        # Render image subplot
        self.ax_img = self.fig.add_subplot(gs[1])
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial plot
        self.plot_current_data()
        
    def get_pointcloud_files(self):
        """Get sorted list of point cloud files"""
        files = glob.glob(os.path.join(self.output_dir, "*_pointcloud.npy"))
        return sorted(files)
    
    def get_render_path(self, pointcloud_path):
        """Get corresponding render path for a pointcloud file"""
        base_path = pointcloud_path.replace('_pointcloud.npy', '_render.png')
        return base_path
    
    def print_pointcloud_info(self):
        """Print information about current point cloud"""
        points = np.load(self.pointcloud_files[self.current_index])
        points = points.reshape(-1, 3)  # Ensure correct shape
        print(f"\nPoint Cloud File: {os.path.basename(self.pointcloud_files[self.current_index])}")
        print(f"Number of points: {len(points)}")
        print(f"Shape: {points.shape}")
        print("\nBounding Box:")
        print(f"X: min={points[:, 0].min():.2f}, max={points[:, 0].max():.2f}")
        print(f"Y: min={points[:, 1].min():.2f}, max={points[:, 1].max():.2f}")
        print(f"Z: min={points[:, 2].min():.2f}, max={points[:, 2].max():.2f}")
    
    def plot_current_data(self):
        """Plot both point cloud and render"""
        # Clear both axes
        self.ax_pc.clear()
        self.ax_img.clear()
        
        current_file = self.pointcloud_files[self.current_index]
        render_file = self.get_render_path(current_file)
        
        # Load and plot point cloud
        points = np.load(current_file)
        points = points.reshape(-1, 3)  # Ensure correct shape
        
        # Calculate height-based colors and transparency
        z_values = points[:, 2]  # Get z coordinates
        min_z = np.percentile(z_values, 5)
        max_z = np.percentile(z_values, 95)
        
        # Normalize z values for coloring
        normalized_z = (z_values - min_z) / (max_z - min_z)
        normalized_z = np.clip(normalized_z, 0, 1)
        
        # Calculate alpha (transparency) based on z value
        # Very transparent near z=0, gradually more opaque as height increases
        ground_threshold = 0.1  # Points below this height are considered ground level
        alphas = np.clip((np.abs(z_values) - ground_threshold) / 0.5, 0, 1) * 0.8 + 0.1
        
        # Create color array with varying transparency
        colors = np.zeros((len(points), 4))  # RGBA array
        colors[:, 2] = normalized_z  # Blue component varies with height
        colors[:, 1] = normalized_z * 0.5  # Some green for mid-heights
        colors[:, 0] = normalized_z * 0.2  # A bit of red for highest points
        colors[:, 3] = alphas  # Alpha channel
        
        # Plot points
        scatter = self.ax_pc.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c=colors,
            marker='.',
            s=2,
        )
        
        # Set view to match render camera perspective
        self.ax_pc.view_init(elev=5, azim=315)
        
        # Set equal aspect ratio and view
        self.ax_pc.set_box_aspect([1, 1, 0.5])
        
        # Set axis labels
        self.ax_pc.set_xlabel('X')
        self.ax_pc.set_ylabel('Y')
        self.ax_pc.set_zlabel('Z')
        
        # Auto-scale axes to data with adjusted zoom
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        # Zoom factor
        zoom = 0.4
        
        self.ax_pc.set_xlim(mid_x - max_range * zoom, mid_x + max_range * zoom)
        self.ax_pc.set_ylim(mid_y - max_range * zoom, mid_y + max_range * zoom)
        self.ax_pc.set_zlim(mid_z - max_range/2 * zoom, mid_z + max_range/2 * zoom)
        
        # Plot render image if it exists
        if os.path.exists(render_file):
            img = imread(render_file)
            self.ax_img.imshow(img)
            self.ax_img.set_title("Rendered View")
            self.ax_img.axis('off')
        else:
            self.ax_img.text(0.5, 0.5, 'No render found', 
                           ha='center', va='center')
            self.ax_img.set_title("Missing Render")
        
        # Print info
        self.print_pointcloud_info()
        
        # Remove old titles
        self.ax_pc.set_title("")
        self.ax_img.set_title("")
        
        # Update display with better spacing
        self.fig.suptitle(os.path.basename(current_file), fontsize=14, y=0.98)
        
        # Add titles as figure text for better control
        self.fig.text(0.3, 0.95, f"Point Cloud {self.current_index + 1}/{len(self.pointcloud_files)}", 
                     fontsize=12, 
                     horizontalalignment='center')
        self.fig.text(0.7, 0.95, "Rendered View", 
                     fontsize=12, 
                     horizontalalignment='center')
        
        # Adjust layout with minimal padding
        plt.tight_layout(rect=[0.05, 0.02, 0.95, 0.92])
        self.fig.canvas.draw_idle()
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.pointcloud_files)
            self.plot_current_data()  # This will clear and redraw everything
        elif event.key == 'left':
            self.current_index = (self.current_index - 1) % len(self.pointcloud_files)
            self.plot_current_data()  # This will clear and redraw everything
        elif event.key == 'r':  # Reset view
            self.ax_pc.view_init(elev=5, azim=315)  # Match our current shallow angle
            self.fig.canvas.draw_idle()  # Just update the view, don't redraw everything
        elif event.key == 'up':
            current_elev = self.ax_pc.elev
            self.ax_pc.view_init(elev=current_elev + 5, azim=self.ax_pc.azim)
            self.fig.canvas.draw_idle()
        elif event.key == 'down':
            current_elev = self.ax_pc.elev
            self.ax_pc.view_init(elev=current_elev - 5, azim=self.ax_pc.azim)
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
        print("- Right Arrow: Next point cloud")
        print("- Left Arrow: Previous point cloud")
        print("- Up/Down Arrow: Adjust view angle")
        print("- +/-: Zoom in/out")
        print("- R: Reset view to match render")
        print("- Q: Quit")
        plt.show()

def main():
    try:
        viewer = PointCloudViewer()
        viewer.show()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 