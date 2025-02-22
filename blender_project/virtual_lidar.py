import bpy
import numpy as np
import json
import os
from datetime import datetime
from mathutils import Vector
import math
import traceback

class VirtualLidarScanner:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # Include milliseconds
        self.metadata_seed = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # LiDAR parameters
        self.resolution = 256
        self.vertical_fov = 45
        self.max_distance = 100
        
    def capture_scene(self):
        """Capture the current scene state and generate all outputs"""
        # Get camera for reference viewpoint
        camera = bpy.context.scene.camera
        if not camera:
            raise Exception("No camera found in scene")
            
        # Create unique ID for this capture
        capture_id = f"capture_{self.timestamp}"
        
        # Collect log positions and metadata
        metadata = self.collect_metadata()
        
        # Generate point cloud
        points = self.generate_point_cloud(camera)
        
        # Save outputs
        self.save_render(capture_id)
        self.save_point_cloud(points, capture_id)
        self.save_metadata(metadata, capture_id)
        
        return capture_id
        
    def collect_metadata(self):
        """Collect basic metadata about the simulation"""
        metadata = {
            "timestamp": self.timestamp,
            "num_logs": len([obj for obj in bpy.data.objects if obj.name.startswith("Log_")]),
            "simulation_frames": bpy.context.scene.frame_end,
            "random_seed": self.metadata_seed,
            "frame_number": bpy.context.scene.frame_current,
            "loop_count": bpy.context.scene.get('loop_count', 0)  # Use get() for safe access
        }
        return metadata
        
    def generate_point_cloud(self, camera):
        """Generate point cloud data from virtual LiDAR scan"""
        points = []
        
        # Calculate scan parameters
        vertical_step = self.vertical_fov / self.resolution
        horizontal_step = 360.0 / self.resolution
        
        # Cast rays from camera position
        origin = camera.location
        
        for v in range(self.resolution):
            vertical_angle = -self.vertical_fov/2 + v * vertical_step
            
            for h in range(self.resolution):
                horizontal_angle = h * horizontal_step
                
                # Calculate ray direction
                direction = Vector((
                    math.cos(math.radians(horizontal_angle)) * math.cos(math.radians(vertical_angle)),
                    math.sin(math.radians(horizontal_angle)) * math.cos(math.radians(vertical_angle)),
                    math.sin(math.radians(vertical_angle))
                ))
                
                # Cast ray
                result = bpy.context.scene.ray_cast(bpy.context.view_layer.depsgraph, origin, direction, distance=self.max_distance)
                
                if result[0]:  # If hit something
                    points.append(result[1])  # Add hit position
                    
        return np.array(points)
        
    def save_render(self, capture_id):
        """Save camera render of the scene"""
        render_path = os.path.join(self.output_dir, f"{capture_id}_render.png")
        
        # Set up render settings
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = render_path
        
        # Render and save
        bpy.ops.render.render(write_still=True)
        
    def save_point_cloud(self, points, capture_id):
        """Save point cloud data"""
        pc_path = os.path.join(self.output_dir, f"{capture_id}_pointcloud.npy")
        np.save(pc_path, points)
        
    def save_metadata(self, metadata, capture_id):
        """Save metadata JSON"""
        meta_path = os.path.join(self.output_dir, f"{capture_id}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

def capture_final_state(scanner=None):
    """Function to be called from the main script"""
    if scanner is None:
        scanner = VirtualLidarScanner()
    try:
        # Use a simple integer for loop counting
        scene = bpy.context.scene
        current_count = scene.get('loop_count', 0)
        scene['loop_count'] = current_count + 1
        
        # Make sure we're in a good state for rendering
        bpy.context.view_layer.update()
        
        # Attempt capture
        capture_id = scanner.capture_scene()
        print(f"Scene captured successfully. Capture ID: {capture_id}")
        print(f"Loop #{scene['loop_count']}")
        print(f"Files saved in: {scanner.output_dir}")
        return True
        
    except Exception as e:
        print(f"Error capturing scene: {str(e)}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    capture_final_state() 