import bpy
import numpy as np
import json
import os
from datetime import datetime
from mathutils import Vector
import math
import traceback

class VirtualLidarScanner:
    def __init__(self, output_dir="C:\\output", message_callback=None):
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # Include milliseconds
        self.metadata_seed = None
        self.show_message = message_callback or (lambda x: None)  # Default no-op if no callback provided
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # LiDAR parameters
        self.resolution = 128  # Increased for better detail
        self.vertical_fov = 45
        self.max_distance = 100
        self.batch_size = 1000  # Process rays in batches
        
    def capture_scene(self):
        """Capture the current scene state and generate all outputs"""
        # Get camera for reference viewpoint
        camera = bpy.context.scene.camera
        if not camera:
            raise Exception("No camera found in scene")
            
        # Make sure we're looking at the current state
        bpy.context.view_layer.update()
        
        # Create unique ID for this capture
        capture_id = f"capture_{self.timestamp}"
        
        try:
            # Collect log positions and metadata
            self.show_message("Collecting metadata...")
            metadata = self.collect_metadata()
            self.save_metadata(metadata, capture_id)
            
            # Generate point cloud
            self.show_message("Generating point cloud...")
            points = self.generate_point_cloud(camera)
            
            # Save outputs
            self.show_message("Saving render...")
            self.save_render(capture_id)
            
            self.show_message("Saving point cloud...")
            self.save_point_cloud(points, capture_id)
            
            return capture_id
            
        except Exception as e:
            error_msg = f"Error in capture_scene: {str(e)}\n{traceback.format_exc()}"
            self.show_message(error_msg)
            raise
        
    def collect_metadata(self):
        """Collect basic metadata about the simulation with type checking"""
        try:
            metadata = {
                "timestamp": str(self.timestamp),  # Ensure string
                "num_logs": int(len([obj for obj in bpy.data.objects if obj.name.startswith("Log_")])),
                "simulation_frames": int(bpy.context.scene.frame_end),
                "random_seed": str(self.metadata_seed) if self.metadata_seed is not None else None,
                "frame_number": int(bpy.context.scene.frame_current),
                "loop_count": int(bpy.context.scene.get('loop_count', 0))
            }
            self.show_message(f"Collected metadata: {metadata}")
            return metadata
        
        except Exception as e:
            error_msg = f"Error collecting metadata: {str(e)}\n{traceback.format_exc()}"
            self.show_message(error_msg)
            return {"timestamp": str(self.timestamp), "error": str(e)}
        
    def generate_point_cloud(self, camera):
        """Generate point cloud data from virtual LiDAR scan with batching"""
        points = []
        
        # Get camera parameters
        cam_data = camera.data
        # Calculate camera's field of view
        horizontal_fov = math.degrees(cam_data.angle_x)
        vertical_fov = math.degrees(cam_data.angle_y)
        
        # Calculate scan parameters
        vertical_step = vertical_fov / self.resolution
        horizontal_step = horizontal_fov / self.resolution
        
        # Get camera orientation
        cam_matrix = camera.matrix_world
        cam_rotation = cam_matrix.to_quaternion()
        origin = camera.location
        
        self.show_message(f"Starting point cloud generation from camera at {origin}")
        self.show_message(f"Camera FOV - Horizontal: {horizontal_fov:.1f}°, Vertical: {vertical_fov:.1f}°")
        
        try:
            total_points = self.resolution * self.resolution
            points_processed = 0
            ray_batch = []
            
            for v in range(self.resolution):
                # Calculate vertical angle from camera's perspective
                vertical_angle = (vertical_fov/2) - (v * vertical_step)
                
                for h in range(self.resolution):
                    # Calculate horizontal angle from camera's perspective
                    horizontal_angle = (-horizontal_fov/2) + (h * horizontal_step)
                    
                    # Create direction vector in camera space
                    direction = Vector((
                        math.tan(math.radians(horizontal_angle)),
                        math.tan(math.radians(vertical_angle)),
                        -1.0  # Forward direction in camera space
                    )).normalized()
                    
                    # Transform direction to world space
                    direction.rotate(cam_rotation)
                    
                    ray_batch.append((origin, direction))
                    
                    # Process batch when it's full or at the end
                    if len(ray_batch) >= self.batch_size or (v == self.resolution-1 and h == self.resolution-1):
                        # Process current batch
                        for ray_origin, ray_direction in ray_batch:
                            result = bpy.context.scene.ray_cast(
                                bpy.context.view_layer.depsgraph,
                                ray_origin,
                                ray_direction,
                                distance=self.max_distance
                            )
                            
                            if result[0]:  # If hit something
                                points.append(result[1])
                        
                        # Update progress
                        points_processed += len(ray_batch)
                        self.show_message(f"Point cloud generation: {(points_processed/total_points)*100:.1f}% complete ({len(points)} points found)")
                        
                        # Clear batch
                        ray_batch = []
                        
                        # Give Blender a chance to update
                        if points_processed % (self.batch_size * 4) == 0:
                            bpy.context.view_layer.update()
            
            self.show_message(f"Point cloud generation complete. Collected {len(points)} points")
            return np.array(points)
            
        except Exception as e:
            error_msg = f"Error generating point cloud: {str(e)}\n{traceback.format_exc()}"
            self.show_message(error_msg)
            raise
        
    def save_render(self, capture_id):
        """Save camera render of the scene"""
        render_path = os.path.abspath(os.path.join(self.output_dir, f"{capture_id}_render.png"))
        self.show_message(f"Saving render to: {render_path}")
        
        # Set up render settings
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = render_path
        
        # Render and save
        bpy.ops.render.render(write_still=True)
        
    def save_point_cloud(self, points, capture_id):
        """Save point cloud data"""
        pc_path = os.path.abspath(os.path.join(self.output_dir, f"{capture_id}_pointcloud.npy"))
        self.show_message(f"Saving point cloud to: {pc_path}")
        np.save(pc_path, points)
        
    def save_metadata(self, metadata, capture_id):
        """Save metadata JSON with error handling"""
        meta_path = os.path.abspath(os.path.join(self.output_dir, f"{capture_id}_metadata.json"))
        try:
            self.show_message(f"Attempting to save metadata to: {meta_path}")
            
            # Check for non-serializable types
            for key, value in metadata.items():
                try:
                    json.dumps({key: value})
                except TypeError as e:
                    error_msg = f"Non-serializable value - Key: {key}, Value: {value}, Type: {type(value)}"
                    self.show_message(error_msg)
                    metadata[key] = str(value)
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.show_message(f"Metadata saved to {meta_path}")
            
        except Exception as e:
            error_msg = f"Error saving metadata: {str(e)}\nContent: {metadata}\n{traceback.format_exc()}"
            self.show_message(error_msg)

def capture_final_state(scanner=None, message_callback=None):
    """Function to be called from the main script"""
    if scanner is None:
        scanner = VirtualLidarScanner(message_callback=message_callback)
    try:
        scene = bpy.context.scene
        current_count = scene.get('loop_count', 0)
        scene['loop_count'] = current_count + 1
        
        bpy.context.view_layer.update()
        
        capture_id = scanner.capture_scene()
        scanner.show_message(f"Scene captured successfully. ID: {capture_id}")
        scanner.show_message(f"Loop #{scene['loop_count']}")
        scanner.show_message(f"Files saved in: {scanner.output_dir}")
        return True
        
    except Exception as e:
        error_msg = f"Error capturing scene: {str(e)}\n{traceback.format_exc()}"
        scanner.show_message(error_msg)
        return False

if __name__ == "__main__":
    capture_final_state() 