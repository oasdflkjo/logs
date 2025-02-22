import bpy
import random
import math
import time
from datetime import datetime
import traceback
import sys
import os

# Add the project directory to Python path to find virtual_lidar
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from virtual_lidar import capture_final_state, VirtualLidarScanner

# Configuration
RANDOM_SEED = None  # Set to None for random, or an integer for reproducible results
NUM_LOGS = 50  # Number of logs to spawn
GROUND_SIZE = 50  # Larger ground plane (50 meters)
SIMULATION_FRAMES = 250  # Longer single simulation
LOG_SIZE = {
    'radius': 0.15,  # 30cm diameter
    'depth': 4.0     # 4 meters length
}

# Spawn configuration
SPAWN_HEIGHT_START = 2.0  # Start spawning from 2 meters up
SPAWN_AREA_RADIUS = 0.6   # Reduced from 0.8 to keep logs more centered
VERTICAL_SPACING = 0.4    # Space between logs in vertical direction
BATCH_SIZE = 3           # How many logs to spawn at once

# Physics configuration
PHYSICS_TIME_SCALE = 2.0    # Speed up physics (2x faster)
PHYSICS_STEPS = 10          # Frames to simulate between spawns
PHYSICS_ITERATIONS = 10     # Solver iterations (lower = faster but less accurate)

def show_message(message):
    """Simple message display"""
    print(message, flush=True)  # Force flush the print buffer

def set_random_seed():
    if RANDOM_SEED is None:
        # Get current timestamp but use a smaller number
        seed = int((datetime.now().timestamp() % 10000) * 1000)  # This will keep the number manageable
    else:
        seed = RANDOM_SEED
    
    random.seed(seed)
    show_message(f"Using random seed: {seed}")
    return seed

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_ground():
    # Create initial plane
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    
    # Add more geometry to allow for smooth deformation
    bpy.ops.object.modifier_add(type='SUBSURF')
    ground.modifiers["Subdivision"].levels = 3  # Reduced from 5
    ground.modifiers["Subdivision"].render_levels = 3
    
    # Add displacement modifier
    displace = ground.modifiers.new(name="Bowl_Shape", type='DISPLACE')
    
    # Create new texture for displacement
    tex = bpy.data.textures.new('Bowl_Texture', type='BLEND')
    tex.use_color_ramp = True
    tex.color_ramp.elements[0].position = 0.0
    tex.color_ramp.elements[1].position = 1.0
    tex.color_ramp.elements[0].color = (1, 1, 1, 1)  # Center (max displacement)
    tex.color_ramp.elements[1].color = (0, 0, 0, 1)  # Edge (no displacement)
    
    # Configure texture for radial gradient
    tex.progression = 'SPHERICAL'
    
    # Apply texture to displacement modifier
    displace.texture = tex
    displace.strength = 0.2  # Much smaller value for subtle effect
    displace.direction = 'Z'  # Displace upward
    displace.mid_level = 1.0  # Changed to 1.0 to invert the effect
    
    # Add rigid body physics
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.friction = 1.0
    ground.rigid_body.collision_shape = 'MESH'
    ground.rigid_body.use_margin = True
    ground.rigid_body.collision_margin = 0.001
    
    # Apply modifiers to make physics work correctly
    bpy.context.view_layer.objects.active = ground
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    bpy.ops.object.modifier_apply(modifier="Bowl_Shape")
    
    show_message("Created subtle bowl-shaped ground")

def spawn_log(index):
    """Spawn a single log with good spacing"""
    # Calculate position with slight randomness
    angle = random.uniform(0, math.pi * 2)  # Random angle around center
    radius = random.uniform(0, SPAWN_AREA_RADIUS)  # Random distance from center
    
    x = math.cos(angle) * radius
    y = math.sin(angle) * radius
    
    # Calculate height - higher index means higher position
    height = SPAWN_HEIGHT_START + (index * VERTICAL_SPACING)
    
    bpy.ops.mesh.primitive_cylinder_add(
        radius=LOG_SIZE['radius'],
        depth=LOG_SIZE['depth'],
        location=(x, y, height)
    )
    
    log = bpy.context.object
    log.name = f"Log_{len(bpy.data.objects)}"
    
    # Almost vertical orientation with slight randomness
    log.rotation_euler = (
        math.radians(85 + random.uniform(-3, 3)),  # Nearly vertical
        random.uniform(-0.1, 0.1),                 # Slight tilt
        random.uniform(0, math.pi * 2)             # Random rotation around vertical
    )
    
    # Physics settings
    bpy.ops.rigidbody.object_add()
    log.rigid_body.type = 'ACTIVE'
    log.rigid_body.mass = 50
    log.rigid_body.friction = 0.8
    log.rigid_body.restitution = 0.0
    log.rigid_body.collision_shape = 'CYLINDER'
    log.rigid_body.collision_margin = 0.001
    log.rigid_body.linear_damping = 0.6
    log.rigid_body.angular_damping = 0.6
    
    return log

class AutoCaptureOperator(bpy.types.Operator):
    """Operator that captures scene state at end of animation"""
    bl_idname = "scene.auto_capture"
    bl_label = "Auto Capture"
    
    _timer = None
    
    def modal(self, context, event):
        if event.type == 'TIMER':
            if context.scene.frame_current == context.scene.frame_end:
                show_message("\nEnd of animation loop - capturing state...")
                scanner = VirtualLidarScanner()
                scanner.metadata_seed = context.scene.get('random_seed', None)
                capture_final_state(scanner)
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        return {'CANCELLED'}

def setup_scene():
    # Set up the physics scene
    scene = bpy.context.scene
    scene.use_gravity = True
    scene.gravity = (0, 0, -9.81)
    
    # Create rigid body world if it doesn't exist
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    
    # Set up rigid body world with faster settings
    scene.rigidbody_world.enabled = True
    scene.rigidbody_world.solver_iterations = PHYSICS_ITERATIONS
    scene.rigidbody_world.time_scale = PHYSICS_TIME_SCALE  # Speed up physics
    
    # Set animation range
    scene.frame_start = 1
    scene.frame_end = SIMULATION_FRAMES
    scene.frame_current = 1
    
    # Set scene frame rate
    scene.render.fps = 24
    
    # Camera setup
    camera_distance = 15
    bpy.ops.object.camera_add(
        location=(camera_distance, -camera_distance, 1.7),
        rotation=(0, 0, math.radians(45))
    )
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    
    # Camera target
    target_empty = bpy.data.objects.new("CameraTarget", None)
    bpy.context.scene.collection.objects.link(target_empty)
    target_empty.location = (0, 0, 1)
    
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = target_empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    cam.data.lens = 35
    cam.data.clip_end = 1000
    
    # Switch to camera view
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces[0].region_3d.view_perspective = 'CAMERA'
            break

    # Register and start the auto-capture operator
    bpy.utils.register_class(AutoCaptureOperator)
    bpy.ops.scene.auto_capture()
    
    show_message("Added automatic capture at end of each loop")

def run_physics_simulation(frame_count):
    scene = bpy.context.scene
    
    # Reset to start frame
    scene.frame_current = 1
    
    # Free bake if it exists
    bpy.ops.ptcache.free_bake_all()
    
    # Let the physics initialize
    bpy.context.view_layer.update()
    
    show_message("Physics ready - press Alt+A to play animation")

def main():
    # Set random seed at the start
    seed = set_random_seed()
    
    clear_scene()
    setup_scene()
    create_ground()
    
    # Store seed in scene for frame handler to access
    bpy.context.scene['random_seed'] = seed
    
    show_message(f"Spawning {NUM_LOGS} logs gradually (seed: {seed})...")
    
    # Spawn logs in small batches
    logs_spawned = 0
    
    while logs_spawned < NUM_LOGS:
        # Spawn a small batch
        logs_to_spawn = min(BATCH_SIZE, NUM_LOGS - logs_spawned)
        
        # Spawn batch
        for i in range(logs_to_spawn):
            spawn_log(logs_spawned + i)
        
        logs_spawned += logs_to_spawn
        
        # Run physics for a few frames to let logs settle
        for _ in range(PHYSICS_STEPS):
            bpy.context.scene.frame_set(bpy.context.scene.frame_current + 1)
            bpy.context.view_layer.update()
        
        show_message(f"Spawned {logs_spawned}/{NUM_LOGS} logs...")
    
    show_message("Scene setup complete. You can now:")
    show_message("1. Press Alt+A to play/restart the animation")
    show_message("2. Press Spacebar to pause")
    show_message("3. Use the timeline to scrub through the animation")
    show_message(f"4. To recreate this exact arrangement, use seed: {seed}")
    
    show_message("\nAutomatic capture enabled - new images will be taken at the end of each loop")

# Make sure to unregister the class when the script is reloaded
if "AutoCaptureOperator" in bpy.types.Operator.__subclasses__():
    bpy.utils.unregister_class(AutoCaptureOperator)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        show_message(f"Error occurred: {str(e)}")
        show_message(traceback.format_exc())