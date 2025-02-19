import bpy
import random
import math
import time
from datetime import datetime

# Configuration
RANDOM_SEED = None  # Set to None for random, or an integer for reproducible results
NUM_LOGS = 100  # Number of logs to spawn
GROUND_SIZE = 50  # Larger ground plane (50 meters)
SIMULATION_FRAMES = 250  # Longer single simulation
LOG_SIZE = {
    'radius': 0.15,  # 30cm diameter
    'depth': 4.0     # 4 meters length
}

# Spawn configuration
SPAWN_HEIGHT_START = 2.0  # Start spawning from 2 meters up
SPAWN_AREA_RADIUS = 0.8   # Tighter radius for vertical stacking
VERTICAL_SPACING = 0.4    # Space between logs in vertical direction
BATCH_SIZE = 3           # How many logs to spawn at once

# Physics configuration
PHYSICS_TIME_SCALE = 2.0    # Speed up physics (2x faster)
PHYSICS_STEPS = 10          # Frames to simulate between spawns
PHYSICS_ITERATIONS = 10     # Solver iterations (lower = faster but less accurate)

def set_random_seed():
    if RANDOM_SEED is None:
        # Get current timestamp for a unique seed
        seed = int(datetime.now().timestamp() * 1000)
    else:
        seed = RANDOM_SEED
    
    random.seed(seed)
    print(f"Using random seed: {seed}")
    return seed

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_ground():
    bpy.ops.mesh.primitive_plane_add(size=GROUND_SIZE, location=(0, 0, 0))
    ground = bpy.context.object
    ground.name = "Ground"
    
    # Add rigid body physics
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.friction = 1.0  # Maximum friction
    ground.rigid_body.collision_shape = 'BOX'  # More stable collision
    ground.rigid_body.use_margin = True
    ground.rigid_body.collision_margin = 0.001

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

def run_physics_simulation(frame_count):
    scene = bpy.context.scene
    
    # Reset to start frame
    scene.frame_current = 1
    
    # Free bake if it exists
    bpy.ops.ptcache.free_bake_all()
    
    # Let the physics initialize
    bpy.context.view_layer.update()
    
    print("Physics ready - press Alt+A to play animation")

def main():
    # Set random seed at the start
    seed = set_random_seed()
    
    clear_scene()
    setup_scene()
    create_ground()
    
    print(f"Spawning {NUM_LOGS} logs gradually (seed: {seed})...")
    
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
        
        print(f"Spawned {logs_spawned}/{NUM_LOGS} logs...")
    
    print("Scene setup complete. You can now:")
    print("1. Press Alt+A to play/restart the animation")
    print("2. Press Spacebar to pause")
    print("3. Use the timeline to scrub through the animation")
    print(f"4. To recreate this exact arrangement, use seed: {seed}")

if __name__ == "__main__":
    main() 