import cv2
import os
import glob
from tqdm import tqdm

def create_video_from_images(input_dir="C:\\output", output_file="dataset_preview.mp4", fps=30):
    """Create a video from all PNG files in the input directory"""
    
    # Get all PNG files
    png_files = glob.glob(os.path.join(input_dir, "*_render.png"))
    png_files.sort()  # Sort files by name
    
    if not png_files:
        print(f"No PNG files found in {input_dir}")
        return
    
    # Read first image to get dimensions
    first_image = cv2.imread(png_files[0])
    height, width, layers = first_image.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Add progress bar
    print(f"\nCreating video from {len(png_files)} images...")
    for png_file in tqdm(png_files):
        # Read image
        image = cv2.imread(png_file)
        
        # Add filename as text on the image
        filename = os.path.basename(png_file)
        cv2.putText(image, filename, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Write frame
        video.write(image)
    
    # Release video writer
    video.release()
    
    print(f"\nVideo created: {output_file}")
    print(f"- Duration: {len(png_files)/fps:.1f} seconds")
    print(f"- Frame count: {len(png_files)}")
    print(f"- FPS: {fps}")
    print(f"- Resolution: {width}x{height}")

if __name__ == "__main__":
    # You can customize these parameters
    INPUT_DIR = "C:\\output"
    OUTPUT_FILE = "C:\\output\\dataset_preview.mp4"
    FPS = 30
    
    try:
        create_video_from_images(INPUT_DIR, OUTPUT_FILE, FPS)
    except Exception as e:
        print(f"Error: {str(e)}") 