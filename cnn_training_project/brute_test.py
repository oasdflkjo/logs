import torch
import torch_directml
import numpy as np
from pointnet_modified import PointNet
from dataset import PointCloudDataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages

def save_plt_to_img():
    """Save current matplotlib figure to bytes buffer"""
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
    img_buffer.seek(0)
    plt.close()
    return img_buffer

def evaluate_model(all_predictions, all_targets):
    print("\nEvaluating model...")
    
    # Make sure both arrays have the same shape
    if len(all_predictions) != len(all_targets):
        print(f"Warning: Shape mismatch - predictions: {all_predictions.shape}, targets: {all_targets.shape}")
        # Take the minimum length to ensure they match
        min_len = min(len(all_predictions), len(all_targets))
        all_predictions = all_predictions[:min_len]
        all_targets = all_targets[:min_len]
    
    # Calculate metrics
    overall_accuracy = (all_predictions == all_targets).mean() * 100
    mae = np.abs(all_predictions - all_targets).mean()
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Generate timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create PDF report
    pdf_filename = f'model_evaluation_{timestamp}.pdf'
    
    with PdfPages(pdf_filename) as pdf:
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Log Count')
        plt.ylabel('Actual Log Count')
        pdf.savefig()
        plt.close()
        
        # Plot prediction distribution
        plt.figure(figsize=(12, 6))
        plt.hist(all_predictions, bins=20, alpha=0.5, label='Predictions')
        plt.hist(all_targets, bins=20, alpha=0.5, label='Actual')
        plt.title('Distribution of Predictions vs Actual')
        plt.xlabel('Log Count')
        plt.ylabel('Frequency')
        plt.legend()
        pdf.savefig()
        plt.close()
        
        # Add summary page
        plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.9, f'Model Evaluation Summary', fontsize=14, fontweight='bold')
        plt.text(0.1, 0.8, f'Overall Accuracy: {overall_accuracy:.2f}%')
        plt.text(0.1, 0.7, f'Mean Absolute Error: {mae:.2f} logs')
        plt.text(0.1, 0.6, f'Total Samples Evaluated: {len(all_predictions)}')
        plt.axis('off')
        pdf.savefig()
        plt.close()
    
    print(f"Evaluation report saved as {pdf_filename}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"Mean Absolute Error: {mae:.2f} logs")

def main():
    print("\nStarting model evaluation...")
    
    # Initialize device, dataset, model and dataloader
    device = torch_directml.device()
    dataset = PointCloudDataset("C:/output", num_points=4096, device=device)
    model = PointNet(num_classes=21).to(device)
    
    # Load the trained model
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,  # Larger batch size for faster evaluation
        shuffle=False,  # No need to shuffle during evaluation
        num_workers=0,
        pin_memory=False
    )
    
    all_predictions = []
    all_targets = []
    
    # Track progress
    total_batches = len(dataloader)
    print(f"\nEvaluating {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, (points, labels) in enumerate(dataloader):
            outputs = model(points)
            predictions = outputs.argmax(1).cpu().numpy()
            targets = labels.cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Print progress
            if (i + 1) % 10 == 0:
                print(f"Processed {(i + 1) * dataloader.batch_size}/{len(dataset)} samples")
    
    # Convert to numpy arrays before evaluation
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    evaluate_model(all_predictions, all_targets)

if __name__ == "__main__":
    main() 