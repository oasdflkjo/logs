import torch
import numpy as np
from dataset import PointCloudDataset
from pointnet_modified import PointNet
import torch_directml
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import seaborn as sns

def test_model(model_path, test_data_dir, device='cpu'):
    # Load dataset
    dataset = PointCloudDataset(test_data_dir, num_points=4096, device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Load checkpoint first to check model type
    checkpoint = torch.load('best_model.pth')
    
    # Check the shape of the last layer in checkpoint
    last_layer_weight = checkpoint['model_state_dict']['count_features.11.weight']
    output_size = last_layer_weight.shape[0]  # Get actual size from checkpoint
    
    # Initialize model with same architecture as checkpoint
    model = PointNet(num_classes=output_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    # Test the model
    with torch.no_grad():
        for points, labels in dataloader:
            outputs = model(points)
            
            if output_size == 1:  # Regression model
                predictions = outputs.squeeze()
            else:  # Classification model
                predictions = outputs.argmax(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    mse = np.mean((all_targets - all_predictions) ** 2)
    rmse = np.sqrt(mse)
    
    # Print results
    print(f"\nTest Results:")
    print(f"Model type: {'Regression' if output_size == 1 else 'Classification'}")
    print(f"Mean Absolute Error: {mae:.2f} logs")
    print(f"Root Mean Square Error: {rmse:.2f} logs")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(131)
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.plot([0, 20], [0, 20], 'r--')  # Perfect prediction line
    plt.xlabel('True Log Count')
    plt.ylabel('Predicted Log Count')
    plt.title('Predictions vs Ground Truth')
    plt.grid(True)
    
    # Error distribution
    plt.subplot(132)
    errors = all_predictions - all_targets
    sns.histplot(errors, bins=30)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    
    # Box plot of errors by true count
    plt.subplot(133)
    error_data = []
    labels = []
    for i in range(1, 21):
        mask = all_targets == i
        if np.any(mask):
            error_data.append(errors[mask])
            labels.append(str(i))
    
    plt.boxplot(error_data, labels=labels)
    plt.xlabel('True Log Count')
    plt.ylabel('Prediction Error')
    plt.title('Error by Log Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return mae, rmse

if __name__ == "__main__":
    device = torch_directml.device()
    test_model('best_model.pth', "C:/output", device) 