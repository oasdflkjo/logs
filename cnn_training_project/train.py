import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
from pointnet_modified import PointNet
import torch_directml
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Setup
device = torch_directml.device()
print(f"Using device: {device}")

# Load dataset and verify preprocessing
dataset = PointCloudDataset("C:/output", num_points=4096, device=device)

# Visualize a few samples
print("\nVisualizing preprocessing...")
for i in range(3):  # Show 3 random samples
    idx = np.random.randint(len(dataset))
    dataset.visualize_sample(idx)

# Print dataset statistics
log_counts = dataset.get_log_counts()
print("\nDataset Statistics:")
print(f"Number of different log counts: {len(log_counts)}")
print(f"Log counts available: {log_counts}")

# Create weighted sampler to handle class imbalance
weights = torch.ones(len(dataset))
for idx in range(len(dataset)):
    _, label = dataset[idx]
    weights[idx] = 1.0 / dataset.data_by_log_count[label.item()].__len__()
sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset))

# Create data loader with sampler
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

# Model and optimizer
model = PointNet(num_classes=21).to(device)

# Early stopping helper class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_mae = float('inf')
        self.early_stop = False
        
    def __call__(self, mae):
        if mae < self.best_mae - self.min_delta:
            self.best_mae = mae
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Modify the training setup
def setup_training(model, learning_rate=0.01):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Modified scheduler for longer training
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    return optimizer, criterion, scheduler

# Initialize training components
optimizer, criterion, scheduler = setup_training(model)

class TrainingVisualizer:
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.maes = []
        
        # Create figure and subplots
        plt.ion()  # Interactive mode on
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle('Training Progress')
        
    def update(self, epoch, loss, accuracy, mae):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.maes.append(mae)
        
        epochs = list(range(1, len(self.losses) + 1))
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot metrics
        self.ax1.plot(epochs, self.losses, 'b-')
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.grid(True)
        
        self.ax2.plot(epochs, self.accuracies, 'g-')
        self.ax2.set_title('Accuracy (%)')
        self.ax2.set_xlabel('Epoch')
        self.ax2.grid(True)
        
        self.ax3.plot(epochs, self.maes, 'r-')
        self.ax3.set_title('Mean Absolute Error')
        self.ax3.set_xlabel('Epoch')
        self.ax3.grid(True)
        
        # Add current values to titles
        self.ax1.set_title(f'Loss: {loss:.4f}')
        self.ax2.set_title(f'Accuracy: {accuracy:.1f}%')
        self.ax3.set_title(f'MAE: {mae:.2f}')
        
        plt.tight_layout()
        plt.pause(0.1)

# Training loop
def train_one_epoch():
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for points, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(points)
        labels = labels.long()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    accuracy = (predictions == targets).mean() * 100
    mae = np.abs(predictions - targets).mean()
    
    return total_loss / len(dataloader), accuracy, mae

# Training with early stopping
max_epochs = 200
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
visualizer = TrainingVisualizer()

best_mae = float('inf')
epoch = 0

print("\nStarting training...")
while epoch < max_epochs and not early_stopping.early_stop:
    # Train one epoch
    avg_loss, accuracy, mae = train_one_epoch()
    
    # Update scheduler with MAE metric
    scheduler.step(mae)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Update visualization
    visualizer.update(epoch + 1, avg_loss, accuracy, mae)
    
    # Print progress
    print(f"\nEpoch {epoch+1}/{max_epochs}:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"MAE: {mae:.2f} logs")
    print(f"Learning Rate: {current_lr:.6f}")
    
    # Early stopping check
    early_stopping(mae)
    
    # Save best model
    if mae < best_mae:
        best_mae = mae
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mae': mae,
            'accuracy': accuracy,
            'training_history': {
                'losses': visualizer.losses,
                'accuracies': visualizer.accuracies,
                'maes': visualizer.maes
            }
        }, 'best_model.pth')
        print(f"New best model saved! MAE: {mae:.2f}")
    
    epoch += 1
    
    # Check if learning rate is too small
    if current_lr < scheduler.min_lrs[0]:
        print("\nLearning rate too small, stopping training")
        break

print(f"\nTraining completed after {epoch} epochs")
print(f"Best MAE achieved: {best_mae:.2f}")

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot visible

def plot_confusion_matrix(predictions, targets, log_counts):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=log_counts,
                yticklabels=log_counts)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Add this after training loop:
model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for points, labels in dataloader:
        outputs = model(points)
        predictions = outputs.argmax(1)
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

plot_confusion_matrix(all_predictions, all_targets, log_counts)
