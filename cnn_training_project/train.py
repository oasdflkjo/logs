import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import random
import os
from datetime import datetime
import torch.nn.functional as F

# Setup
device = torch_directml.device()
print(f"Using device: {device}")

# Load dataset with reduced point clouds
dataset = PointCloudDataset("C:/output", num_points=2048, device=device)  # Reduced from 4096 to 2048

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Total samples: {len(dataset)}")
log_counts = dataset.get_log_counts()
print(f"Number of different log counts: {len(log_counts)}")
print(f"Log counts available: {log_counts}")
for count in sorted(dataset.data_by_log_count.keys()):
    print(f"Class {count}: {len(dataset.data_by_log_count[count])} samples")

# Create weighted sampler with balanced weights
weights = torch.ones(len(dataset))
for idx in range(len(dataset)):
    _, label = dataset[idx]
    label_val = label.item()
    weights[idx] = 1.0 / dataset.data_by_log_count[label_val].__len__()

sampler = torch.utils.data.WeightedRandomSampler(weights, len(dataset))

# Increase batch size since we have fewer points per cloud
batch_size = 16  # Increased from 8 to 16
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    sampler=sampler,
    num_workers=0,
    pin_memory=False
)

# Model and optimizer
model = PointNet(num_classes=20).to(device)

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
def setup_training(model, learning_rate=0.005):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.momentum = 0.01
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True
    )
    
    # Class weights for 20 classes (1-20)
    class_weights = torch.ones(20)  # Changed from 21 to 20
    total_samples = sum(len(samples) for samples in dataset.data_by_log_count.values())
    for i in range(1, 21):  # Changed range to 1-20
        samples = len(dataset.data_by_log_count.get(i, []))
        if samples > 0:
            class_weights[i-1] = total_samples / (20 * samples)  # Adjusted index
    
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,
        verbose=True,
        min_lr=1e-6
    )
    
    return optimizer, criterion, scheduler

# Initialize training components
optimizer, criterion, scheduler = setup_training(model)

# Simplified visualization
class TrainingVisualizer:
    def __init__(self):
        self.losses = []
        self.maes = []
        
        plt.ion()  # Interactive mode on
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle('Training Progress')
        
    def update(self, epoch, loss, mae, predictions=None, targets=None):
        self.losses.append(loss)
        self.maes.append(mae)
        
        epochs = list(range(1, len(self.losses) + 1))
        
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot loss
        self.ax1.plot(epochs, self.losses, 'b-')
        self.ax1.set_title(f'Loss: {loss:.4f}')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.grid(True)
        
        # Plot MAE
        self.ax2.plot(epochs, self.maes, 'r-')
        self.ax2.set_title(f'MAE: {mae:.2f}')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('MAE')
        self.ax2.grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)

# Training loop with adjusted prediction logic
def train_one_epoch():
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for points, labels in dataloader:
        model.zero_grad(set_to_none=True)
        
        # Reduced noise augmentation since points are already filtered
        if random.random() < 0.3:  # Reduced probability
            points += torch.randn_like(points) * 0.01
        
        classification, regression = model(points)
        labels = labels.long()
        
        loss = model.get_loss(classification, regression, labels)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Modified prediction logic
        class_pred = classification.argmax(1)
        reg_pred = regression.squeeze()
        
        # Get classification confidence
        class_probs = F.softmax(classification, dim=1)
        confidence = class_probs.max(1)[0]
        
        # Simplified prediction logic (no empty scene handling needed)
        final_pred = torch.zeros_like(class_pred)
        
        # Two-stage prediction
        high_conf_mask = confidence > 0.8
        final_pred[high_conf_mask] = class_pred[high_conf_mask] + 1  # Add 1 to get back to 1-20 range
        
        # For low confidence cases, use weighted average
        low_conf_mask = ~high_conf_mask
        if low_conf_mask.any():
            weighted_pred = (class_pred.float() * 0.4 + reg_pred * 0.6)
            final_pred[low_conf_mask] = (weighted_pred[low_conf_mask].round().clamp(1, 20)).long()
        
        predictions.extend(final_pred.cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    mae = np.abs(predictions - targets).mean()
    
    return total_loss / len(dataloader), predictions, targets, mae

# Training function
def train():
    best_mae = float('inf')
    patience_counter = 0
    max_patience = 15
    visualizer = TrainingVisualizer()
    
    print("\nStarting training...")
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        avg_loss, predictions, targets, mae = train_one_epoch()
        
        # Update learning rate
        scheduler.step(mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update visualization
        visualizer.update(epoch + 1, avg_loss, mae)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{max_epochs}:")
        print(f"Loss: {avg_loss:.4f}")
        print(f"MAE: {mae:.2f} logs")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if mae < best_mae:
            best_mae = mae
            patience_counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'mae': mae,
                'training_history': {
                    'losses': visualizer.losses,
                    'maes': visualizer.maes
                }
            }, 'best_model.pth')
            
            print(f"New best model saved! MAE: {mae:.2f}")
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= max_patience:
            print(f"\nEarly stopping triggered! No improvement for {max_patience} epochs")
            break
        
        if current_lr < scheduler.min_lrs[0]:
            print("\nLearning rate too small, stopping training")
            break
    
    print(f"\nTraining completed after {epoch + 1} epochs")
    print(f"Best MAE achieved: {best_mae:.2f}")

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

def main():
    # Setup and dataset loading code stays at module level
    
    # Wrap the training in a main function
    print("\nStarting training...")
    train()  # Run the training
    
    # Only plot confusion matrix after training is complete
    print("\nGenerating confusion matrix...")
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

if __name__ == "__main__":
    main()
