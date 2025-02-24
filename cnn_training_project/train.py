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
from visualization import TrainingVisualizer, plot_confusion_matrix, create_training_video

# Setup
device = torch_directml.device()
print(f"Using device: {device}")

# Load dataset and verify preprocessing
dataset = PointCloudDataset("C:/output", num_points=4096, device=device)

# Visualize preprocessing for just one sample
print("\nVisualizing preprocessing...")
idx = np.random.randint(len(dataset))
dataset.visualize_sample(idx)

# Print dataset statistics
log_counts = dataset.get_log_counts()
print("\nDataset Statistics:")
print(f"Number of different log counts: {len(log_counts)}")
print(f"Log counts available: {log_counts}")

# Modify batch size and dataloader settings
batch_size = 32  # Keep batch size large enough
min_batch_size = 2  # Ensure at least 2 samples per batch

dataloader = DataLoader(
    dataset, 
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True  # Drop the last incomplete batch
)

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
def setup_training(model, learning_rate=0.005):
    # Switch to SGD for better DirectML compatibility
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,  # Add momentum for better convergence
        nesterov=True  # Use Nesterov momentum
    )
    
    # Keep MSE Loss for regression
    criterion = nn.MSELoss()
    
    # Adjust scheduler parameters for SGD
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=7,  # Increased patience for SGD
        verbose=True,
        min_lr=1e-6,
        cooldown=3  # Increased cooldown for SGD
    )
    
    return optimizer, criterion, scheduler

# Add warmup scheduler wrapper
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, initial_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.current_epoch = 0
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.initial_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

# Initialize training components with warmup
initial_lr = 0.01  # Higher initial learning rate for SGD
warmup_epochs = 5
optimizer, criterion, scheduler = setup_training(model, learning_rate=initial_lr)
warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, initial_lr)

# Training loop
def train_one_epoch():
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    for points, labels in dataloader:
        if points.size(0) < min_batch_size:
            continue
            
        model.zero_grad(set_to_none=True)
        
        # Add more augmentation
        if random.random() < 0.7:  # Increased probability
            points += torch.randn_like(points) * 0.02  # Increased noise
            # Random rotation around z-axis
            angle = random.random() * 2 * np.pi
            rot_matrix = torch.tensor([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ], device=device, dtype=torch.float32)
            points = torch.matmul(rot_matrix, points)
        
        outputs = model(points)
        # Convert to float for regression
        labels = labels.float()
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.squeeze().detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    mae = np.abs(predictions - targets).mean()
    
    return total_loss / len(dataloader), predictions, targets, mae

# Training with early stopping
max_epochs = 300  # More epochs
early_stopping = EarlyStopping(patience=15, min_delta=0.0005)  # More patience
warmup_epochs = 10  # Longer warmup
warmup_lr_multiplier = 0.01  # Gentler warmup

# Modify the training loop
def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('training_outputs', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    best_mae = float('inf')
    patience_counter = 0
    max_patience = 20
    visualizer = TrainingVisualizer(save_dir=output_dir)
    
    try:
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            all_predictions = []
            all_targets = []
            
            # Apply warmup learning rate
            if epoch < warmup_epochs:
                warmup_scheduler.step()
                current_lr = warmup_scheduler.get_lr()
                print(f"\nWarmup epoch {epoch+1}, LR = {current_lr:.6f}")
            
            for batch_idx, (points, labels) in enumerate(dataloader):
                if points.size(0) < min_batch_size:
                    continue
                
                # Move data to device
                points = points.to(device)
                labels = labels.to(device).float()
                
                # Zero gradients
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                outputs = model(points)
                loss = criterion(outputs.squeeze(), labels)
                
                # Backward pass
                loss.backward()
                # Adjust gradient clipping for SGD
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                
                # Record batch results
                running_loss += loss.item()
                all_predictions.extend(outputs.squeeze().detach().cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                # Print progress every 5 batches
                if (batch_idx + 1) % 5 == 0:
                    print(f"Epoch [{epoch+1}/{max_epochs}] "
                          f"Batch [{batch_idx+1}/{len(dataloader)}] "
                          f"Loss: {loss.item():.4f}")
            
            # Calculate epoch metrics
            avg_loss = running_loss / len(dataloader)
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            mae = np.abs(predictions - targets).mean()
            
            # Update learning rate
            scheduler.step(mae)
            
            # Update visualization
            visualizer.update(epoch + 1, avg_loss, predictions, targets, mae)
            
            # Save best model and check early stopping
            if mae < best_mae:
                best_mae = mae
                patience_counter = 0
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mae': mae,
                }, 'best_model.pth')
                print(f"\nNew best model saved! MAE: {mae:.2f}")
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining error: {e}")
    finally:
        visualizer.close()
        # Create confusion matrix for best model
        plot_confusion_matrix(predictions, targets, log_counts, save_dir=output_dir)
        # Create training video
        create_training_video(output_dir)
        print(f"\nTraining completed. Best MAE: {best_mae:.2f}")
        print(f"Training outputs saved to: {output_dir}")

def main():
    try:
        print("\nStarting training...")
        train()
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        plt.ioff()
        plt.close('all')

if __name__ == "__main__":
    main()
