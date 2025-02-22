import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PointCloudDataset
from pointnet_modified import PointNet
import torch_directml
import time
from collections import defaultdict

# Setup
device = torch_directml.device()
print(f"Using device: {device}")

# Load data
dataset = PointCloudDataset("C:/output", device=device)

# Print dataset statistics
log_counts = dataset.get_log_counts()
print("\nDataset Statistics:")
print(f"Number of different log counts: {len(log_counts)}")
print(f"Log counts available: {log_counts}")

# Create data loader with batch size equal to samples per log count
# This ensures each batch contains different log counts
batch_size = len(dataset) // len(log_counts)  # Should be around 50
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model
model = PointNet(num_classes=21).to(device)

# Loss and optimizer with better learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    model.parameters(),
    lr=0.001,  # Reduced initial learning rate
    momentum=0.9,
    weight_decay=0.0001  # Added weight decay
)

# Better learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',
    factor=0.5,
    patience=2,
    verbose=True
)

# Training loop
num_epochs = 50
print("\nStarting training...")
start_time = time.time()
best_accuracy = 0

for epoch in range(num_epochs):
    model.train()
    losses = []
    correct_by_count = defaultdict(int)
    total_by_count = defaultdict(int)
    epoch_start = time.time()
    
    for batch_idx, (points, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Track loss
        losses.append(loss.item())
        
        # Track accuracy by log count
        pred = outputs.argmax(1)
        for p, t in zip(pred, labels):
            correct_by_count[t.item()] += (p == t).item()
            total_by_count[t.item()] += 1
            
        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f}")
    
    # Calculate epoch time
    epoch_time = time.time() - epoch_start
    
    # Calculate overall accuracy
    total_correct = sum(correct_by_count.values())
    total_samples = sum(total_by_count.values())
    overall_accuracy = 100 * total_correct / total_samples
    avg_loss = sum(losses) / len(losses)
    
    # Print accuracy for each log count
    print(f"\nEpoch [{epoch+1}/{num_epochs}] Stats:")
    print(f"Time: {epoch_time:.2f}s")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% (Best: {best_accuracy:.2f}%)")
    print("\nAccuracy by log count:")
    for count in sorted(total_by_count.keys()):
        acc = 100 * correct_by_count[count] / total_by_count[count]
        print(f"{count} logs: {acc:.1f}%")
    
    # Update scheduler based on overall accuracy
    scheduler.step(overall_accuracy)
    
    # Track best accuracy
    if overall_accuracy > best_accuracy:
        best_accuracy = overall_accuracy
        # Save best model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': overall_accuracy,
        }, 'best_model.pth')
        print(f"\nNew best model saved! Accuracy: {overall_accuracy:.2f}%")
    
    print(f"\nLearning Rate: {optimizer.param_groups[0]['lr']:.6f}")

total_time = time.time() - start_time
print(f"\nTraining complete! Total time: {total_time/60:.2f} minutes")
print(f"Best accuracy achieved: {best_accuracy:.2f}%")
