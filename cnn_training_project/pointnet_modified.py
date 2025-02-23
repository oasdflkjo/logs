import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class PointNet(nn.Module):
    def __init__(self, num_classes=20):
        super(PointNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Initial feature extraction
        self.input_transform = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.Sigmoid()
        )
        
        # Deep feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.ReLU()  # Ensure non-negative output
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial transform
        x = self.input_transform(x)
        
        # Apply attention
        attention = self.attention(x)
        x = x * attention
        
        # Extract features
        x = self.feature_extraction(x)
        
        # Global feature pooling
        x_max = torch.max(x, 2)[0]
        x_avg = torch.mean(x, 2)
        x = x_max + x_avg
        
        # Get classification and regression outputs
        classification = self.classification_head(x)
        regression = self.regression_head(x)
        
        return classification, regression

    def get_loss(self, classification, regression, targets):
        # Adjust targets to 0-19 range (since they're originally 1-20)
        adjusted_targets = targets - 1
        
        # Focal loss for classification
        focal_loss = FocalLoss(gamma=2)(classification, adjusted_targets)
        
        # MSE loss for regression (using original 1-20 range)
        regression_loss = F.mse_loss(regression.squeeze(), targets.float())
        
        # Combine losses with weights
        weights = torch.linspace(1, 2, self.num_classes, device=targets.device)
        target_weights = weights[adjusted_targets]
        
        total_loss = (focal_loss * target_weights).mean() + (regression_loss).mean()
        return total_loss

# Optional: Add STN if needed
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=x.dtype, device=x.device)
        x = x.view(-1, self.k, self.k) + iden
        return x 