import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, num_classes=21, feature_transform=False, global_feat=True):
        super(PointNet, self).__init__()
        self.num_classes = num_classes
        self.feature_transform = feature_transform
        self.global_feat = global_feat
        
        # Feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.in1 = nn.InstanceNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.in2 = nn.InstanceNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.in3 = nn.InstanceNorm1d(256)
        
        # Global feature learning
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.in4 = nn.InstanceNorm1d(512)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.in5 = nn.InstanceNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.in6 = nn.InstanceNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(128, num_classes)

        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # Feature extraction
        x = F.relu(self.in1(self.conv1(x)))
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))
        
        # Global feature learning
        x = F.relu(self.in4(self.conv4(x)))
        
        # Global max pooling
        x = torch.max(x, 2)[0]
        
        if not self.global_feat:
            return x
        
        # Fully connected layers
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x

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