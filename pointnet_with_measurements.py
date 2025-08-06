#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./pointnet.pytorch')
from pointnet.model import STN3d, STNkd, feature_transform_regularizer

class PointNetWithMeasurements(nn.Module):
    """
    Modified PointNet that combines point cloud features with anatomical measurements
    """
    def __init__(self, k=2, num_measurements=3, feature_transform=False):
        super(PointNetWithMeasurements, self).__init__()
        self.feature_transform = feature_transform
        self.k = k
        
        # Original PointNet feature extraction
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # Measurement processing branch
        self.meas_fc1 = nn.Linear(num_measurements, 32)
        self.meas_fc2 = nn.Linear(32, 64)
        self.meas_bn1 = nn.BatchNorm1d(32)
        self.meas_bn2 = nn.BatchNorm1d(64)
        self.meas_dropout = nn.Dropout(p=0.3)
        
        # Combined feature processing (Late Fusion)
        # 1024 from PointNet + 64 from measurements = 1088
        self.fc1 = nn.Linear(1088, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x, measurements):
        """
        Args:
            x: point cloud data (B, 3, N) where B=batch, N=num_points
            measurements: anatomical measurements (B, 3) 
                         [left_subclavian_diameter, aortic_arch_diameter, angle]
        """
        n_pts = x.size()[2]
        
        # PointNet branch
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)  # Global point cloud feature
        
        # Measurement branch
        meas = F.relu(self.meas_bn1(self.meas_fc1(measurements)))
        meas = self.meas_dropout(meas)
        meas = F.relu(self.meas_bn2(self.meas_fc2(meas)))
        
        # Combine features (Late Fusion)
        combined = torch.cat([x, meas], dim=1)
        
        # Final classification
        x = F.relu(self.bn4(self.fc1(combined)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetWithMeasurementsEarlyFusion(nn.Module):
    """
    Alternative: Early fusion - concatenate measurements to each point
    """
    def __init__(self, k=2, num_measurements=3, feature_transform=False):
        super(PointNetWithMeasurementsEarlyFusion, self).__init__()
        self.feature_transform = feature_transform
        self.k = k
        
        # Modified first layer to accept 6 channels (3 coords + 3 measurements)
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(6, 64, 1)  # Changed from 3 to 6
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # Standard classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x, measurements):
        """
        Args:
            x: point cloud data (B, 3, N)
            measurements: anatomical measurements (B, 3)
        """
        batch_size, _, n_pts = x.size()
        
        # Apply STN to original coordinates only
        trans = self.stn(x)
        x_transformed = x.transpose(2, 1)
        x_transformed = torch.bmm(x_transformed, trans)
        x_transformed = x_transformed.transpose(2, 1)
        
        # Expand measurements to match point cloud size
        meas_expanded = measurements.unsqueeze(2).expand(-1, -1, n_pts)
        
        # Concatenate coordinates with measurements (early fusion)
        x_combined = torch.cat([x_transformed, meas_expanded], dim=1)
        
        # Process through network
        x = F.relu(self.bn1(self.conv1(x_combined)))
        
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Classification
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1), trans, trans_feat


def normalize_measurements(measurements):
    """
    Normalize anatomical measurements to [0, 1] or standardize
    
    Args:
        measurements: tensor of shape (B, 3) containing:
            [:, 0] - left subclavian artery diameter (mm)
            [:, 1] - aortic arch diameter (mm)
            [:, 2] - angle (degrees)
    """
    # Example normalization ranges (adjust based on your data)
    # Typical ranges:
    # Left subclavian diameter: 5-15 mm
    # Aortic arch diameter: 20-40 mm
    # Angle: 30-150 degrees
    
    normalized = measurements.clone()
    
    # Min-max normalization
    normalized[:, 0] = (measurements[:, 0] - 5) / 10    # subclavian diameter
    normalized[:, 1] = (measurements[:, 1] - 20) / 20   # aortic arch diameter
    normalized[:, 2] = (measurements[:, 2] - 30) / 120  # angle
    
    # Clip to [0, 1]
    normalized = torch.clamp(normalized, 0, 1)
    
    return normalized


if __name__ == "__main__":
    # Test the models
    batch_size = 8
    num_points = 1024
    num_classes = 2
    
    # Dummy data
    points = torch.randn(batch_size, 3, num_points)
    measurements = torch.randn(batch_size, 3)  # 3 anatomical measurements
    
    # Test late fusion model
    print("Testing Late Fusion Model:")
    model_late = PointNetWithMeasurements(k=num_classes)
    output, _, _ = model_late(points, measurements)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_late.parameters())}")
    
    # Test early fusion model
    print("\nTesting Early Fusion Model:")
    model_early = PointNetWithMeasurementsEarlyFusion(k=num_classes)
    output, _, _ = model_early(points, measurements)
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model_early.parameters())}")