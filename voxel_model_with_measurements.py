"""
Enhanced Voxel CNN Models with Anatomical Measurements
=======================================================
Combines 3D voxel features with anatomical measurements for improved classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelCNNWithMeasurements(nn.Module):
    """
    3D CNN that combines voxel features with anatomical measurements
    """
    def __init__(self, num_classes=2, voxel_size=64, num_measurements=3):
        super(VoxelCNNWithMeasurements, self).__init__()
        
        # Voxel processing branch (3D CNN)
        self.conv1 = nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm3d(512)
        
        self.pool = nn.MaxPool3d(2)
        self.dropout3d = nn.Dropout3d(0.3)
        
        # Calculate flattened size for voxel features
        feat_size = voxel_size // 16
        voxel_feat_dim = 512 * feat_size * feat_size * feat_size
        
        # Voxel feature processing
        self.voxel_fc = nn.Linear(voxel_feat_dim, 256)
        self.voxel_bn = nn.BatchNorm1d(256)
        
        # Measurement processing branch
        self.measure_fc1 = nn.Linear(num_measurements, 32)
        self.measure_bn1 = nn.BatchNorm1d(32)
        self.measure_fc2 = nn.Linear(32, 64)
        self.measure_bn2 = nn.BatchNorm1d(64)
        
        # Combined feature processing
        self.combined_fc1 = nn.Linear(256 + 64, 256)  # Voxel features + measurement features
        self.combined_bn1 = nn.BatchNorm1d(256)
        self.combined_fc2 = nn.Linear(256, 128)
        self.combined_bn2 = nn.BatchNorm1d(128)
        self.combined_fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, voxel, measurements):
        # Process voxel data
        x = voxel.unsqueeze(1) if len(voxel.shape) == 4 else voxel
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout3d(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten voxel features
        x = x.view(x.size(0), -1)
        voxel_features = F.relu(self.voxel_bn(self.voxel_fc(x)))
        voxel_features = self.dropout(voxel_features)
        
        # Process measurements
        m = F.relu(self.measure_bn1(self.measure_fc1(measurements)))
        m = self.dropout(m)
        m = F.relu(self.measure_bn2(self.measure_fc2(m)))
        measurement_features = self.dropout(m)
        
        # Combine features
        combined = torch.cat([voxel_features, measurement_features], dim=1)
        
        # Final classification
        combined = F.relu(self.combined_bn1(self.combined_fc1(combined)))
        combined = self.dropout(combined)
        combined = F.relu(self.combined_bn2(self.combined_fc2(combined)))
        combined = self.dropout(combined)
        output = self.combined_fc3(combined)
        
        return output


class AttentionFusionVoxelCNN(nn.Module):
    """
    Advanced model using attention mechanism to fuse voxel and measurement features
    """
    def __init__(self, num_classes=2, voxel_size=64, num_measurements=3):
        super(AttentionFusionVoxelCNN, self).__init__()
        
        # Voxel feature extractor
        self.voxel_encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Measurement encoder
        self.measurement_encoder = nn.Sequential(
            nn.Linear(num_measurements, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Attention mechanism for feature fusion
        self.attention = nn.Sequential(
            nn.Linear(512 + 256, 256),
            nn.Tanh(),
            nn.Linear(256, 512 + 256),
            nn.Softmax(dim=1)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, voxel, measurements):
        # Extract voxel features
        voxel = voxel.unsqueeze(1) if len(voxel.shape) == 4 else voxel
        voxel_features = self.voxel_encoder(voxel)
        voxel_features = voxel_features.view(voxel_features.size(0), -1)
        
        # Extract measurement features
        measurement_features = self.measurement_encoder(measurements)
        
        # Concatenate features
        combined_features = torch.cat([voxel_features, measurement_features], dim=1)
        
        # Apply attention weights
        attention_weights = self.attention(combined_features)
        weighted_features = combined_features * attention_weights
        
        # Classification
        output = self.classifier(weighted_features)
        
        return output


class HybridVoxelMeasurementNet(nn.Module):
    """
    Hybrid network with separate pathways and late fusion
    """
    def __init__(self, num_classes=2, voxel_size=64, num_measurements=3):
        super(HybridVoxelMeasurementNet, self).__init__()
        
        # Voxel pathway (deeper)
        self.voxel_path = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )
        
        # Measurement pathway
        self.measurement_path = nn.Sequential(
            nn.Linear(num_measurements, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Fusion and classification
        voxel_feat_dim = 256 * 2 * 2 * 2  # 2048
        combined_dim = voxel_feat_dim + 128
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, voxel, measurements):
        # Process voxel
        voxel = voxel.unsqueeze(1) if len(voxel.shape) == 4 else voxel
        voxel_features = self.voxel_path(voxel)
        voxel_features = voxel_features.view(voxel_features.size(0), -1)
        
        # Process measurements
        measurement_features = self.measurement_path(measurements)
        
        # Combine and classify
        combined = torch.cat([voxel_features, measurement_features], dim=1)
        output = self.fusion(combined)
        
        return output