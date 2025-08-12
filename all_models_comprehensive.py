#!/usr/bin/env python3
"""
Comprehensive Model Architectures for Vessel Classification
============================================================
All model variations for complete comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import numpy as np

# ============== 1. PURE POINTNET ==============

class PurePointNet(nn.Module):
    """Pure PointNet without measurements"""
    
    def __init__(self, num_points=2048, num_classes=2):
        super().__init__()
        
        # Transform network
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Classification head
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch, num_points, 3)
        x = x.transpose(1, 2)  # (batch, 3, num_points)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = torch.max(x, 2)[0]  # (batch, 1024)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ============== 2. POINTNET + MEASUREMENTS ==============

class PointNetWithMeasurements(nn.Module):
    """PointNet with anatomical measurements"""
    
    def __init__(self, num_points=2048, measurement_dim=10, num_classes=2):
        super().__init__()
        
        # PointNet branch
        self.pointnet = PurePointNet(num_points, num_classes)
        
        # Measurement branch
        self.meas_fc1 = nn.Linear(measurement_dim, 64)
        self.meas_fc2 = nn.Linear(64, 128)
        self.meas_bn1 = nn.BatchNorm1d(64)
        self.meas_bn2 = nn.BatchNorm1d(128)
        
        # Combined classifier
        self.combined_fc = nn.Linear(1024 + 128, 512)
        self.final_fc1 = nn.Linear(512, 256)
        self.final_fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, point_cloud, measurements):
        # Get PointNet features (before final classification)
        x = point_cloud.transpose(1, 2)
        x = F.relu(self.pointnet.bn1(self.pointnet.conv1(x)))
        x = F.relu(self.pointnet.bn2(self.pointnet.conv2(x)))
        x = F.relu(self.pointnet.bn3(self.pointnet.conv3(x)))
        pointnet_features = torch.max(x, 2)[0]
        
        # Process measurements
        meas = F.relu(self.meas_bn1(self.meas_fc1(measurements)))
        meas = F.relu(self.meas_bn2(self.meas_fc2(meas)))
        
        # Combine features
        combined = torch.cat([pointnet_features, meas], dim=1)
        
        # Final classification
        x = F.relu(self.combined_fc(combined))
        x = self.dropout(x)
        x = F.relu(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)
        
        return x


# ============== 3. PURE MESHCNN ==============

class PureMeshCNN(nn.Module):
    """Pure MeshCNN without measurements"""
    
    def __init__(self, edge_feat_dim=5, num_classes=2):
        super().__init__()
        
        # Edge convolutions
        self.conv1 = nn.Conv1d(edge_feat_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Global pooling
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classification head
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, edge_features):
        # edge_features: (batch, num_edges, feat_dim)
        x = edge_features.transpose(1, 2)  # (batch, feat_dim, num_edges)
        
        # Convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ============== 4. MESHCNN + MEASUREMENTS ==============

class MeshCNNWithMeasurements(nn.Module):
    """MeshCNN with anatomical measurements"""
    
    def __init__(self, edge_feat_dim=5, measurement_dim=10, num_classes=2):
        super().__init__()
        
        # MeshCNN branch
        self.meshcnn = PureMeshCNN(edge_feat_dim, num_classes)
        
        # Measurement branch
        self.meas_fc1 = nn.Linear(measurement_dim, 64)
        self.meas_fc2 = nn.Linear(64, 128)
        self.meas_bn1 = nn.BatchNorm1d(64)
        self.meas_bn2 = nn.BatchNorm1d(128)
        
        # Combined classifier
        self.combined_fc = nn.Linear(512 + 128, 384)
        self.final_fc1 = nn.Linear(384, 192)
        self.final_fc2 = nn.Linear(192, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, edge_features, measurements):
        # Get MeshCNN features (before final classification)
        x = edge_features.transpose(1, 2)
        x = F.relu(self.meshcnn.bn1(self.meshcnn.conv1(x)))
        x = F.relu(self.meshcnn.bn2(self.meshcnn.conv2(x)))
        x = F.relu(self.meshcnn.bn3(self.meshcnn.conv3(x)))
        x = F.relu(self.meshcnn.bn4(self.meshcnn.conv4(x)))
        mesh_features = self.meshcnn.global_pool(x).squeeze(-1)
        
        # Process measurements
        meas = F.relu(self.meas_bn1(self.meas_fc1(measurements)))
        meas = F.relu(self.meas_bn2(self.meas_fc2(meas)))
        
        # Combine features
        combined = torch.cat([mesh_features, meas], dim=1)
        
        # Final classification
        x = F.relu(self.combined_fc(combined))
        x = self.dropout(x)
        x = F.relu(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)
        
        return x


# ============== 5. PURE GNN ==============

class PureGNN(nn.Module):
    """Pure Graph Neural Network without measurements"""
    
    def __init__(self, node_feat_dim=6, num_classes=2):
        super().__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(node_feat_dim, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 256)
        self.conv4 = GCNConv(256, 512)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Classification head
        self.fc1 = nn.Linear(512 * 2, 256)  # *2 for mean+max pool
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, node_features, edge_index, batch):
        # Graph convolutions
        x = F.relu(self.bn1(self.conv1(node_features, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.bn3(self.conv3(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.bn4(self.conv4(x, edge_index)))
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


# ============== 6. GNN + MEASUREMENTS ==============

class GNNWithMeasurements(nn.Module):
    """GNN with anatomical measurements"""
    
    def __init__(self, node_feat_dim=6, measurement_dim=10, num_classes=2):
        super().__init__()
        
        # GNN branch
        self.gnn = PureGNN(node_feat_dim, num_classes)
        
        # Measurement branch
        self.meas_fc1 = nn.Linear(measurement_dim, 64)
        self.meas_fc2 = nn.Linear(64, 128)
        self.meas_bn1 = nn.BatchNorm1d(64)
        self.meas_bn2 = nn.BatchNorm1d(128)
        
        # Combined classifier
        self.combined_fc = nn.Linear(512 * 2 + 128, 512)
        self.final_fc1 = nn.Linear(512, 256)
        self.final_fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, node_features, edge_index, batch, measurements):
        # Get GNN features (before final classification)
        x = F.relu(self.gnn.bn1(self.gnn.conv1(node_features, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gnn.bn2(self.gnn.conv2(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gnn.bn3(self.gnn.conv3(x, edge_index)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gnn.bn4(self.gnn.conv4(x, edge_index)))
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        gnn_features = torch.cat([x_mean, x_max], dim=1)
        
        # Process measurements
        meas = F.relu(self.meas_bn1(self.meas_fc1(measurements)))
        meas = F.relu(self.meas_bn2(self.meas_fc2(meas)))
        
        # Combine features
        combined = torch.cat([gnn_features, meas], dim=1)
        
        # Final classification
        x = F.relu(self.combined_fc(combined))
        x = self.dropout(x)
        x = F.relu(self.final_fc1(x))
        x = self.dropout(x)
        x = self.final_fc2(x)
        
        return x


# ============== 7. MESHCNN/GNN HYBRID (Already exists) ==============
# Using the existing MeshGNNHybrid from meshcnn_gnn_hybrid.py


# ============== 8. MESHCNN/GNN + VOXEL + MEASUREMENTS ==============

class UltraHybrid(nn.Module):
    """
    Ultimate hybrid: MeshCNN + GNN + Voxel CNN + Measurements
    This combines ALL modalities for maximum performance
    """
    
    def __init__(self, edge_feat_dim=5, node_feat_dim=6, 
                 voxel_size=32, measurement_dim=10, num_classes=2):
        super().__init__()
        
        # MeshCNN branch
        self.mesh_conv1 = nn.Conv1d(edge_feat_dim, 64, 1)
        self.mesh_conv2 = nn.Conv1d(64, 128, 1)
        self.mesh_bn1 = nn.BatchNorm1d(64)
        self.mesh_bn2 = nn.BatchNorm1d(128)
        self.mesh_pool = nn.AdaptiveMaxPool1d(1)
        
        # GNN branch
        self.gnn_conv1 = GCNConv(node_feat_dim, 64)
        self.gnn_conv2 = GCNConv(64, 128)
        self.gnn_bn1 = nn.BatchNorm1d(64)
        self.gnn_bn2 = nn.BatchNorm1d(128)
        
        # Voxel CNN branch
        self.voxel_conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.voxel_conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.voxel_pool = nn.MaxPool3d(2)
        self.voxel_conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.voxel_global_pool = nn.AdaptiveMaxPool3d(1)
        
        # Measurement branch
        self.meas_fc1 = nn.Linear(measurement_dim, 64)
        self.meas_fc2 = nn.Linear(64, 128)
        self.meas_bn1 = nn.BatchNorm1d(64)
        self.meas_bn2 = nn.BatchNorm1d(128)
        
        # Attention mechanism for feature fusion
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 512),  # 4 branches
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, edge_features, node_features, edge_index, 
                batch, voxel_grid, measurements):
        
        # MeshCNN branch
        mesh_x = edge_features.transpose(1, 2)
        mesh_x = F.relu(self.mesh_bn1(self.mesh_conv1(mesh_x)))
        mesh_x = F.relu(self.mesh_bn2(self.mesh_conv2(mesh_x)))
        mesh_feat = self.mesh_pool(mesh_x).squeeze(-1)
        
        # GNN branch
        gnn_x = F.relu(self.gnn_bn1(self.gnn_conv1(node_features, edge_index)))
        gnn_x = F.relu(self.gnn_bn2(self.gnn_conv2(gnn_x, edge_index)))
        gnn_mean = global_mean_pool(gnn_x, batch)
        gnn_max = global_max_pool(gnn_x, batch)
        gnn_feat = (gnn_mean + gnn_max) / 2
        
        # Voxel CNN branch
        if len(voxel_grid.shape) == 4:
            voxel_grid = voxel_grid.unsqueeze(1)
        voxel_x = F.relu(self.voxel_conv1(voxel_grid))
        voxel_x = self.voxel_pool(voxel_x)
        voxel_x = F.relu(self.voxel_conv2(voxel_x))
        voxel_x = self.voxel_pool(voxel_x)
        voxel_x = F.relu(self.voxel_conv3(voxel_x))
        voxel_feat = self.voxel_global_pool(voxel_x).squeeze(-1).squeeze(-1).squeeze(-1)
        
        # Measurement branch
        meas_feat = F.relu(self.meas_bn1(self.meas_fc1(measurements)))
        meas_feat = F.relu(self.meas_bn2(self.meas_fc2(meas_feat)))
        
        # Stack features for attention
        features = torch.stack([mesh_feat, gnn_feat, voxel_feat, meas_feat], dim=0)
        
        # Apply attention mechanism
        attended_features, _ = self.attention(features, features, features)
        
        # Concatenate all features
        combined = torch.cat([mesh_feat, gnn_feat, voxel_feat, meas_feat], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output


# ============== MODEL SUMMARY ==============

def get_all_models():
    """Return dictionary of all model architectures"""
    return {
        # Pure models (no measurements)
        'Pure_PointNet': PurePointNet(),
        'Pure_MeshCNN': PureMeshCNN(),
        'Pure_GNN': PureGNN(),
        
        # Models with measurements
        'PointNet_Meas': PointNetWithMeasurements(),
        'MeshCNN_Meas': MeshCNNWithMeasurements(),
        'GNN_Meas': GNNWithMeasurements(),
        
        # Hybrid models (from other files)
        # 'MeshGNN_Hybrid': loaded from meshcnn_gnn_hybrid.py
        # 'PointNet_Voxel_Meas': loaded from hybrid_multimodal_model.py
        
        # Ultimate hybrid
        'Ultra_Hybrid': UltraHybrid()
    }


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("="*60)
    print("ALL MODEL ARCHITECTURES FOR VESSEL CLASSIFICATION")
    print("="*60)
    
    models = get_all_models()
    
    print("\nModel Parameter Counts:")
    print("-"*40)
    for name, model in models.items():
        params = count_parameters(model)
        print(f"{name:20s}: {params:,} parameters")
    
    print("\n" + "="*60)
    print("MODEL CATEGORIES:")
    print("="*60)
    
    print("\n1. PURE MODELS (No Measurements):")
    print("   - Pure PointNet")
    print("   - Pure MeshCNN")
    print("   - Pure GNN")
    
    print("\n2. MODELS WITH MEASUREMENTS:")
    print("   - PointNet + Measurements")
    print("   - MeshCNN + Measurements")
    print("   - GNN + Measurements")
    
    print("\n3. HYBRID MODELS:")
    print("   - MeshCNN/GNN + Measurements (from meshcnn_gnn_hybrid.py)")
    print("   - PointNet + Voxel + Measurements (your 96.2% model)")
    print("   - Ultra Hybrid (MeshCNN + GNN + Voxel + Measurements)")
    
    print("\nReady for comprehensive training and comparison!")