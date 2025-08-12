"""
Hybrid Multi-Modal Model: Point Cloud + Voxel + Measurements
==============================================================
Combines three data modalities for improved classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
import os
from tqdm import tqdm

# ============== PART 1: DATA PREPARATION ==============

def stl_to_pointcloud(stl_path, num_points=2048):
    """Extract point cloud from STL file"""
    mesh = trimesh.load(stl_path)
    
    # Sample points uniformly from the mesh surface
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    
    # Normalize to unit sphere
    centroid = points.mean(axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_dist
    
    return points.astype(np.float32)

def stl_to_voxel_fixed(stl_path, voxel_size=32):
    """Extract voxel grid from STL (smaller size for fusion)"""
    mesh = trimesh.load(stl_path)
    
    if not mesh.is_watertight:
        mesh.fill_holes()
    
    # Normalize mesh
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2
    scale = (bounds[1] - bounds[0]).max()
    
    mesh.vertices = (mesh.vertices - center) / scale * (voxel_size - 1)
    mesh.vertices += voxel_size / 2
    
    # Voxelize
    voxel_grid = mesh.voxelized(pitch=1.0).matrix
    
    # Ensure fixed size
    output = np.zeros((voxel_size, voxel_size, voxel_size), dtype=np.float32)
    
    min_shape = tuple(min(s, voxel_size) for s in voxel_grid.shape)
    output[:min_shape[0], :min_shape[1], :min_shape[2]] = \
        voxel_grid[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    return output

def prepare_hybrid_data(stl_dir='STL', output_dir='hybrid_data', 
                        num_points=2048, voxel_size=32):
    """Prepare both point cloud and voxel data"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pointclouds'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'voxels'), exist_ok=True)
    
    stl_files = [f for f in os.listdir(stl_dir) if f.endswith('.stl')]
    
    print(f"Processing {len(stl_files)} STL files...")
    
    for stl_file in tqdm(stl_files):
        stl_path = os.path.join(stl_dir, stl_file)
        base_name = stl_file.replace('.stl', '')
        
        try:
            # Extract point cloud
            points = stl_to_pointcloud(stl_path, num_points)
            pc_path = os.path.join(output_dir, 'pointclouds', f'{base_name}.npy')
            np.save(pc_path, points)
            
            # Extract voxels
            voxels = stl_to_voxel_fixed(stl_path, voxel_size)
            vox_path = os.path.join(output_dir, 'voxels', f'{base_name}.npy')
            np.save(vox_path, voxels)
            
        except Exception as e:
            print(f"Error processing {stl_file}: {e}")
    
    print(f"Hybrid data saved to {output_dir}/")

# ============== PART 2: HYBRID MODEL ARCHITECTURE ==============

class PointNetEncoder(nn.Module):
    """PointNet encoder for point cloud features"""
    def __init__(self, input_dim=3, output_dim=256):
        super(PointNetEncoder, self).__init__()
        
        # Point-wise MLPs
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, output_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(1, 2)
        
        # Point-wise feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=False)[0]  # (B, output_dim)
        
        return x

class VoxelEncoder(nn.Module):
    """3D CNN encoder for voxel features"""
    def __init__(self, voxel_size=32, output_dim=256):
        super(VoxelEncoder, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.pool = nn.MaxPool3d(2)
        
        # Calculate flattened size
        feat_size = voxel_size // 8  # 3 pooling layers
        self.fc = nn.Linear(128 * feat_size**3, output_dim)
        self.fc_bn = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # x: (B, 32, 32, 32) -> (B, 1, 32, 32, 32)
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_bn(self.fc(x)))
        
        return x

class HybridMultiModalNet(nn.Module):
    """
    Combines Point Cloud + Voxel + Measurements
    Uses attention mechanism for adaptive feature fusion
    """
    def __init__(self, num_classes=2, num_points=2048, voxel_size=32, num_measurements=3):
        super(HybridMultiModalNet, self).__init__()
        
        # Encoders for each modality
        self.pointnet_encoder = PointNetEncoder(input_dim=3, output_dim=256)
        self.voxel_encoder = VoxelEncoder(voxel_size=voxel_size, output_dim=256)
        
        # Measurement encoder
        self.measure_encoder = nn.Sequential(
            nn.Linear(num_measurements, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        # Feature fusion layers
        total_features = 256 + 256 + 128  # pointnet + voxel + measurements
        
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # Learnable fusion weights
        self.modal_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, pointcloud, voxel, measurements):
        # Extract features from each modality
        pc_features = self.pointnet_encoder(pointcloud)  # (B, 256)
        vox_features = self.voxel_encoder(voxel)        # (B, 256)
        meas_features = self.measure_encoder(measurements)  # (B, 128)
        
        # Apply cross-modal attention between point cloud and voxel features
        # Reshape for attention: (B, 256) -> (1, B, 256)
        pc_attn = pc_features.unsqueeze(0)
        vox_attn = vox_features.unsqueeze(0)
        
        # Cross-attention
        attended_pc, _ = self.attention(pc_attn, vox_attn, vox_attn)
        attended_vox, _ = self.attention(vox_attn, pc_attn, pc_attn)
        
        # Squeeze back: (1, B, 256) -> (B, 256)
        attended_pc = attended_pc.squeeze(0)
        attended_vox = attended_vox.squeeze(0)
        
        # Weighted combination
        weights = F.softmax(self.modal_weights, dim=0)
        pc_weighted = attended_pc * weights[0]
        vox_weighted = attended_vox * weights[1]
        meas_weighted = meas_features * weights[2]
        
        # Concatenate all features
        combined = torch.cat([pc_weighted, vox_weighted, meas_weighted], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output

# ============== PART 3: DATASET LOADER ==============

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class HybridMultiModalDataset(Dataset):
    """Dataset for hybrid point cloud + voxel + measurements"""
    
    def __init__(self, hybrid_dir='hybrid_data', 
                 labels_file='classification_labels_with_measurements.csv',
                 mode='train', test_size=0.2, random_seed=42):
        
        self.pc_dir = os.path.join(hybrid_dir, 'pointclouds')
        self.vox_dir = os.path.join(hybrid_dir, 'voxels')
        
        # Load labels and measurements
        labels_df = pd.read_csv(labels_file)
        labels_df['filename'] = labels_df['filename'].str.replace('.npy', '')
        
        # Measurement columns
        measurement_cols = ['left_subclavian_diameter_mm', 
                          'aortic_arch_diameter_mm', 
                          'angle_degrees']
        
        # Filter for existing files
        self.data = []
        self.labels = []
        self.measurements = []
        
        for _, row in labels_df.iterrows():
            pc_file = os.path.join(self.pc_dir, row['filename'] + '.npy')
            vox_file = os.path.join(self.vox_dir, row['filename'] + '.npy')
            
            if os.path.exists(pc_file) and os.path.exists(vox_file):
                self.data.append(row['filename'])
                self.labels.append(row['label'])
                self.measurements.append(row[measurement_cols].values)
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.measurements = np.array(self.measurements, dtype=np.float32)
        
        # Normalize measurements
        self.scaler = StandardScaler()
        self.measurements = self.scaler.fit_transform(self.measurements)
        
        # Split data
        X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
            self.data, self.labels, self.measurements,
            test_size=test_size, random_state=random_seed, 
            stratify=self.labels
        )
        
        if mode == 'train':
            self.data = X_train
            self.labels = y_train
            self.measurements = m_train
        else:
            self.data = X_test
            self.labels = y_test
            self.measurements = m_test
        
        print(f"{mode} dataset: {len(self.data)} samples")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        filename = self.data[idx]
        
        # Load point cloud
        pc = np.load(os.path.join(self.pc_dir, filename + '.npy'))
        
        # Load voxel
        vox = np.load(os.path.join(self.vox_dir, filename + '.npy'))
        
        # Get label and measurements
        label = self.labels[idx]
        measurements = self.measurements[idx]
        
        return (torch.FloatTensor(pc),
                torch.FloatTensor(vox),
                torch.FloatTensor(measurements),
                torch.LongTensor([label])[0])

# ============== PART 4: TRAINING SCRIPT ==============

def train_hybrid_model(epochs=50, batch_size=8, lr=0.001):
    """Train the hybrid multi-modal model"""
    
    print("\n" + "="*70)
    print("HYBRID MULTI-MODAL TRAINING")
    print("Combining: Point Cloud + Voxel + Measurements")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Prepare data if not exists
    if not os.path.exists('hybrid_data'):
        print("\nPreparing hybrid data...")
        prepare_hybrid_data()
    
    # Create datasets
    train_dataset = HybridMultiModalDataset(mode='train')
    test_dataset = HybridMultiModalDataset(mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = HybridMultiModalNet(
        num_classes=2,
        num_points=2048,
        voxel_size=32,
        num_measurements=3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for pc, vox, meas, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pc = pc.to(device)
            vox = vox.to(device)
            meas = meas.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(pc, vox, meas)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for pc, vox, meas, labels in test_loader:
                pc = pc.to(device)
                vox = vox.to(device)
                meas = meas.to(device)
                labels = labels.to(device)
                
                outputs = model(pc, vox, meas)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = val_correct / val_total
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_hybrid_model.pth')
            print(f"  [SAVED] New best model: {best_acc:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"{'='*70}")
    
    return history, best_acc

if __name__ == "__main__":
    # Run training
    history, best_acc = train_hybrid_model(epochs=30, batch_size=8, lr=0.001)