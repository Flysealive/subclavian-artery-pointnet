"""
Simple Robust Comparison Script
================================
Trains both models with error handling
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import json
import time
import warnings
warnings.filterwarnings('ignore')

print("Starting simple comparison...")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    # Import models and datasets
    from voxel_cnn_model import VoxelCNN
    from voxel_model_with_measurements import VoxelCNNWithMeasurements
    from voxel_dataset import get_voxel_dataloaders
    from voxel_dataset_with_measurements import get_measurement_dataloaders
    
    results = {}
    
    # ============== MODEL 1: VOXEL ONLY ==============
    print("\n" + "="*60)
    print("TRAINING MODEL 1: VOXEL ONLY")
    print("="*60)
    
    try:
        # Load data
        train_loader, test_loader = get_voxel_dataloaders(
            batch_size=8,
            balanced=True,
            num_workers=0
        )
        
        # Create model
        model1 = VoxelCNN(num_classes=2, voxel_size=64).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model1.parameters(), lr=0.001)
        
        print("Training for 20 epochs...")
        best_acc1 = 0
        history1 = {'acc': [], 'loss': []}
        
        for epoch in range(20):
            # Train
            model1.train()
            train_loss = 0
            for batch_idx, (data, labels) in enumerate(train_loader):
                data, labels = data.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model1(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model1.eval()
            correct = 0
            total = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model1(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            acc = correct / total
            avg_loss = train_loss / len(train_loader)
            history1['acc'].append(acc)
            history1['loss'].append(avg_loss)
            
            if acc > best_acc1:
                best_acc1 = acc
                torch.save(model1.state_dict(), 'simple_voxel_only.pth')
            
            print(f"Epoch {epoch+1}/20: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        
        # Final metrics
        f1_1 = f1_score(val_labels, val_preds, average='weighted')
        cm1 = confusion_matrix(val_labels, val_preds)
        
        results['voxel_only'] = {
            'best_acc': best_acc1,
            'f1_score': f1_1,
            'confusion_matrix': cm1.tolist(),
            'history': history1
        }
        
        print(f"\nModel 1 Complete: Best Acc={best_acc1:.4f}, F1={f1_1:.4f}")
        
    except Exception as e:
        print(f"Error training Model 1: {e}")
        results['voxel_only'] = {'error': str(e)}
    
    # ============== MODEL 2: VOXEL + MEASUREMENTS ==============
    print("\n" + "="*60)
    print("TRAINING MODEL 2: VOXEL + MEASUREMENTS")
    print("="*60)
    
    try:
        # Load data with measurements
        train_loader2, test_loader2 = get_measurement_dataloaders(
            batch_size=8,
            balanced=True,
            num_workers=0
        )
        
        # Create model
        model2 = VoxelCNNWithMeasurements(num_classes=2, voxel_size=64, num_measurements=3).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model2.parameters(), lr=0.001)
        
        print("Training for 20 epochs...")
        best_acc2 = 0
        history2 = {'acc': [], 'loss': []}
        
        for epoch in range(20):
            # Train
            model2.train()
            train_loss = 0
            for batch_idx, (voxels, measurements, labels) in enumerate(train_loader2):
                voxels = voxels.to(device)
                measurements = measurements.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model2(voxels, measurements)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model2.eval()
            correct = 0
            total = 0
            val_preds2 = []
            val_labels2 = []
            
            with torch.no_grad():
                for voxels, measurements, labels in test_loader2:
                    voxels = voxels.to(device)
                    measurements = measurements.to(device)
                    labels = labels.to(device)
                    
                    outputs = model2(voxels, measurements)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    val_preds2.extend(predicted.cpu().numpy())
                    val_labels2.extend(labels.cpu().numpy())
            
            acc = correct / total
            avg_loss = train_loss / len(train_loader2)
            history2['acc'].append(acc)
            history2['loss'].append(avg_loss)
            
            if acc > best_acc2:
                best_acc2 = acc
                torch.save(model2.state_dict(), 'simple_voxel_measurements.pth')
            
            print(f"Epoch {epoch+1}/20: Loss={avg_loss:.4f}, Acc={acc:.4f}")
        
        # Final metrics
        f1_2 = f1_score(val_labels2, val_preds2, average='weighted')
        cm2 = confusion_matrix(val_labels2, val_preds2)
        
        results['voxel_with_measurements'] = {
            'best_acc': best_acc2,
            'f1_score': f1_2,
            'confusion_matrix': cm2.tolist(),
            'history': history2
        }
        
        print(f"\nModel 2 Complete: Best Acc={best_acc2:.4f}, F1={f1_2:.4f}")
        
    except Exception as e:
        print(f"Error training Model 2: {e}")
        results['voxel_with_measurements'] = {'error': str(e)}
    
    # ============== COMPARISON ==============
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    if 'voxel_only' in results and 'error' not in results['voxel_only']:
        print(f"\nVOXEL ONLY:")
        print(f"  Best Accuracy: {results['voxel_only']['best_acc']:.4f}")
        print(f"  F1 Score: {results['voxel_only']['f1_score']:.4f}")
    
    if 'voxel_with_measurements' in results and 'error' not in results['voxel_with_measurements']:
        print(f"\nVOXEL + MEASUREMENTS:")
        print(f"  Best Accuracy: {results['voxel_with_measurements']['best_acc']:.4f}")
        print(f"  F1 Score: {results['voxel_with_measurements']['f1_score']:.4f}")
        
        if 'voxel_only' in results and 'error' not in results['voxel_only']:
            acc_imp = ((results['voxel_with_measurements']['best_acc'] - 
                       results['voxel_only']['best_acc']) / 
                      results['voxel_only']['best_acc'] * 100)
            
            print(f"\nIMPROVEMENT WITH MEASUREMENTS:")
            print(f"  Accuracy: {acc_imp:+.1f}%")
    
    # Save results
    with open('simple_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Create simple plot
    if all(k in results and 'error' not in results[k] for k in ['voxel_only', 'voxel_with_measurements']):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Accuracy curves
        axes[0].plot(results['voxel_only']['history']['acc'], label='Voxel Only', linewidth=2)
        axes[0].plot(results['voxel_with_measurements']['history']['acc'], label='Voxel + Measurements', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Validation Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Final comparison
        models = ['Voxel Only', 'Voxel +\nMeasurements']
        accs = [results['voxel_only']['best_acc'], results['voxel_with_measurements']['best_acc']]
        f1s = [results['voxel_only']['f1_score'], results['voxel_with_measurements']['f1_score']]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[1].bar(x - width/2, accs, width, label='Accuracy', color='blue', alpha=0.7)
        axes[1].bar(x + width/2, f1s, width, label='F1 Score', color='green', alpha=0.7)
        axes[1].set_ylabel('Score')
        axes[1].set_title('Final Metrics Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].legend()
        axes[1].set_ylim([0, 1.1])
        
        for i, (acc, f1) in enumerate(zip(accs, f1s)):
            axes[1].text(i - width/2, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
            axes[1].text(i + width/2, f1 + 0.01, f'{f1:.3f}', ha='center', fontsize=9)
        
        # Confusion matrices
        axes[2].axis('off')
        improvement = ((results['voxel_with_measurements']['best_acc'] - 
                       results['voxel_only']['best_acc']) / 
                      results['voxel_only']['best_acc'] * 100)
        
        summary = f"""
        SUMMARY
        
        Voxel Only:
          Accuracy: {results['voxel_only']['best_acc']:.4f}
          F1 Score: {results['voxel_only']['f1_score']:.4f}
        
        Voxel + Measurements:
          Accuracy: {results['voxel_with_measurements']['best_acc']:.4f}
          F1 Score: {results['voxel_with_measurements']['f1_score']:.4f}
        
        Improvement: {improvement:+.1f}%
        """
        
        axes[2].text(0.5, 0.5, summary, transform=axes[2].transAxes,
                    fontsize=12, verticalalignment='center',
                    horizontalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Voxel CNN Comparison: With vs Without Measurements', fontsize=14)
        plt.tight_layout()
        plt.savefig('simple_comparison.png', dpi=150)
        plt.show()
        
        print("\n[SAVED] Results to simple_comparison_results.json and simple_comparison.png")
    
    print("\n[COMPLETE] Comparison finished!")
    
except Exception as e:
    print(f"Fatal error: {e}")
    import traceback
    traceback.print_exc()