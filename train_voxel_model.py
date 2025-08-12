import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from tqdm import tqdm
import argparse

from voxel_cnn_model import VoxelCNN, VoxelResNet
from voxel_dataset import get_voxel_dataloaders

class VoxelTrainer:
    def __init__(self, model, device, model_name="voxel_cnn"):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
        
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc="Training")):
            data, labels = data.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc="Validation"):
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        metrics = self.calculate_metrics(all_labels, all_preds)
        
        return epoch_loss, metrics, all_labels, all_preds
    
    def calculate_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=20):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                     patience=patience//2, verbose=True)
        
        best_val_acc = 0
        best_epoch = 0
        patience_counter = 0
        
        print(f"Training {self.model_name} for {epochs} epochs...")
        print(f"Device: {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            val_loss, val_metrics, val_labels, val_preds = self.validate(val_loader, criterion)
            
            scheduler.step(val_metrics['accuracy'])
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, "
                  f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch + 1
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': self.history
                }, f'best_{self.model_name}_model.pth')
                
                print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
        
        if epoch == epochs - 1:
            cm = confusion_matrix(val_labels, val_preds)
            self.plot_confusion_matrix(cm)
        
        return self.history
    
    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{self.model_name}_confusion_matrix.png')
        plt.close()

def plot_training_history(history, model_name="voxel_cnn"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(history['val_precision'], label='Precision')
    axes[1, 0].plot(history['val_recall'], label='Recall')
    axes[1, 0].plot(history['val_f1'], label='F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].bar(['Best Train Acc', 'Best Val Acc', 'Best F1'], 
                   [max(history['train_acc']), max(history['val_acc']), max(history['val_f1'])],
                   color=['blue', 'green', 'orange'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Best Scores')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate([max(history['train_acc']), max(history['val_acc']), max(history['val_f1'])]):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.suptitle(f'{model_name} Training History', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Voxel CNN Model')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'resnet'],
                      help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--voxel_size', type=int, default=64, help='Voxel grid size')
    parser.add_argument('--balanced', action='store_true', help='Use balanced dataset')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists('voxel_arrays'):
        print("Voxel arrays not found. Running conversion...")
        from stl_to_voxel import convert_dataset_to_voxels
        convert_dataset_to_voxels(voxel_size=args.voxel_size)
    
    train_loader, test_loader = get_voxel_dataloaders(
        batch_size=args.batch_size,
        balanced=args.balanced
    )
    
    if args.model == 'resnet':
        model = VoxelResNet(num_classes=2, voxel_size=args.voxel_size)
        model_name = "voxel_resnet"
    else:
        model = VoxelCNN(num_classes=2, voxel_size=args.voxel_size)
        model_name = "voxel_cnn"
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = VoxelTrainer(model, device, model_name)
    history = trainer.train(
        train_loader, test_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience
    )
    
    with open(f'{model_name}_history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    plot_training_history(history, model_name)
    
    print("\nTraining complete! Results saved.")

if __name__ == "__main__":
    main()