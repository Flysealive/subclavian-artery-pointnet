"""
Real-time Training Monitor
==========================
Monitor the training progress of both models
"""

import json
import os
import time
from datetime import datetime

def monitor_training():
    print("\n" + "="*70)
    print("TRAINING MONITOR - VOXEL CNN COMPARISON")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nMonitoring training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_update = ""
    
    while True:
        try:
            # Check for model files
            models = {
                'voxel_only': os.path.exists('best_voxel_only.pth'),
                'voxel_with_measurements': os.path.exists('best_voxel_with_measurements.pth')
            }
            
            # Check for result files
            results_exist = os.path.exists('comparison_results.json')
            
            status = []
            
            if models['voxel_only']:
                status.append("[✓] Voxel-only model trained")
            else:
                status.append("[◯] Training voxel-only model...")
            
            if models['voxel_with_measurements']:
                status.append("[✓] Voxel+measurements model trained")
            elif models['voxel_only']:
                status.append("[◯] Training voxel+measurements model...")
            else:
                status.append("[◯] Waiting to train voxel+measurements model")
            
            if results_exist:
                status.append("[✓] Comparison complete!")
                
                # Load and display results
                with open('comparison_results.json', 'r') as f:
                    results = json.load(f)
                
                print("\n" + "="*70)
                print("TRAINING COMPLETE - FINAL RESULTS")
                print("="*70)
                
                print("\n📊 VOXEL-ONLY MODEL:")
                print(f"  • Best Accuracy: {results['voxel_only']['best_accuracy']:.4f}")
                print(f"  • Best AUC: {results['voxel_only']['best_auc']:.4f}")
                print(f"  • Training Time: {results['voxel_only']['training_time_minutes']:.1f} minutes")
                
                print("\n📊 VOXEL + MEASUREMENTS MODEL:")
                print(f"  • Best Accuracy: {results['voxel_with_measurements']['best_accuracy']:.4f}")
                print(f"  • Best AUC: {results['voxel_with_measurements']['best_auc']:.4f}")
                print(f"  • Training Time: {results['voxel_with_measurements']['training_time_minutes']:.1f} minutes")
                
                print("\n🎯 IMPROVEMENTS WITH MEASUREMENTS:")
                print(f"  • Accuracy Improvement: {results['improvements']['accuracy_improvement_percent']:+.1f}%")
                print(f"  • AUC Improvement: {results['improvements']['auc_improvement_percent']:+.1f}%")
                
                print("\n✅ All results saved to:")
                print("  • model_comparison_results.png")
                print("  • comparison_results.json")
                print("  • best_voxel_only.pth")
                print("  • best_voxel_with_measurements.pth")
                
                break
            
            # Update status display
            current_status = "\r" + " | ".join(status)
            if current_status != last_update:
                print(current_status, end="", flush=True)
                last_update = current_status
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training()