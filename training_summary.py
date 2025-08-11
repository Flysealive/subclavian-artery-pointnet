#!/usr/bin/env python3
"""
Training Summary - Text-based visualization
"""

import numpy as np

# Data from current training (up to epoch 44)
epochs = list(range(1, 45))
train_acc = [45.3, 32.8, 39.1, 37.5, 43.8, 54.7, 48.4, 54.7, 62.5, 59.4, 
             76.6, 57.8, 71.9, 65.6, 70.3, 70.3, 70.3, 82.8, 76.6, 75.0,
             84.4, 75.0, 81.2, 81.2, 67.2, 76.6, 76.6, 79.7, 84.4, 78.1,
             82.8, 82.8, 78.1, 84.4, 73.4, 73.4, 76.6, 84.4, 73.4, 76.6,
             82.8, 76.6, 79.7, 76.6]

val_acc = [7.1, 7.1, 42.9, 92.9, 92.9, 85.7, 57.1, 57.1, 57.1, 71.4,
           71.4, 71.4, 71.4, 71.4, 71.4, 71.4, 71.4, 64.3, 71.4, 85.7,
           85.7, 85.7, 78.6, 71.4, 71.4, 85.7, 85.7, 85.7, 78.6, 78.6,
           78.6, 78.6, 78.6, 71.4, 64.3, 71.4, 71.4, 71.4, 71.4, 78.6,
           85.7, 78.6, 78.6, 78.6]

bal_acc = [50.0, 50.0, 69.2, 50.0, 50.0, 46.2, 30.8, 30.8, 30.8, 38.5,
           38.5, 38.5, 38.5, 38.5, 38.5, 38.5, 84.6, 34.6, 38.5, 46.2,
           46.2, 46.2, 42.3, 38.5, 38.5, 46.2, 46.2, 46.2, 42.3, 42.3,
           42.3, 42.3, 42.3, 38.5, 34.6, 38.5, 38.5, 38.5, 38.5, 42.3,
           46.2, 42.3, 42.3, 42.3]

def create_ascii_chart(data, title, width=60, height=15):
    """Create ASCII chart for visualization"""
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val > min_val else 1
    
    # Normalize to height
    normalized = [(v - min_val) / range_val * (height - 1) for v in data]
    
    # Create chart
    chart = []
    for h in range(height, -1, -1):
        row = []
        y_val = min_val + (h / height) * range_val
        row.append(f"{y_val:5.1f}│")
        
        for i, v in enumerate(normalized):
            if abs(v - h) < 0.5:
                row.append("●")
            elif v > h:
                row.append("│")
            else:
                row.append(" ")
        chart.append("".join(row))
    
    # Add x-axis
    chart.append("     └" + "─" * len(data))
    chart.append(f"      Epochs (1-{len(data)})")
    
    return "\n".join([f"    {title}", "    " + "=" * (len(data) + 6)] + chart)

print("="*70)
print("   GPU TRAINING PROGRESS VISUALIZATION - EPOCH 44/1000")
print("="*70)
print()

# Training accuracy chart
print(create_ascii_chart(train_acc[-30:], "TRAINING ACCURACY (Last 30 epochs)", height=10))
print()

# Validation balanced accuracy chart
print(create_ascii_chart(bal_acc[-30:], "VALIDATION BALANCED ACCURACY (Last 30 epochs)", height=10))
print()

# Statistics
print("="*70)
print("   CURRENT STATISTICS (Epoch 44)")
print("="*70)

best_bal_acc = max(bal_acc)
best_epoch = bal_acc.index(best_bal_acc) + 1
current_train = train_acc[-1]
current_val = val_acc[-1]
current_bal = bal_acc[-1]

print(f"""
Current Performance:
  • Training Accuracy:     {current_train:.1f}%
  • Validation Accuracy:   {current_val:.1f}%
  • Balanced Accuracy:     {current_bal:.1f}%

Best Performance:
  • Best Balanced Acc:     {best_bal_acc:.1f}% (Epoch {best_epoch})
  • Improvement:           {best_bal_acc - bal_acc[0]:.1f}% from start

Training Dynamics:
  • Average Train Acc:     {np.mean(train_acc):.1f}%
  • Train Acc Std Dev:     {np.std(train_acc):.1f}%
  • Val Acc Variability:   {np.std(val_acc):.1f}%
  
Class Balance Issue:
  • Class 0 (Majority):    Usually 70-90% accurate
  • Class 1 (Minority):    Struggling (0-100% swings)
  • Imbalance Ratio:       83% vs 17%

GPU Performance:
  • Device:                NVIDIA RTX 4060 Ti (8GB)
  • Speed:                 ~5-6 seconds/epoch
  • Estimated Time:        ~1.5 hours for 1000 epochs
  • Early Stop Patience:   100 epochs
""")

# Progress bar
completed = 44
total = 1000
bar_length = 50
filled = int(bar_length * completed / total)
bar = "█" * filled + "░" * (bar_length - filled)
print(f"Progress: [{bar}] {completed}/{total} epochs ({completed/total*100:.1f}%)")

# Trend analysis
recent_trend = np.polyfit(range(len(bal_acc[-10:])), bal_acc[-10:], 1)[0]
if recent_trend > 0.5:
    trend_str = "↑ Improving"
elif recent_trend < -0.5:
    trend_str = "↓ Declining"
else:
    trend_str = "→ Stable"

print(f"\nRecent Trend (Last 10 epochs): {trend_str} ({recent_trend:.2f}% per epoch)")

print("\n" + "="*70)
print("   RECOMMENDATIONS")
print("="*70)
print("""
1. Class Imbalance: Model struggles with minority class (Class 1)
   → Consider: Oversampling, SMOTE, or adjusting class weights
   
2. Best Model: Saved at Epoch 17 with 84.6% balanced accuracy
   → Current training hasn't exceeded this yet
   
3. Training continues on GPU - monitoring for improvements
   → Will automatically stop if no improvement for 100 epochs
""")

print("\n" + "="*70)