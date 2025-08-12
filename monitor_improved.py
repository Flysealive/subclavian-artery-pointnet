"""
Monitor Improved Training Progress
===================================
"""
import os
import time
import json

print("Monitoring improved training...")
print("="*50)

best_file = 'improved_model_best.pth'
start_time = time.time()

while True:
    try:
        # Check if best model exists
        if os.path.exists(best_file):
            file_size = os.path.getsize(best_file) / (1024*1024)  # MB
            mod_time = os.path.getmtime(best_file)
            time_ago = time.time() - mod_time
            
            print(f"\r[{time.strftime('%H:%M:%S')}] Model saved {time_ago:.0f}s ago | Size: {file_size:.1f}MB", end="", flush=True)
            
            if time_ago < 5:
                print("\n✓ New best model saved!")
        else:
            elapsed = time.time() - start_time
            print(f"\r[{time.strftime('%H:%M:%S')}] Training in progress... ({elapsed:.0f}s)", end="", flush=True)
        
        time.sleep(2)
        
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        break
    except Exception as e:
        print(f"\nError: {e}")
        time.sleep(5)