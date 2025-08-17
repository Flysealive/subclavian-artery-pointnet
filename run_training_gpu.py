#!/usr/bin/env python3
"""
Wrapper to run GPU training with proper encoding
"""

import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Run the actual training script
if __name__ == "__main__":
    # Import and run the training
    import hybrid_5fold_cv_training