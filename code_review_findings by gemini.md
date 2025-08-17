# Code Review Findings

This document summarizes the findings from a review of the project's Python scripts, focusing on the machine learning pipeline.

## 1. Critical Issue: Unstable Data Splitting & Potential Data Leakage

-   **Problem:** The `SubclavianDataset` class in `subclavian_dataset.py` re-creates the train/validation/test split every time it is initialized. It splits the data by randomly shuffling all file paths based on a fixed random seed.
-   **Impact:** This is a critical flaw.
    -   **Reproducibility:** It makes experiments difficult to reproduce reliably across different scripts or sessions.
    -   **Data Leakage:** There is a high risk of the test data overlapping with the training data if different scripts initialize the dataset, which would make evaluation metrics (like accuracy) misleadingly high and not representative of the model's true performance on unseen data.
-   **Recommendation:** The data should be split **once** into fixed `train`, `validation`, and `test` sets. These splits should be saved (e.g., as `train_labels.csv`, `val_labels.csv`, `test_labels.csv`). The `SubclavianDataset` class should then be modified to load these pre-defined splits directly.

## 2. Moderate Issue: Inconsistent Device (CPU/GPU) Handling

-   **Problem:** In `train_subclavian.py`, the script explicitly sets the device to CPU (`device = torch.device('cpu')`) but then, in the final evaluation loop, it attempts to move data tensors to the GPU (`points.cuda()`, `target.cuda()`).
-   **Impact:** This will cause the script to crash if run on a machine without a GPU.
-   **Recommendation:** Use the `device` variable consistently throughout the script to move both the model and all data tensors to the correct device. This makes the code portable and robust.

## 3. Moderate Issue: Noisy Validation Metric

-   **Problem:** During training, `train_subclavian.py` evaluates the model's performance on a *single batch* of validation data every 10 training steps.
-   **Impact:** A single batch is not representative of the entire validation set. This provides a very noisy and unstable validation accuracy, making it difficult to judge if the model is truly improving epoch-over-epoch.
-   **Recommendation:** The validation should be performed on the **entire** validation dataset at the end of each epoch. This provides a stable and reliable metric for model selection and hyperparameter tuning.

## 4. Minor Issues & Code Style

-   **Non-Fixed Random Seed:** The training script `train_subclavian.py` generates a new random seed for every run (`opt.manualSeed = random.randint(1, 10000)`). For reproducible research, this seed should be a fixed value that can be set via a command-line argument.
-   **Deprecated PyTorch Syntax:** The PointNet model code in `pointnet/model.py` uses `Variable`, which is deprecated. While this is not a functional bug, the code should be updated to use modern PyTorch standards for long-term maintainability.
-   **Label Tensor Shape:** The training script reshapes the label tensor with `target = target[:, 0]`. It would be cleaner to fix this in the `SubclavianDataset` class so it returns a tensor with the correct shape `[batch_size]` instead of `[batch_size, 1]`.
