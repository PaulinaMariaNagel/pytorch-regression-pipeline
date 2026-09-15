# pytorch-regression-pipeline
Minimal, leak-free PyTorch regression pipeline featuring K-Fold cross-validation, mini-batch DataLoader, early stopping, and out-of-fold calibration.

# PyTorch K-Fold Regression Pipeline

A modular, production-ready PyTorch template for continuous regression tasks featuring strictly leak-free $K$-Fold cross-validation, dynamic mini-batching via `DataLoader`, validation-monitored early stopping with checkpoint restoration, and aggregated out-of-fold calibration.

---

## Key Highlights & Methodological Best Practices

* **Zero Data Leakage**: Feature standardization (`StandardScaler`) is fitted exclusively on each training fold (`fit_transform`) and subsequently mapped to the validation fold (`transform`).
* **State Checkpointing & Early Stopping**: Monitors out-of-sample validation loss per epoch. If performance fails to improve within the specified patience window, training halts early and optimal parameter tensors (`state_dict`) are restored via `copy.deepcopy`.
* **Clean Mini-Batching**: Integrates `TensorDataset` and `DataLoader` with per-epoch shuffle to ensure stable stochastic gradient descent (Adam) with $L_2$ weight decay regularization.
* **Out-of-Fold (OOF) Evaluation**: Collects raw model predictions across all held-out folds to compute unbiased overall performance metrics ($R^2$, MSE, MAE) and visualize true vs. predicted regression calibration.

---

## Pipeline Overview

```text
       Raw Dataset (X, y)
               │
      [5-Fold Partitioning]
               │
    ┌──────────┴──────────┐
                         
Training Folds (4/5)   Validation Fold (1/5)
    │                     │
 Fit StandardScaler     Transform StandardScaler
    │                     │
 TensorDataset +          Evaluate Validation Loss
 Mini-Batch DataLoader         │
    │               Check Early Stopping Threshold
 Optimize Network   (Restore Lowest-Loss Weights)
    │                     │
    └──────────┬──────────┘
               
   Aggregated Out-of-Fold (OOF)
       Metrics & Scatter Plot
