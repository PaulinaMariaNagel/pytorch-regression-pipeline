import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility and hardware config

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# generate dataset

X_raw, y_raw = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    noise=15.0,
    random_state=42
)

# model architecture

class RegressionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(RegressionMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# early stop utility

class EarlyStopping:
    """Monitors validation loss and preserves best model weights via deepcopy."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model: nn.Module):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


# k fold cross validation

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

max_epochs = 200
batch_size = 64
patience = 15

cv_mse = []
cv_mae = []
cv_r2 = []
oof_predictions = np.zeros_like(y_raw)

# cross-alidation oop with mini batching and arly stopping

for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw, y_raw)):
    print(f"\n--- Training Fold {fold + 1}/{n_splits} ---")

    # Split into training and validation sets
    X_train, X_val = X_raw[train_idx], X_raw[val_idx]
    y_train, y_val = y_raw[train_idx], y_raw[val_idx]

    # Preprocessing: Fit scaler strictly on training split to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Tensor instantiation
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Mini-batch DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Reinitialise model, criterion, optimiser, and early stopping handler per fold
    model = RegressionMLP(input_dim=10, hidden_dim=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=patience)

    # Training and validation epochs
    for epoch in range(max_epochs):
        # Optimisation phase
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        # Validation phase (evaluation on the complete validation fold)
        model.eval()
        with torch.no_grad():
            val_preds_epoch = model(X_val_t)
            epoch_val_loss = criterion(val_preds_epoch, y_val_t).item()

        # Assess early stopping criteria
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}. Restoring best checkpoint.")
            break

    # Restore optimal model parameters prior to overfitting
    early_stopping.restore_best_weights(model)

    # Final out-of-fold inference using the restored checkpoint
    model.eval()
    with torch.no_grad():
        val_preds_final = model(X_val_t).cpu().numpy().flatten()

    fold_mse = mean_squared_error(y_val, val_preds_final)
    fold_mae = mean_absolute_error(y_val, val_preds_final)
    fold_r2 = r2_score(y_val, val_preds_final)

    cv_mse.append(fold_mse)
    cv_mae.append(fold_mae)
    cv_r2.append(fold_r2)
    oof_predictions[val_idx] = val_preds_final

    print(f"Fold {fold + 1} | Optimal Val MSE: {fold_mse:.2f} | MAE: {fold_mae:.2f} | R²: {fold_r2:.3f}")


# aggregated summary statistics

print("\n" + "=" * 45)
print(f"=== {n_splits}-Fold Cross-Validation Summary ===")
print("=" * 45)
print(f"Mean MSE : {np.mean(cv_mse):.2f} (+/- {np.std(cv_mse):.2f})")
print(f"Mean MAE : {np.mean(cv_mae):.2f} (+/- {np.std(cv_mae):.2f})")
print(f"Mean R²  : {np.mean(cv_r2):.3f} (+/- {np.std(cv_r2):.3f})")

overall_r2 = r2_score(y_raw, oof_predictions)
print(f"Overall OOF R² : {overall_r2:.3f}")

# Out-of-Fold Calibration Plot

plt.figure(figsize=(7, 6))
plt.scatter(y_raw, oof_predictions, alpha=0.5, color="#1f77b4", edgecolors="none")

min_val = min(y_raw.min(), oof_predictions.min())
max_val = max(y_raw.max(), oof_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Identity Line (y = x)")

plt.title(f"Out-of-Fold Predictions with Early Stopping (R² = {overall_r2:.3f})")
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values (OOF)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
