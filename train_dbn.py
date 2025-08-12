import numpy as np
import torch
from dbn import DBN, Device

def build_supervised_xy(data: np.ndarray):
    """
    Turn a sequence of anomaly vectors into (X_t -> y_{t+1}) pairs.
    data shape: (T, F) where T = timesteps, F = features (flattened grid)
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 time steps to build supervised pairs.")
    X = data[:-1]
    y = data[1:]  # next-step prediction (vector target)
    return X, y

def main():
    # Load preprocessed splits
    train = np.load("train_data.npy")
    val = np.load("val_data.npy")

    # Build supervised datasets
    X_train, y_train = build_supervised_xy(train)
    X_val, y_val = build_supervised_xy(val)

    # Convert to torch
    X_train = torch.tensor(X_train, dtype=torch.float32, device=Device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=Device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=Device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=Device)

    n_features = X_train.shape[1]
    print(f"n_features={n_features}")

    # Define DBN sizes (shrink if OOM)
    layer_sizes = [n_features, 512, 256]

    # Out dim equals features for vector next-step forecast
    model = DBN(layer_sizes=layer_sizes, dropout_p=0.2, task="regression", out_dim=n_features)

    # Pretrain RBMs (unsupervised)
    print("Pretraining RBMs...")
    model.pretrain_rbms(X_train, epochs=3, batch_size=64, lr=1e-2)

    # Supervised fine-tuning
    print("Fine-tuning...")
    model.fit(X_train, y_train, val=(X_val, y_val), epochs=10, batch_size=64, lr=1e-3)

    # Save final model
    torch.save(model.state_dict(), "dbn_final.pt")
    print("Saved dbn_final.pt")

if __name__ == "__main__":
    main()