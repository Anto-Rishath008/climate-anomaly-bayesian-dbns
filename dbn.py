import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from rbm import RBM

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DBN(nn.Module):
    """
    Stacked RBMs + Bayesian (MC Dropout) head for regression/classification.
    - Pretrain RBMs with Contrastive Divergence (unsupervised)
    - Fine-tune with supervised head (Bayesian via MC Dropout)
    """
    def __init__(self, layer_sizes: List[int], dropout_p: float = 0.2, task: str = "regression", out_dim: int = 1):
        """
        layer_sizes: [n_visible, h1, h2, ...]
        task: 'regression' or 'classification'
        out_dim: output dimension (1 for scalar target; equals n_features for vector target)
        """
        super().__init__()
        assert len(layer_sizes) >= 2, "Provide at least input and one hidden size"
        self.layer_sizes = layer_sizes
        self.task = task
        self.out_dim = out_dim

        # Build RBMs for pretraining
        self.rbms = nn.ModuleList([
            RBM(n_visible=layer_sizes[i], n_hidden=layer_sizes[i+1], k=1)
            for i in range(len(layer_sizes)-1)
        ])

        # Build feedforward network initialized from RBMs
        fcs = []
        for i in range(len(layer_sizes)-1):
            fcs.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            fcs.append(nn.Dropout(p=dropout_p))
            fcs.append(nn.ReLU(inplace=True))
        self.feature_net = nn.Sequential(*fcs)

        # Output head
        if self.task == "regression":
            self.head = nn.Linear(layer_sizes[-1], out_dim)
            self.loss_fn = nn.MSELoss()
        elif self.task == "classification":
            self.head = nn.Linear(layer_sizes[-1], out_dim)
            self.loss_fn = nn.BCEWithLogitsLoss() if out_dim == 1 else nn.CrossEntropyLoss()
        else:
            raise ValueError("task must be 'regression' or 'classification'")

        self.to(Device)

    @torch.no_grad()
    def init_from_rbms(self):
        """Copy RBM weights into the feedforward layers."""
        rbm_idx = 0
        for m in self.feature_net:
            if isinstance(m, nn.Linear):
                W = self.rbms[rbm_idx].W  # (hidden, visible)
                h_bias = self.rbms[rbm_idx].h_bias
                m.weight.data = W.data.clone()
                m.bias.data = h_bias.data.clone()
                rbm_idx += 1
        return self

    def forward(self, x: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        self.train(mc_dropout)  # enable dropout during MC sampling
        feats = self.feature_net(x)
        return self.head(feats)

    def pretrain_rbms(self, X: torch.Tensor, epochs: int = 5, batch_size: int = 128, lr: float = 1e-2):
        """
        Greedy layer-wise pretraining of RBMs on input X.
        X: (N, n_visible)
        """
        data = X.to(Device)
        for i, rbm in enumerate(self.rbms):
            v = data
            for ep in range(epochs):
                perm = torch.randperm(v.size(0), device=Device)
                for b in range(0, v.size(0), batch_size):
                    batch_idx = perm[b:b+batch_size]
                    vb = v[batch_idx]
                    rbm.contrastive_divergence(vb, lr=lr)
                with torch.no_grad():
                    p_h, _ = rbm.v_to_h(v)
                    v = p_h  # next layer input
            data = v
        self.init_from_rbms()

    def fit(self, X: torch.Tensor, y: torch.Tensor, val: Tuple[torch.Tensor, torch.Tensor]=None,
            epochs: int = 20, batch_size: int = 128, lr: float = 1e-3):
        X = X.to(Device); y = y.to(Device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        best_val = float("inf")

        for ep in range(epochs):
            self.train(True)  # enable dropout during training
            perm = torch.randperm(X.size(0), device=Device)
            total_loss = 0.0
            for b in range(0, X.size(0), batch_size):
                idx = perm[b:b+batch_size]
                xb = X[idx]; yb = y[idx]
                opt.zero_grad()
                preds = self.forward(xb, mc_dropout=True)
                loss = self.loss_fn(preds, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)

            msg = f"[Epoch {ep+1}/{epochs}] train_loss={total_loss/X.size(0):.4f}"

            if val is not None:
                self.eval()
                with torch.no_grad():
                    Xv, yv = val
                    Xv = Xv.to(Device); yv = yv.to(Device)
                    pv = self.head(self.feature_net(Xv))
                    vloss = self.loss_fn(pv, yv).item()
                msg += f" | val_loss={vloss:.4f}"
                if vloss < best_val:
                    best_val = vloss
                    torch.save(self.state_dict(), "best_dbn.pt")
            print(msg)

    @torch.no_grad()
    def mc_predict(self, X: torch.Tensor, T: int = 20):
        """
        Monte Carlo Dropout predictions.
        Returns mean and std over T stochastic forward passes.
        """
        X = X.to(Device)
        preds = []
        for _ in range(T):
            self.train(True)  # keep dropout on
            preds.append(self.forward(X, mc_dropout=True))
        P = torch.stack(preds, dim=0)  # (T, N, out_dim)
        mean = P.mean(dim=0)
        std = P.std(dim=0)
        return mean, std