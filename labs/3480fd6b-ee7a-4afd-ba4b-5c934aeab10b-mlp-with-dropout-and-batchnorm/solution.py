# --- cell 1 ---
import torch
import torch.nn as nn

class RegularizedMLP(nn.Module):
    """MLP with BatchNorm1d and Dropout for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.3):
        super().__init__()
        # TODO: Build at least two hidden blocks:
        #   Linear -> BatchNorm1d -> ReLU (or similar) -> Dropout
        # Final layer: Linear to 1 logit.
        # Store layers on self (Sequential is fine).
        layers, prev = [nn.BatchNorm1d(input_dim)], input_dim
        layers+=[nn.Linear(prev, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        prev = hidden_dim
        layers+=[nn.Linear(prev, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout_p)]
        prev = hidden_dim
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) logits for batch x of shape (N, input_dim)."""
        # TODO: run x through your network and squeeze the last dim if needed
        return self.net(x).squeeze(-1)


def train_model(model, X_train, y_train, epochs=150, lr=1e-2):
    """Train model in-place with BCEWithLogitsLoss + Adam. Return model."""
    # TODO:
    # - criterion = nn.BCEWithLogitsLoss()
    # - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # - loop epochs: zero_grad -> forward -> loss -> backward -> step
    # - y_train is float 0/1 with shape (N,)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma = 0.5)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return model