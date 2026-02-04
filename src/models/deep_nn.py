"""
Enhanced Deep Neural Network for Football Prediction

ENHANCED VERSION with:
- Residual connections (skip connections)
- Self-attention layer for feature importance
- Advanced regularization (label smoothing, mixup)
- Learning rate warmup + cosine annealing
- Model export for production (ONNX, TorchScript)

Author: FootyPredict Pro
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import pickle
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# Self-Attention Module
# =============================================================================

class SelfAttention(nn.Module):
    """
    Self-attention layer for feature importance weighting.
    Learns which features are most important for prediction.
    """
    
    def __init__(self, input_dim: int, n_heads: int = 4):
        super().__init__()
        
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads if input_dim >= n_heads else input_dim
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, features)
        batch_size = x.size(0)
        
        # Compute Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Attention scores (simplified for 1D input)
        attn = torch.softmax(q * k * self.scale, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = attn * v
        out = self.out(out)
        
        return out


# =============================================================================
# Residual Block
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    Helps with gradient flow and training deeper networks.
    """
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        
        # Shortcut connection
        self.shortcut = nn.Identity() if in_features == out_features else nn.Linear(in_features, out_features)
        
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.block(x)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


# =============================================================================
# Mixup Data Augmentation
# =============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation.
    Blends pairs of samples to create synthetic training examples.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, 
                    y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    """Compute mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Label Smoothing Loss
# =============================================================================

class LabelSmoothingLoss(nn.Module):
    """
    Cross entropy loss with label smoothing.
    Prevents overconfident predictions.
    """
    
    def __init__(self, num_classes: int = 3, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# =============================================================================
# Focal Loss for Class Imbalance
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses learning on hard examples (like draws in football).
    
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
    
    Args:
        gamma: Focusing parameter (default 2.0)
        alpha: Class weights (default balanced for H/D/A)
    """
    
    def __init__(self, gamma: float = 2.0, alpha: list = None):
        super().__init__()
        self.gamma = gamma
        # Default: upweight draws (class 1)
        self.alpha = torch.tensor(alpha or [0.25, 0.50, 0.25])
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Get probabilities
        probs = torch.softmax(pred, dim=-1)
        
        # One-hot encode targets
        target_one_hot = torch.zeros_like(probs)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Get probability of true class
        pt = (probs * target_one_hot).sum(dim=-1)
        
        # Focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Class weights
        alpha_t = self.alpha.to(pred.device)[target]
        
        # Cross entropy
        ce = -torch.log(pt + 1e-10)
        
        loss = alpha_t * focal_weight * ce
        return loss.mean()


# =============================================================================
# Cosine Annealing with Warmup
# =============================================================================

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.
    """
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            decay_steps = max(1, self.max_epochs - self.warmup_epochs)
            progress = (self.last_epoch - self.warmup_epochs) / decay_steps
            progress = min(1.0, progress)  # Cap at 1.0
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]


# =============================================================================
# Enhanced Deep Football Net
# =============================================================================

class DeepFootballNet(nn.Module):
    """
    Enhanced deep neural network for football prediction.
    
    Architecture:
    - Input projection layer
    - Self-attention for feature importance
    - Multiple residual blocks with skip connections
    - Output layer with softmax (3 classes)
    
    Training Enhancements:
    - Label smoothing for better calibration
    - Mixup data augmentation
    - Cosine annealing with warmup
    """
    
    def __init__(self,
                 input_dim: int = 76,
                 hidden_layers: int = 4,
                 hidden_units: int = 256,
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 use_residual: bool = True,
                 num_classes: int = 3):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.scaler = StandardScaler()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
        
        # Self-attention
        if use_attention:
            self.attention = SelfAttention(hidden_units, n_heads=4)
            self.attn_norm = nn.LayerNorm(hidden_units)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        current_dim = hidden_units
        
        for i in range(hidden_layers):
            out_dim = hidden_units // (2 ** min(i, 2))
            
            if use_residual:
                self.blocks.append(ResidualBlock(current_dim, out_dim, dropout * (1 - i * 0.1)))
            else:
                self.blocks.append(nn.Sequential(
                    nn.Linear(current_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout * (1 - i * 0.1))
                ))
            current_dim = out_dim
        
        # Output layer
        self.output = nn.Linear(current_dim, num_classes)
        
        # Training config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = 300
        self.batch_size = 128
        self.learning_rate = 0.001
        self.patience = 30
        self.warmup_epochs = 10
        self.use_mixup = True
        self.mixup_alpha = 0.2
        self.label_smoothing = 0.1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Self-attention with residual
        if self.use_attention:
            attn_out = self.attention(x)
            x = self.attn_norm(x + attn_out)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        return self.output(x)
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train the enhanced neural network"""
        self.to(self.device)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss (with label smoothing)
        criterion = LabelSmoothingLoss(num_classes=3, smoothing=self.label_smoothing)
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        # Scheduler with warmup + cosine annealing
        scheduler = CosineWarmupScheduler(
            optimizer, 
            warmup_epochs=self.warmup_epochs, 
            max_epochs=self.epochs
        )
        
        best_acc = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Mixup augmentation
                if self.use_mixup and self.training:
                    mixed_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, self.mixup_alpha)
                    outputs = self(mixed_X)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            
            # Validation accuracy
            self.eval()
            with torch.no_grad():
                preds = self(X_tensor).argmax(dim=1)
                acc = (preds == y_tensor).float().mean().item()
            
            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                best_state = self.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                if verbose:
                    logger.info(f"  Early stopping at epoch {epoch+1}")
                break
            
            if verbose and (epoch + 1) % 50 == 0:
                lr = scheduler.get_last_lr()[0]
                logger.info(f"  Epoch {epoch+1}/{self.epochs} - Loss: {total_loss/len(dataloader):.4f}, Acc: {acc:.2%}, LR: {lr:.6f}")
        
        # Load best state
        if best_state:
            self.load_state_dict(best_state)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        self.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        self.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        
        return probs
    
    def predict_with_uncertainty(self, X: np.ndarray, n_samples: int = 50
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout for uncertainty estimation.
        Runs multiple forward passes with dropout enabled.
        
        Returns:
            predictions: Mean class predictions
            confidences: Mean confidence (max probability)
            uncertainties: Standard deviation of probabilities
        """
        # Keep model in training mode (dropout active)
        self.train()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        all_probs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self(X_tensor)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # Stack and compute statistics
        all_probs = np.stack(all_probs, axis=0)  # (n_samples, batch, 3)
        mean_probs = all_probs.mean(axis=0)  # (batch, 3)
        std_probs = all_probs.std(axis=0)  # (batch, 3)
        
        predictions = mean_probs.argmax(axis=1)
        confidences = mean_probs.max(axis=1)
        uncertainties = std_probs.mean(axis=1)  # Average std across classes
        
        # Return to eval mode
        self.eval()
        
        return predictions, confidences, uncertainties
    
    def save(self, path: str):
        """Save model to file"""
        self.to('cpu')
        with open(path, 'wb') as f:
            pickle.dump({
                'state_dict': self.state_dict(),
                'scaler': self.scaler,
                'input_dim': self.input_dim,
                'use_attention': self.use_attention,
                'use_residual': self.use_residual,
            }, f)
        self.to(self.device)
        logger.info(f"Saved DeepFootballNet to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DeepFootballNet':
        """Load model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            input_dim=data['input_dim'],
            use_attention=data['use_attention'],
            use_residual=data['use_residual']
        )
        model.load_state_dict(data['state_dict'])
        model.scaler = data['scaler']
        
        return model
    
    def export_onnx(self, path: str, input_shape: Tuple[int, int] = (1, 76)):
        """Export model to ONNX format"""
        self.eval()
        self.to('cpu')
        
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            self,
            dummy_input,
            path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        logger.info(f"Exported to ONNX: {path}")
        self.to(self.device)
    
    def export_torchscript(self, path: str):
        """Export model to TorchScript"""
        self.eval()
        self.to('cpu')
        
        scripted = torch.jit.script(self)
        scripted.save(path)
        
        logger.info(f"Exported to TorchScript: {path}")
        self.to(self.device)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Testing Enhanced Deep Neural Network...")
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    X = np.random.randn(1000, 76)
    y = np.random.randint(0, 3, 1000)
    
    # Test with all features
    model = DeepFootballNet(
        input_dim=76,
        use_attention=True,
        use_residual=True
    )
    model.epochs = 50  # Quick test
    model.fit(X[:800], y[:800], verbose=True)
    
    # Evaluate
    preds = model.predict(X[800:])
    acc = (preds == y[800:]).mean()
    print(f"\n✅ Enhanced Deep NN Accuracy: {acc:.2%}")
    
    # Test save/load
    model.save('/tmp/deep_nn_test.pkl')
    loaded = DeepFootballNet.load('/tmp/deep_nn_test.pkl')
    preds2 = loaded.predict(X[800:])
    assert np.array_equal(preds, preds2), "Save/load mismatch!"
    print("✅ Save/Load verified")
