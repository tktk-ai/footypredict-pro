"""
Temporal CNN Model
Dilated causal convolutions for time series prediction.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CausalConv1d(nn.Module):
        """Causal 1D convolution with dilation."""
        
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int = 1
        ):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=self.padding, dilation=dilation
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)
            if self.padding > 0:
                x = x[:, :, :-self.padding]
            return x
    
    
    class TemporalBlock(nn.Module):
        """Temporal convolutional block with residual connection."""
        
        def __init__(
            self,
            n_inputs: int,
            n_outputs: int,
            kernel_size: int,
            dilation: int,
            dropout: float = 0.2
        ):
            super().__init__()
            
            self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
            self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
            
            self.net = nn.Sequential(
                self.conv1,
                nn.ReLU(),
                nn.Dropout(dropout),
                self.conv2,
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)
    
    
    class TemporalCNN(nn.Module):
        """Temporal CNN for match sequence prediction."""
        
        def __init__(
            self,
            input_dim: int = 32,
            hidden_channels: int = 64,
            num_levels: int = 4,
            kernel_size: int = 3,
            output_dim: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            
            layers = []
            for i in range(num_levels):
                dilation = 2 ** i
                in_channels = input_dim if i == 0 else hidden_channels
                layers.append(TemporalBlock(
                    in_channels, hidden_channels, kernel_size, dilation, dropout
                ))
            
            self.network = nn.Sequential(*layers)
            self.output = nn.Linear(hidden_channels, output_dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, features) -> (batch, features, seq_len)
            x = x.transpose(1, 2)
            x = self.network(x)
            # Use last time step
            x = x[:, :, -1]
            return self.output(x)


class TemporalCNNModel:
    """Wrapper for Temporal CNN model."""
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_channels: int = 64,
        seq_len: int = 10
    ):
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.seq_len = seq_len
        self.model = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        if TORCH_AVAILABLE:
            self.model = TemporalCNN(input_dim, hidden_channels).to(self.device)
    
    def encode_sequence(self, matches: list) -> np.ndarray:
        """Encode match sequence."""
        sequence = np.zeros((self.seq_len, self.input_dim))
        
        for i, match in enumerate(matches[-self.seq_len:]):
            idx = self.seq_len - len(matches[-self.seq_len:]) + i
            sequence[idx, 0] = match.get('goals_scored', 0)
            sequence[idx, 1] = match.get('goals_conceded', 0)
            sequence[idx, 2] = match.get('xg', 0)
            sequence[idx, 3] = match.get('shots', 0) / 20.0
        
        return sequence
    
    def predict(self, sequence: list) -> Dict:
        """Predict from sequence."""
        if not TORCH_AVAILABLE or self.model is None:
            return {'home': 0.4, 'draw': 0.25, 'away': 0.35}
        
        enc = self.encode_sequence(sequence)
        
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(enc, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return {
            'home': float(probs[0]),
            'draw': float(probs[1]),
            'away': float(probs[2]),
            'model': 'temporal_cnn'
        }


_model: Optional[TemporalCNNModel] = None

def get_model() -> TemporalCNNModel:
    global _model
    if _model is None:
        _model = TemporalCNNModel()
    return _model
