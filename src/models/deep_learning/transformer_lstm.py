"""
Transformer-LSTM Hybrid Model
Combines Transformer attention with LSTM for sequence modeling.

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
    class TransformerLSTM(nn.Module):
        """Transformer-LSTM hybrid for match sequence prediction."""
        
        def __init__(
            self,
            input_dim: int = 32,
            hidden_dim: int = 128,
            num_heads: int = 4,
            num_layers: int = 2,
            output_dim: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            
            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # LSTM for sequential refinement
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
            
            # Output layers
            self.output = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            
        def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor = None
        ) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: (batch, seq_len, input_dim)
                mask: Optional attention mask
            """
            # Project input
            x = self.input_proj(x)
            
            # Transformer encoding
            x = self.transformer(x, src_key_padding_mask=mask)
            
            # LSTM refinement
            x, _ = self.lstm(x)
            
            # Use last hidden state
            x = x[:, -1, :]
            
            return self.output(x)


class TransformerLSTMModel:
    """Wrapper for Transformer-LSTM model."""
    
    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 128,
        seq_len: int = 10
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.model = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        if TORCH_AVAILABLE:
            self.model = TransformerLSTM(input_dim, hidden_dim).to(self.device)
    
    def encode_match_sequence(self, matches: list) -> np.ndarray:
        """Encode a sequence of matches."""
        sequence = np.zeros((self.seq_len, self.input_dim))
        
        for i, match in enumerate(matches[-self.seq_len:]):
            idx = self.seq_len - len(matches[-self.seq_len:]) + i
            # Basic encoding
            sequence[idx, 0] = match.get('home_goals', 0)
            sequence[idx, 1] = match.get('away_goals', 0)
            sequence[idx, 2] = 1 if match.get('result') == 'W' else 0
            sequence[idx, 3] = 1 if match.get('result') == 'D' else 0
        
        return sequence
    
    def predict(
        self,
        home_sequence: list,
        away_sequence: list
    ) -> Dict:
        """Predict match outcome from team sequences."""
        if not TORCH_AVAILABLE or self.model is None:
            return {'home': 0.4, 'draw': 0.25, 'away': 0.35}
        
        # Encode sequences
        home_enc = self.encode_match_sequence(home_sequence)
        away_enc = self.encode_match_sequence(away_sequence)
        
        # Combine
        combined = np.concatenate([home_enc, away_enc], axis=1)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return {
            'home': float(probs[0]),
            'draw': float(probs[1]),
            'away': float(probs[2]),
            'model': 'transformer_lstm'
        }
    
    def train_step(self, batch: Tuple) -> float:
        """Single training step."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        x, y = batch
        self.model.train()
        
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        outputs = self.model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        return loss.item()


_model: Optional[TransformerLSTMModel] = None

def get_model() -> TransformerLSTMModel:
    global _model
    if _model is None:
        _model = TransformerLSTMModel()
    return _model
