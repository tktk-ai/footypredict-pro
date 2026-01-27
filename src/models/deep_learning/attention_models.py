"""
Attention Models
Various attention mechanisms for football prediction.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class MultiHeadSelfAttention(nn.Module):
        """Multi-head self-attention for match features."""
        
        def __init__(
            self,
            embed_dim: int = 64,
            num_heads: int = 4,
            dropout: float = 0.1
        ):
            super().__init__()
            self.attention = nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(embed_dim)
        
        def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
            return self.norm(x + attn_out), attn_weights
    
    
    class CrossAttention(nn.Module):
        """Cross-attention between home and away team features."""
        
        def __init__(self, embed_dim: int = 64, num_heads: int = 4):
            super().__init__()
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, num_heads, batch_first=True
            )
            self.norm = nn.LayerNorm(embed_dim)
        
        def forward(
            self,
            query: torch.Tensor,
            key_value: torch.Tensor
        ) -> torch.Tensor:
            attn_out, _ = self.cross_attn(query, key_value, key_value)
            return self.norm(query + attn_out)
    
    
    class AttentionModel(nn.Module):
        """Full attention-based prediction model."""
        
        def __init__(
            self,
            input_dim: int = 32,
            embed_dim: int = 64,
            num_heads: int = 4,
            num_layers: int = 2,
            output_dim: int = 3,
            dropout: float = 0.2
        ):
            super().__init__()
            
            self.input_proj = nn.Linear(input_dim, embed_dim)
            
            # Self-attention layers
            self.self_attention = nn.ModuleList([
                MultiHeadSelfAttention(embed_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
            
            # Cross-attention for home vs away
            self.cross_attention = CrossAttention(embed_dim, num_heads)
            
            # Output
            self.output = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, output_dim)
            )
        
        def forward(
            self,
            home_features: torch.Tensor,
            away_features: torch.Tensor
        ) -> Tuple[torch.Tensor, Dict]:
            """
            Forward pass with attention.
            
            Args:
                home_features: (batch, seq_len, input_dim)
                away_features: (batch, seq_len, input_dim)
            """
            # Project inputs
            home = self.input_proj(home_features)
            away = self.input_proj(away_features)
            
            attention_weights = {}
            
            # Self-attention
            for i, layer in enumerate(self.self_attention):
                home, home_attn = layer(home)
                away, away_attn = layer(away)
                attention_weights[f'self_layer_{i}'] = {
                    'home': home_attn.detach(),
                    'away': away_attn.detach()
                }
            
            # Cross-attention
            home_cross = self.cross_attention(home, away)
            away_cross = self.cross_attention(away, home)
            
            # Pool and combine
            home_pooled = home_cross.mean(dim=1)
            away_pooled = away_cross.mean(dim=1)
            
            combined = torch.cat([home_pooled, away_pooled], dim=-1)
            output = self.output(combined)
            
            return output, attention_weights


class AttentionPredictor:
    """Wrapper for attention-based prediction."""
    
    def __init__(
        self,
        input_dim: int = 32,
        embed_dim: int = 64,
        seq_len: int = 10
    ):
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.model = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        if TORCH_AVAILABLE:
            self.model = AttentionModel(input_dim, embed_dim).to(self.device)
    
    def encode_team_history(self, matches: list) -> np.ndarray:
        """Encode team match history."""
        sequence = np.zeros((self.seq_len, self.input_dim))
        
        for i, match in enumerate(matches[-self.seq_len:]):
            idx = self.seq_len - len(matches[-self.seq_len:]) + i
            sequence[idx, 0] = match.get('goals_for', 0)
            sequence[idx, 1] = match.get('goals_against', 0)
            sequence[idx, 2] = match.get('xg', 0)
            sequence[idx, 3] = match.get('possession', 50) / 100
            sequence[idx, 4] = match.get('shots', 0) / 20
        
        return sequence
    
    def predict(
        self,
        home_history: list,
        away_history: list
    ) -> Dict:
        """Predict match with attention weights."""
        if not TORCH_AVAILABLE or self.model is None:
            return {'home': 0.4, 'draw': 0.25, 'away': 0.35}
        
        home_enc = self.encode_team_history(home_history)
        away_enc = self.encode_team_history(away_history)
        
        self.model.eval()
        with torch.no_grad():
            home_t = torch.tensor(home_enc, dtype=torch.float32).unsqueeze(0).to(self.device)
            away_t = torch.tensor(away_enc, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            logits, attn_weights = self.model(home_t, away_t)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return {
            'home': float(probs[0]),
            'draw': float(probs[1]),
            'away': float(probs[2]),
            'model': 'attention',
            'attention_available': True
        }
    
    def get_attention_explanation(
        self,
        home_history: list,
        away_history: list
    ) -> Dict:
        """Get attention weights for interpretation."""
        if not TORCH_AVAILABLE or self.model is None:
            return {}
        
        home_enc = self.encode_team_history(home_history)
        away_enc = self.encode_team_history(away_history)
        
        self.model.eval()
        with torch.no_grad():
            home_t = torch.tensor(home_enc, dtype=torch.float32).unsqueeze(0).to(self.device)
            away_t = torch.tensor(away_enc, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            _, attn_weights = self.model(home_t, away_t)
        
        # Extract attention patterns
        explanation = {
            'most_important_home_match': 0,
            'most_important_away_match': 0,
        }
        
        if 'self_layer_0' in attn_weights:
            home_attn = attn_weights['self_layer_0']['home'].cpu().numpy()
            away_attn = attn_weights['self_layer_0']['away'].cpu().numpy()
            
            explanation['most_important_home_match'] = int(np.argmax(home_attn.mean(axis=(0, 1))))
            explanation['most_important_away_match'] = int(np.argmax(away_attn.mean(axis=(0, 1))))
        
        return explanation


_model: Optional[AttentionPredictor] = None

def get_model() -> AttentionPredictor:
    global _model
    if _model is None:
        _model = AttentionPredictor()
    return _model
