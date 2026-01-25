"""
CNN-BiLSTM-Attention Model for Football Prediction V3.0
State-of-the-art deep learning architecture

Architecture:
1. CNN extracts local patterns from features
2. BiLSTM captures sequential dependencies
3. Multi-head attention weighs important features
4. Temporal attention aggregates sequence
5. Multi-task heads for different predictions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Deep learning models will not be available.")


if TORCH_AVAILABLE:
    
    class CNNFeatureExtractor(nn.Module):
        """
        CNN for local feature extraction from player/team data.
        Uses multiple kernel sizes to capture different patterns.
        """
        
        def __init__(
            self,
            input_dim: int,
            num_filters: int = 128,
            kernel_sizes: List[int] = [2, 3, 4, 5]
        ):
            super().__init__()
            
            self.convs = nn.ModuleList([
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                    padding=k // 2
                )
                for k in kernel_sizes
            ])
            
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_filters)
                for _ in kernel_sizes
            ])
            
            self.output_dim = num_filters * len(kernel_sizes)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, input_dim)
            Returns:
                (batch, seq_len, output_dim)
            """
            # Transpose for conv1d: (batch, input_dim, seq_len)
            x = x.transpose(1, 2)
            
            conv_outputs = []
            for conv, bn in zip(self.convs, self.batch_norms):
                out = F.leaky_relu(bn(conv(x)))
                conv_outputs.append(out)
            
            # Concatenate all filter outputs
            combined = torch.cat(conv_outputs, dim=1)
            
            # Transpose back: (batch, seq_len, output_dim)
            return combined.transpose(1, 2)


    class BiLSTMEncoder(nn.Module):
        """
        Bidirectional LSTM for sequential pattern learning.
        """
        
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3
        ):
            super().__init__()
            
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            self.output_dim = hidden_dim * 2  # Bidirectional
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
            """
            Args:
                x: (batch, seq_len, input_dim)
            Returns:
                outputs: (batch, seq_len, hidden_dim * 2)
                (hidden, cell): Final states
            """
            outputs, (hidden, cell) = self.lstm(x)
            return outputs, (hidden, cell)


    class MultiHeadSelfAttention(nn.Module):
        """
        Multi-head self-attention mechanism.
        Assigns different weights to different time steps/features.
        """
        
        def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1
        ):
            super().__init__()
            
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
            
            self.q_linear = nn.Linear(embed_dim, embed_dim)
            self.k_linear = nn.Linear(embed_dim, embed_dim)
            self.v_linear = nn.Linear(embed_dim, embed_dim)
            self.out_linear = nn.Linear(embed_dim, embed_dim)
            
            self.dropout = nn.Dropout(dropout)
            self.scale = np.sqrt(self.head_dim)
            
        def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (batch, seq_len, embed_dim)
                mask: Optional attention mask
            Returns:
                output: (batch, seq_len, embed_dim)
                attention_weights: (batch, num_heads, seq_len, seq_len)
            """
            batch_size, seq_len, _ = x.shape
            
            # Linear projections
            Q = self.q_linear(x)
            K = self.k_linear(x)
            V = self.v_linear(x)
            
            # Reshape to (batch, num_heads, seq_len, head_dim)
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, V)
            
            # Reshape back
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            output = self.out_linear(context)
            
            return output, attention_weights


    class TemporalAttention(nn.Module):
        """
        Temporal attention for weighting different time steps.
        """
        
        def __init__(self, hidden_dim: int):
            super().__init__()
            
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: (batch, seq_len, hidden_dim)
            Returns:
                context: (batch, hidden_dim)
                weights: (batch, seq_len)
            """
            # Calculate attention scores
            scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
            weights = F.softmax(scores, dim=-1)
            
            # Weighted sum
            context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
            
            return context, weights


    class CNNBiLSTMAttention(nn.Module):
        """
        Complete CNN-BiLSTM-Attention model for football match prediction.
        
        Architecture:
        1. CNN extracts local patterns from features
        2. BiLSTM captures sequential dependencies
        3. Multi-head attention weighs important features
        4. Temporal attention aggregates sequence
        5. Multi-task heads for different predictions
        """
        
        def __init__(
            self,
            input_dim: int,
            cnn_filters: int = 128,
            cnn_kernel_sizes: List[int] = [2, 3, 4, 5],
            lstm_hidden: int = 128,
            lstm_layers: int = 2,
            attention_heads: int = 8,
            dropout: float = 0.3,
            num_classes: int = 3
        ):
            super().__init__()
            
            # CNN feature extractor
            self.cnn = CNNFeatureExtractor(
                input_dim=input_dim,
                num_filters=cnn_filters,
                kernel_sizes=cnn_kernel_sizes
            )
            
            # BiLSTM encoder
            self.bilstm = BiLSTMEncoder(
                input_dim=self.cnn.output_dim,
                hidden_dim=lstm_hidden,
                num_layers=lstm_layers,
                dropout=dropout
            )
            
            # Multi-head self-attention
            self.self_attention = MultiHeadSelfAttention(
                embed_dim=self.bilstm.output_dim,
                num_heads=attention_heads,
                dropout=dropout
            )
            
            # Temporal attention for sequence aggregation
            self.temporal_attention = TemporalAttention(self.bilstm.output_dim)
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(self.bilstm.output_dim)
            
            # Dropout
            self.dropout = nn.Dropout(dropout)
            
            # Combined feature dimension
            feature_dim = self.bilstm.output_dim
            
            # Multi-task prediction heads
            # 1X2 Result
            self.result_head = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )
            
            # Goal prediction (home and away)
            self.home_goals_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 8)  # 0-7 goals
            )
            
            self.away_goals_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 8)
            )
            
            # BTTS prediction
            self.btts_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            
            # Over 2.5 prediction
            self.over25_head = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            
            # HT/FT prediction (9 combinations)
            self.htft_head = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 9)
            )
            
        def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                x: Input features (batch, seq_len, input_dim)
                return_attention: Whether to return attention weights
                
            Returns:
                Dictionary with predictions for all markets
            """
            # CNN feature extraction
            cnn_features = self.cnn(x)
            
            # BiLSTM encoding
            lstm_outputs, _ = self.bilstm(cnn_features)
            
            # Self-attention
            attended, attention_weights = self.self_attention(lstm_outputs)
            
            # Residual connection and normalization
            attended = self.layer_norm(lstm_outputs + attended)
            attended = self.dropout(attended)
            
            # Temporal attention for aggregation
            context, temporal_weights = self.temporal_attention(attended)
            
            # Generate predictions
            predictions = {
                'result': F.softmax(self.result_head(context), dim=-1),
                'home_goals': F.softmax(self.home_goals_head(context), dim=-1),
                'away_goals': F.softmax(self.away_goals_head(context), dim=-1),
                'btts': F.softmax(self.btts_head(context), dim=-1),
                'over_25': F.softmax(self.over25_head(context), dim=-1),
                'htft': F.softmax(self.htft_head(context), dim=-1)
            }
            
            if return_attention:
                predictions['attention_weights'] = attention_weights
                predictions['temporal_weights'] = temporal_weights
            
            return predictions
        
        def predict(self, x: torch.Tensor) -> Dict:
            """Generate predictions with post-processing."""
            self.eval()
            with torch.no_grad():
                preds = self.forward(x)
                
                # Calculate correct score probabilities
                home_probs = preds['home_goals'].squeeze().cpu().numpy()
                away_probs = preds['away_goals'].squeeze().cpu().numpy()
                
                correct_scores = {}
                for h in range(min(8, len(home_probs))):
                    for a in range(min(8, len(away_probs))):
                        correct_scores[f'{h}-{a}'] = float(home_probs[h] * away_probs[a])
                
                # Normalize
                total = sum(correct_scores.values())
                if total > 0:
                    correct_scores = {k: v/total for k, v in correct_scores.items()}
                
                result_probs = preds['result'].squeeze().cpu().numpy()
                btts_probs = preds['btts'].squeeze().cpu().numpy()
                over25_probs = preds['over_25'].squeeze().cpu().numpy()
                htft_probs = preds['htft'].squeeze().cpu().numpy()
                
                return {
                    'result': {
                        'home_win': float(result_probs[0]) if len(result_probs) > 0 else 0.33,
                        'draw': float(result_probs[1]) if len(result_probs) > 1 else 0.33,
                        'away_win': float(result_probs[2]) if len(result_probs) > 2 else 0.33
                    },
                    'correct_scores': dict(sorted(
                        correct_scores.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:15]),
                    'btts_yes': float(btts_probs[1]) if len(btts_probs) > 1 else 0.5,
                    'over_25': float(over25_probs[1]) if len(over25_probs) > 1 else 0.5,
                    'htft': {
                        'H/H': float(htft_probs[0]) if len(htft_probs) > 0 else 0.11,
                        'H/D': float(htft_probs[1]) if len(htft_probs) > 1 else 0.11,
                        'H/A': float(htft_probs[2]) if len(htft_probs) > 2 else 0.11,
                        'D/H': float(htft_probs[3]) if len(htft_probs) > 3 else 0.11,
                        'D/D': float(htft_probs[4]) if len(htft_probs) > 4 else 0.11,
                        'D/A': float(htft_probs[5]) if len(htft_probs) > 5 else 0.11,
                        'A/H': float(htft_probs[6]) if len(htft_probs) > 6 else 0.11,
                        'A/D': float(htft_probs[7]) if len(htft_probs) > 7 else 0.11,
                        'A/A': float(htft_probs[8]) if len(htft_probs) > 8 else 0.11
                    }
                }


    class CNNBiLSTMTrainer:
        """
        Trainer for CNN-BiLSTM-Attention model with multi-task learning.
        """
        
        def __init__(
            self,
            model: CNNBiLSTMAttention,
            learning_rate: float = 1e-4,
            weight_decay: float = 1e-5,
            task_weights: Dict[str, float] = None
        ):
            self.model = model
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            # Task weights for multi-task learning
            self.task_weights = task_weights or {
                'result': 1.0,
                'home_goals': 0.3,
                'away_goals': 0.3,
                'btts': 0.3,
                'over_25': 0.3,
                'htft': 0.2
            }
            
            # Loss functions
            self.ce_loss = nn.CrossEntropyLoss()
            
        def train_epoch(self, dataloader) -> Dict[str, float]:
            """Train for one epoch."""
            self.model.train()
            total_loss = 0
            task_losses = {k: 0.0 for k in self.task_weights.keys()}
            
            for batch in dataloader:
                features = batch['features'].to(self.device)
                
                self.optimizer.zero_grad()
                
                predictions = self.model(features)
                
                # Calculate multi-task loss
                loss = torch.tensor(0.0, device=self.device)
                
                if 'result' in batch:
                    result_loss = self.ce_loss(predictions['result'], batch['result'].to(self.device))
                    loss = loss + self.task_weights['result'] * result_loss
                    task_losses['result'] += result_loss.item()
                
                if 'home_goals' in batch:
                    hg_loss = self.ce_loss(predictions['home_goals'], batch['home_goals'].to(self.device))
                    loss = loss + self.task_weights['home_goals'] * hg_loss
                    task_losses['home_goals'] += hg_loss.item()
                
                if 'away_goals' in batch:
                    ag_loss = self.ce_loss(predictions['away_goals'], batch['away_goals'].to(self.device))
                    loss = loss + self.task_weights['away_goals'] * ag_loss
                    task_losses['away_goals'] += ag_loss.item()
                
                if 'btts' in batch:
                    btts_loss = self.ce_loss(predictions['btts'], batch['btts'].to(self.device))
                    loss = loss + self.task_weights['btts'] * btts_loss
                    task_losses['btts'] += btts_loss.item()
                
                if 'over_25' in batch:
                    over_loss = self.ce_loss(predictions['over_25'], batch['over_25'].to(self.device))
                    loss = loss + self.task_weights['over_25'] * over_loss
                    task_losses['over_25'] += over_loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            n_batches = len(dataloader) if len(dataloader) > 0 else 1
            return {
                'total_loss': total_loss / n_batches,
                **{k: v / n_batches for k, v in task_losses.items()}
            }
        
        def evaluate(self, dataloader) -> Dict[str, float]:
            """Evaluate model on validation set."""
            self.model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in dataloader:
                    features = batch['features'].to(self.device)
                    predictions = self.model(features)
                    
                    if 'result' in batch:
                        result_loss = self.ce_loss(predictions['result'], batch['result'].to(self.device))
                        total_loss += result_loss.item()
                        
                        pred_labels = predictions['result'].argmax(dim=1)
                        correct += (pred_labels == batch['result'].to(self.device)).sum().item()
                        total += len(batch['result'])
            
            n_batches = len(dataloader) if len(dataloader) > 0 else 1
            return {
                'val_loss': total_loss / n_batches,
                'accuracy': correct / total if total > 0 else 0
            }
        
        def save(self, path: str):
            """Save model weights."""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)
        
        def load(self, path: str):
            """Load model weights."""
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


else:
    # Dummy classes when PyTorch is not available
    class CNNBiLSTMAttention:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for deep learning models. Install with: pip install torch")
    
    class CNNBiLSTMTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for deep learning models. Install with: pip install torch")
