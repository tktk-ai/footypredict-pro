"""
Graph Neural Network (GNN) for Football Prediction
Uses team relationships and match context as a graph structure.

Based on the blueprint for advanced deep learning models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Check for torch_geometric
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logger.warning("torch_geometric not installed. GNN features limited.")


class TeamEmbedding(nn.Module):
    """Learnable team embeddings."""
    
    def __init__(self, num_teams: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_teams, embedding_dim)
        
    def forward(self, team_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(team_ids)


class MatchGraphEncoder(nn.Module):
    """
    Encode match context using graph neural networks.
    
    Nodes: Teams
    Edges: Recent matches between teams
    Node features: Team statistics
    Edge features: Match statistics
    """
    
    def __init__(
        self,
        node_features: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if not HAS_TORCH_GEOMETRIC:
            # Fallback to simple MLP
            self.use_gnn = False
            self.fallback = nn.Sequential(
                nn.Linear(node_features * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
            return
        
        self.use_gnn = True
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(node_features, hidden_dim, heads=4, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        batch: torch.Tensor = None
    ) -> torch.Tensor:
        
        if not self.use_gnn or edge_index is None:
            # Fallback
            return self.fallback(x) if hasattr(self, 'fallback') else x
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = self.dropout(x)
        
        x = self.convs[-1](x, edge_index)
        
        return x


class GraphFootballPredictor(nn.Module):
    """
    Complete GNN-based football prediction model.
    
    Architecture:
    1. Team embeddings
    2. Graph encoder for league context
    3. Match predictor head
    """
    
    def __init__(
        self,
        num_teams: int = 1000,
        team_embed_dim: int = 64,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Team embeddings
        self.team_embedding = TeamEmbedding(num_teams, team_embed_dim)
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, team_embed_dim)
        )
        
        # Graph encoder
        self.graph_encoder = MatchGraphEncoder(
            node_features=team_embed_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim // 2,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # Match representation
        match_dim = hidden_dim // 2 + team_embed_dim * 2
        
        # Prediction heads
        # 1X2 Result
        self.result_head = nn.Sequential(
            nn.Linear(match_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        # Goals prediction
        self.home_goals_head = nn.Sequential(
            nn.Linear(match_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8)  # 0-7 goals
        )
        
        self.away_goals_head = nn.Sequential(
            nn.Linear(match_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        
        # BTTS
        self.btts_head = nn.Sequential(
            nn.Linear(match_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # Over 2.5
        self.over25_head = nn.Sequential(
            nn.Linear(match_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(
        self,
        home_team_id: torch.Tensor,
        away_team_id: torch.Tensor,
        match_features: torch.Tensor,
        edge_index: torch.Tensor = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # Get team embeddings
        home_embed = self.team_embedding(home_team_id)
        away_embed = self.team_embedding(away_team_id)
        
        # Encode features
        encoded_features = self.feature_encoder(match_features)
        
        # Combine for graph
        combined = torch.cat([home_embed, away_embed], dim=-1)
        
        # Graph encoding
        if edge_index is not None:
            graph_out = self.graph_encoder(combined, edge_index)
        else:
            graph_out = self.graph_encoder(combined)
        
        # Match representation
        match_repr = torch.cat([
            graph_out,
            home_embed,
            away_embed
        ], dim=-1)
        
        # Predictions
        result = F.softmax(self.result_head(match_repr), dim=-1)
        home_goals = F.softmax(self.home_goals_head(match_repr), dim=-1)
        away_goals = F.softmax(self.away_goals_head(match_repr), dim=-1)
        btts = F.softmax(self.btts_head(match_repr), dim=-1)
        over25 = F.softmax(self.over25_head(match_repr), dim=-1)
        
        output = {
            'result': result,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'btts': btts,
            'over_25': over25
        }
        
        if return_embeddings:
            output['home_embedding'] = home_embed
            output['away_embedding'] = away_embed
            output['match_representation'] = match_repr
        
        return output
    
    def predict(self, home_team_id: int, away_team_id: int, features: np.ndarray) -> Dict:
        """Generate predictions for a single match."""
        self.eval()
        
        with torch.no_grad():
            home_id = torch.tensor([home_team_id])
            away_id = torch.tensor([away_team_id])
            feat_tensor = torch.FloatTensor(features).unsqueeze(0)
            
            output = self.forward(home_id, away_id, feat_tensor)
            
            # Calculate correct scores
            home_probs = output['home_goals'].squeeze().cpu().numpy()
            away_probs = output['away_goals'].squeeze().cpu().numpy()
            
            correct_scores = {}
            for h in range(8):
                for a in range(8):
                    correct_scores[f'{h}-{a}'] = float(home_probs[h] * away_probs[a])
            
            # Normalize
            total = sum(correct_scores.values())
            if total > 0:
                correct_scores = {k: v/total for k, v in correct_scores.items()}
            
            return {
                'result': {
                    'home_win': float(output['result'][0, 0]),
                    'draw': float(output['result'][0, 1]),
                    'away_win': float(output['result'][0, 2])
                },
                'correct_scores': dict(sorted(
                    correct_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                'btts_yes': float(output['btts'][0, 1]),
                'over_25': float(output['over_25'][0, 1])
            }


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for sequence prediction.
    Processes team's recent match history.
    """
    
    def __init__(
        self,
        feature_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 50, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.result_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3)
        )
        
        self.goals_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 16)  # 8 home + 8 away
        )
        
    def forward(
        self,
        home_sequence: torch.Tensor,  # (batch, seq_len, feature_dim)
        away_sequence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len, _ = home_sequence.shape
        
        # Project inputs
        home_proj = self.input_proj(home_sequence)
        away_proj = self.input_proj(away_sequence)
        
        # Add positional encoding
        home_proj = home_proj + self.pos_encoding[:, :seq_len, :]
        away_proj = away_proj + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        home_encoded = self.transformer(home_proj)
        away_encoded = self.transformer(away_proj)
        
        # Pool (mean over sequence)
        home_pooled = home_encoded.mean(dim=1)
        away_pooled = away_encoded.mean(dim=1)
        
        # Combine
        combined = torch.cat([home_pooled, away_pooled], dim=-1)
        
        # Predictions
        result = F.softmax(self.result_head(combined), dim=-1)
        goals = self.goals_head(combined)
        
        home_goals = F.softmax(goals[:, :8], dim=-1)
        away_goals = F.softmax(goals[:, 8:], dim=-1)
        
        return {
            'result': result,
            'home_goals': home_goals,
            'away_goals': away_goals
        }


# Factory functions
def get_gnn_model(num_teams: int = 1000, feature_dim: int = 128) -> GraphFootballPredictor:
    """Get GNN model instance."""
    return GraphFootballPredictor(
        num_teams=num_teams,
        feature_dim=feature_dim
    )


def get_transformer_model(feature_dim: int = 128) -> TransformerPredictor:
    """Get Transformer model instance."""
    return TransformerPredictor(feature_dim=feature_dim)
