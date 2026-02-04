"""
Embeddings Module
Creates team and player embeddings for deep learning models.

Part of the complete blueprint implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TeamEmbeddings:
    """
    Creates and manages team embeddings.
    
    Features:
    - Learnable embeddings
    - Pre-trained loading
    - Similarity calculations
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        num_teams: int = 500
    ):
        self.embedding_dim = embedding_dim
        self.num_teams = num_teams
        self.team_to_idx = {}
        self.idx_to_team = {}
        
        if TORCH_AVAILABLE:
            self.embeddings = nn.Embedding(num_teams, embedding_dim)
        else:
            self.embeddings = np.random.randn(num_teams, embedding_dim) * 0.1
    
    def register_team(self, team: str) -> int:
        """Register a team and get its index."""
        if team in self.team_to_idx:
            return self.team_to_idx[team]
        
        idx = len(self.team_to_idx)
        if idx >= self.num_teams:
            logger.warning(f"Max teams ({self.num_teams}) reached")
            return 0
        
        self.team_to_idx[team] = idx
        self.idx_to_team[idx] = team
        return idx
    
    def get_embedding(self, team: str) -> np.ndarray:
        """Get embedding vector for a team."""
        idx = self.team_to_idx.get(team)
        if idx is None:
            idx = self.register_team(team)
        
        if TORCH_AVAILABLE:
            with torch.no_grad():
                idx_tensor = torch.tensor([idx])
                return self.embeddings(idx_tensor).numpy()[0]
        else:
            return self.embeddings[idx]
    
    def get_match_embedding(
        self,
        home_team: str,
        away_team: str
    ) -> np.ndarray:
        """Get combined embedding for a match."""
        home_emb = self.get_embedding(home_team)
        away_emb = self.get_embedding(away_team)
        
        # Concatenate home and away embeddings
        return np.concatenate([home_emb, away_emb])
    
    def get_similarity(self, team1: str, team2: str) -> float:
        """Calculate cosine similarity between teams."""
        emb1 = self.get_embedding(team1)
        emb2 = self.get_embedding(team2)
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def find_similar_teams(
        self,
        team: str,
        n: int = 5
    ) -> List[Tuple[str, float]]:
        """Find most similar teams."""
        target_emb = self.get_embedding(team)
        similarities = []
        
        for other_team in self.team_to_idx:
            if other_team != team:
                sim = self.get_similarity(team, other_team)
                similarities.append((other_team, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def save(self, path: str):
        """Save embeddings and mappings."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save mappings
        with open(path / 'team_mapping.json', 'w') as f:
            json.dump(self.team_to_idx, f)
        
        # Save embeddings
        if TORCH_AVAILABLE:
            torch.save(self.embeddings.state_dict(), path / 'embeddings.pt')
        else:
            np.save(path / 'embeddings.npy', self.embeddings)
    
    def load(self, path: str):
        """Load embeddings and mappings."""
        path = Path(path)
        
        # Load mappings
        mapping_file = path / 'team_mapping.json'
        if mapping_file.exists():
            with open(mapping_file) as f:
                self.team_to_idx = json.load(f)
                self.idx_to_team = {v: k for k, v in self.team_to_idx.items()}
        
        # Load embeddings
        if TORCH_AVAILABLE:
            pt_file = path / 'embeddings.pt'
            if pt_file.exists():
                self.embeddings.load_state_dict(torch.load(pt_file, weights_only=True))
        else:
            npy_file = path / 'embeddings.npy'
            if npy_file.exists():
                self.embeddings = np.load(npy_file)


class PositionalEncoding:
    """Positional encoding for sequence models."""
    
    def __init__(self, d_model: int, max_len: int = 100):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._create_encoding()
    
    def _create_encoding(self) -> np.ndarray:
        """Create positional encoding matrix."""
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Add positional encoding to input."""
        seq_len = x.shape[0] if len(x.shape) >= 1 else 1
        return x + self.pe[:seq_len]


class MatchSequenceEmbedding:
    """Creates embeddings for sequences of matches."""
    
    def __init__(
        self,
        match_dim: int = 32,
        seq_len: int = 10
    ):
        self.match_dim = match_dim
        self.seq_len = seq_len
        self.pos_encoding = PositionalEncoding(match_dim, seq_len)
    
    def encode_match_result(
        self,
        goals_for: int,
        goals_against: int
    ) -> np.ndarray:
        """Encode a single match result."""
        features = np.zeros(self.match_dim)
        
        # Basic features
        features[0] = goals_for
        features[1] = goals_against
        features[2] = goals_for - goals_against
        features[3] = 1 if goals_for > goals_against else (0.5 if goals_for == goals_against else 0)
        features[4] = 1 if goals_for > 0 and goals_against > 0 else 0  # BTTS
        features[5] = 1 if goals_for + goals_against > 2.5 else 0  # Over 2.5
        
        return features
    
    def encode_match_sequence(
        self,
        matches: List[Dict]
    ) -> np.ndarray:
        """Encode a sequence of matches."""
        sequence = np.zeros((self.seq_len, self.match_dim))
        
        for i, match in enumerate(matches[-self.seq_len:]):
            idx = self.seq_len - len(matches[-self.seq_len:]) + i
            sequence[idx] = self.encode_match_result(
                match.get('goals_for', 0),
                match.get('goals_against', 0)
            )
        
        # Add positional encoding
        sequence = self.pos_encoding.encode(sequence)
        
        return sequence


# Global instances
_team_embeddings: Optional[TeamEmbeddings] = None


def get_team_embeddings() -> TeamEmbeddings:
    """Get or create team embeddings."""
    global _team_embeddings
    if _team_embeddings is None:
        _team_embeddings = TeamEmbeddings()
    return _team_embeddings


def get_match_embedding(home_team: str, away_team: str) -> np.ndarray:
    """Get embedding for a match."""
    return get_team_embeddings().get_match_embedding(home_team, away_team)
