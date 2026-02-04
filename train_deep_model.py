"""
Deep Model Training Script V3.0
Train CNN-BiLSTM-Attention model on historical match data

Usage:
    python train_deep_model.py --epochs 50 --batch-size 64
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.engineering.advanced_features import AdvancedFeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from src.models.deep_learning.cnn_bilstm_attention import CNNBiLSTMAttention, CNNBiLSTMTrainer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


class MatchDataset(Dataset):
    """PyTorch dataset for match data."""
    
    def __init__(self, features: np.ndarray, labels: Dict[str, np.ndarray]):
        self.features = torch.FloatTensor(features)
        self.labels = {k: torch.LongTensor(v) for k, v in labels.items()}
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        item = {'features': self.features[idx]}
        for k, v in self.labels.items():
            item[k] = v[idx]
        return item


def load_training_data(data_path: str = 'data') -> pd.DataFrame:
    """
    Load historical match data for training.
    Supports multiple data formats.
    """
    logger.info(f"Loading training data from {data_path}")
    
    # Priority order of data files to try
    possible_files = [
        os.path.join(data_path, 'training_data.csv'),
        os.path.join(data_path, 'matches.csv'),
        os.path.join(data_path, 'all_matches.csv'),
        os.path.join(data_path, 'historical', 'matches.csv'),
        'data/training_data.csv',
        'matches.csv'
    ]
    
    for filepath in possible_files:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} matches from {filepath}")
            
            # Standardize column names
            column_mapping = {
                'home_score': 'home_goals',
                'away_score': 'away_goals',
                'date': 'match_date',
                'tournament': 'league'
            }
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Add result column if not present
            if 'result' not in df.columns and 'home_goals' in df.columns:
                df['result'] = df.apply(
                    lambda x: 'H' if x['home_goals'] > x['away_goals'] 
                              else ('A' if x['home_goals'] < x['away_goals'] else 'D'),
                    axis=1
                )
            
            # Add derived features if missing
            if 'home_xg' not in df.columns:
                # Estimate xG from goals with slight variance
                df['home_xg'] = df['home_goals'] + np.random.normal(0, 0.3, len(df))
                df['home_xg'] = df['home_xg'].clip(0, 6)
                
            if 'away_xg' not in df.columns:
                df['away_xg'] = df['away_goals'] + np.random.normal(0, 0.3, len(df))
                df['away_xg'] = df['away_xg'].clip(0, 6)
            
            # Estimate shots from goals (average ~10 shots per goal)
            if 'home_shots' not in df.columns:
                df['home_shots'] = (df['home_goals'] * 10 + np.random.poisson(5, len(df))).astype(int)
                
            if 'away_shots' not in df.columns:
                df['away_shots'] = (df['away_goals'] * 10 + np.random.poisson(4, len(df))).astype(int)
            
            # Possession (random if not available)
            if 'home_possession' not in df.columns:
                df['home_possession'] = np.random.uniform(35, 65, len(df))
                df['away_possession'] = 100 - df['home_possession']
            
            # Half-time goals (estimate ~42% in first half)
            if 'home_goals_ht' not in df.columns:
                df['home_goals_ht'] = (df['home_goals'] * np.random.uniform(0.3, 0.5, len(df))).astype(int)
                df['home_goals_ht'] = df.apply(lambda x: min(x['home_goals_ht'], x['home_goals']), axis=1)
                
            if 'away_goals_ht' not in df.columns:
                df['away_goals_ht'] = (df['away_goals'] * np.random.uniform(0.3, 0.5, len(df))).astype(int)
                df['away_goals_ht'] = df.apply(lambda x: min(x['away_goals_ht'], x['away_goals']), axis=1)
            
            # Filter to recent data for better relevance
            if 'match_date' in df.columns:
                try:
                    df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
                    # Use last 10 years of data
                    cutoff = pd.Timestamp.now() - pd.DateOffset(years=10)
                    original_len = len(df)
                    df = df[df['match_date'] >= cutoff]
                    if len(df) < 1000:
                        # Not enough recent data, use more
                        df = pd.read_csv(filepath)
                        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                    else:
                        logger.info(f"Filtered to {len(df)} matches (last 10 years from {original_len})")
                except Exception as e:
                    logger.warning(f"Could not filter by date: {e}")
            
            logger.info(f"Final dataset: {len(df)} matches with {len(df.columns)} columns")
            return df
    
    # No data found - raise error instead of using synthetic
    logger.error("No training data found! Please provide data in data/training_data.csv")
    logger.error("Expected columns: date, home_team, away_team, home_score, away_score, tournament")
    raise FileNotFoundError(
        "Training data not found. Please place your match data in data/training_data.csv"
    )


def prepare_features(df: pd.DataFrame, sequence_length: int = 10) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Prepare features and labels for training.
    """
    logger.info("Creating advanced features...")
    
    # Create advanced features
    engineer = AdvancedFeatureEngineer(df)
    df = engineer.create_all_features()
    
    # Select numeric features
    feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    feature_cols = [col for col in feature_cols if col not in [
        'home_goals', 'away_goals', 'home_goals_ht', 'away_goals_ht'
    ]]
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Fill NaN values
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Normalize features
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - mean) / std
    
    # Create sequences
    features_list = []
    labels_result = []
    labels_home_goals = []
    labels_away_goals = []
    labels_btts = []
    labels_over25 = []
    
    for i in range(sequence_length, len(df)):
        # Get sequence of features
        seq = df[feature_cols].iloc[i-sequence_length:i].values
        features_list.append(seq)
        
        # Labels
        result_map = {'H': 0, 'D': 1, 'A': 2}
        labels_result.append(result_map.get(df['result'].iloc[i], 1))
        labels_home_goals.append(min(df['home_goals'].iloc[i], 7))
        labels_away_goals.append(min(df['away_goals'].iloc[i], 7))
        labels_btts.append(int((df['home_goals'].iloc[i] > 0) & (df['away_goals'].iloc[i] > 0)))
        labels_over25.append(int((df['home_goals'].iloc[i] + df['away_goals'].iloc[i]) > 2.5))
    
    features = np.array(features_list, dtype=np.float32)
    labels = {
        'result': np.array(labels_result, dtype=np.int64),
        'home_goals': np.array(labels_home_goals, dtype=np.int64),
        'away_goals': np.array(labels_away_goals, dtype=np.int64),
        'btts': np.array(labels_btts, dtype=np.int64),
        'over_25': np.array(labels_over25, dtype=np.int64)
    }
    
    logger.info(f"Created {len(features)} training sequences")
    return features, labels


def train_model(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    sequence_length: int = 10,
    save_path: str = 'models/cnn_bilstm_best.pt'
):
    """
    Train the CNN-BiLSTM-Attention model.
    """
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Cannot train model.")
        return None
    
    logger.info("=" * 60)
    logger.info("Starting CNN-BiLSTM-Attention Training")
    logger.info("=" * 60)
    
    # Load and prepare data
    df = load_training_data()
    features, labels = prepare_features(df, sequence_length)
    
    # Split data
    split_idx = int(len(features) * 0.8)
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_labels = {k: v[:split_idx] for k, v in labels.items()}
    val_labels = {k: v[split_idx:] for k, v in labels.items()}
    
    logger.info(f"Training samples: {len(train_features)}")
    logger.info(f"Validation samples: {len(val_features)}")
    
    # Create datasets
    train_dataset = MatchDataset(train_features, train_labels)
    val_dataset = MatchDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = features.shape[2]
    logger.info(f"Input dimension: {input_dim}")
    
    model = CNNBiLSTMAttention(
        input_dim=input_dim,
        cnn_filters=128,
        lstm_hidden=128,
        lstm_layers=2,
        attention_heads=8,
        dropout=0.3,
        num_classes=3
    )
    
    trainer = CNNBiLSTMTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=1e-5
    )
    
    # Training loop
    best_accuracy = 0
    history = {'train_loss': [], 'val_loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        
        history['train_loss'].append(train_metrics['total_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['accuracy'].append(val_metrics['accuracy'])
        
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Accuracy: {val_metrics['accuracy']:.2%}"
            )
        
        # Save best model
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trainer.save(save_path)
            logger.info(f"✅ New best model saved! Accuracy: {best_accuracy:.2%}")
    
    logger.info("=" * 60)
    logger.info(f"Training complete! Best accuracy: {best_accuracy:.2%}")
    logger.info(f"Model saved to: {save_path}")
    logger.info("=" * 60)
    
    # Save training history
    history_path = save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description='Train CNN-BiLSTM-Attention model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=10, help='Sequence length')
    parser.add_argument('--save-path', type=str, default='models/cnn_bilstm_best.pt', help='Model save path')
    
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=args.sequence_length,
        save_path=args.save_path
    )


if __name__ == '__main__':
    main()
