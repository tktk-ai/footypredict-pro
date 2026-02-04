"""
Attention Visualization
=======================
Visualize attention weights from deep learning models for interpretability.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention visualization."""
    output_dir: str = "visualizations"
    colormap: str = "viridis"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100


class AttentionVisualizer:
    """
    Visualizes attention weights from transformer and attention-based models.
    
    Supports:
    - Self-attention heatmaps
    - Feature importance from attention
    - Multi-head attention analysis
    - Temporal attention patterns
    """
    
    def __init__(self, config: AttentionConfig = None):
        self.config = config or AttentionConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._has_matplotlib = False
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check if visualization dependencies are available."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            self._has_matplotlib = True
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Visualization will be limited.")
            self._has_matplotlib = False
    
    def extract_attention_weights(
        self,
        model: Any,
        input_data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights from a model.
        
        Args:
            model: Model with attention layers
            input_data: Input data to process
            
        Returns:
            Dict of layer names to attention weight matrices
        """
        attention_weights = {}
        
        # Try to extract from common model types
        if hasattr(model, 'attention_weights'):
            # Direct access
            attention_weights['main'] = model.attention_weights
            
        elif hasattr(model, 'get_attention_weights'):
            # Method access
            attention_weights = model.get_attention_weights(input_data)
            
        elif hasattr(model, 'layers'):
            # Iterate through layers (Keras/TF style)
            for i, layer in enumerate(model.layers):
                if 'attention' in layer.name.lower():
                    if hasattr(layer, 'attention_scores'):
                        attention_weights[f'layer_{i}_{layer.name}'] = layer.attention_scores
        
        elif hasattr(model, 'named_modules'):
            # PyTorch style
            for name, module in model.named_modules():
                if 'attention' in name.lower():
                    if hasattr(module, 'attention_weights'):
                        attention_weights[name] = module.attention_weights.detach().cpu().numpy()
        
        return attention_weights
    
    def visualize_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        row_labels: List[str] = None,
        col_labels: List[str] = None,
        title: str = "Attention Weights",
        save_path: str = None
    ) -> Optional[str]:
        """
        Create a heatmap visualization of attention weights.
        
        Args:
            attention_weights: 2D attention matrix
            row_labels: Labels for rows (query tokens)
            col_labels: Labels for columns (key tokens)
            title: Plot title
            save_path: Optional path to save the figure
            
        Returns:
            Path to saved figure or None
        """
        if not self._has_matplotlib:
            logger.warning("Cannot create heatmap: matplotlib not available")
            return self._save_attention_data(attention_weights, save_path or "attention_heatmap.json")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create heatmap
        sns.heatmap(
            attention_weights,
            xticklabels=col_labels if col_labels else False,
            yticklabels=row_labels if row_labels else False,
            cmap=self.config.colormap,
            annot=True if attention_weights.shape[0] <= 10 else False,
            fmt='.2f',
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / f"attention_heatmap_{id(attention_weights)}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Attention heatmap saved to {save_path}")
        return str(save_path)
    
    def visualize_multihead_attention(
        self,
        attention_weights: np.ndarray,
        head_names: List[str] = None,
        title: str = "Multi-Head Attention",
        save_path: str = None
    ) -> Optional[str]:
        """
        Visualize multi-head attention as multiple heatmaps.
        
        Args:
            attention_weights: 3D array (heads, query, key)
            head_names: Names for each attention head
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        if not self._has_matplotlib:
            return self._save_attention_data(attention_weights, save_path or "multihead_attention.json")
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_heads = attention_weights.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes)
        
        for i in range(n_heads):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            head_name = head_names[i] if head_names and i < len(head_names) else f"Head {i+1}"
            
            sns.heatmap(
                attention_weights[i],
                cmap=self.config.colormap,
                ax=ax,
                cbar=False
            )
            ax.set_title(head_name)
        
        # Hide unused subplots
        for i in range(n_heads, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "multihead_attention.png"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def get_feature_importance_from_attention(
        self,
        attention_weights: np.ndarray,
        feature_names: List[str]
    ) -> List[Dict[str, float]]:
        """
        Extract feature importance from attention weights.
        
        Args:
            attention_weights: Attention matrix
            feature_names: Names of input features
            
        Returns:
            Sorted list of feature importance
        """
        # Average attention over all queries
        if attention_weights.ndim == 3:
            # Multi-head: average over heads first
            avg_attention = attention_weights.mean(axis=0).mean(axis=0)
        else:
            avg_attention = attention_weights.mean(axis=0)
        
        # Normalize
        avg_attention = avg_attention / avg_attention.sum() if avg_attention.sum() > 0 else avg_attention
        
        # Create importance list
        importance = []
        for i, name in enumerate(feature_names):
            if i < len(avg_attention):
                importance.append({
                    'feature': name,
                    'attention_score': float(avg_attention[i]),
                    'rank': 0
                })
        
        # Sort and assign ranks
        importance.sort(key=lambda x: x['attention_score'], reverse=True)
        for i, item in enumerate(importance):
            item['rank'] = i + 1
        
        return importance
    
    def visualize_temporal_attention(
        self,
        attention_over_time: List[np.ndarray],
        timestamps: List[str] = None,
        save_path: str = None
    ) -> Optional[str]:
        """
        Visualize how attention changes over time/sequence.
        
        Args:
            attention_over_time: List of attention matrices at each timestep
            timestamps: Labels for each timestep
            save_path: Path to save figure
            
        Returns:
            Path to saved figure
        """
        if not self._has_matplotlib:
            return None
        
        import matplotlib.pyplot as plt
        
        n_steps = len(attention_over_time)
        
        if timestamps is None:
            timestamps = [f"t={i}" for i in range(n_steps)]
        
        # Get average attention per step
        avg_attention = [w.mean() for w in attention_over_time]
        max_attention = [w.max() for w in attention_over_time]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        x = range(n_steps)
        ax.plot(x, avg_attention, 'b-o', label='Average Attention')
        ax.plot(x, max_attention, 'r-s', label='Max Attention')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Temporal Attention Pattern')
        ax.set_xticks(x)
        ax.set_xticklabels(timestamps, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "temporal_attention.png"
        
        plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def visualize_attention_for_match(
        self,
        model: Any,
        match_features: Dict[str, float],
        feature_names: List[str],
        title: str = "Match Prediction Attention"
    ) -> Dict[str, Any]:
        """
        Visualize attention for a specific match prediction.
        
        Args:
            model: Model to extract attention from
            match_features: Feature dict for the match
            feature_names: Names of features
            title: Visualization title
            
        Returns:
            Dict with attention analysis
        """
        # Prepare input
        input_array = np.array([match_features.get(f, 0) for f in feature_names]).reshape(1, -1)
        
        # Extract attention
        attention = self.extract_attention_weights(model, input_array)
        
        if not attention:
            return {'error': 'Could not extract attention weights'}
        
        # Get main attention
        main_attention = list(attention.values())[0]
        
        # Get feature importance
        importance = self.get_feature_importance_from_attention(main_attention, feature_names)
        
        # Create visualization
        heatmap_path = None
        if len(main_attention.shape) == 2:
            heatmap_path = self.visualize_attention_heatmap(
                main_attention,
                row_labels=feature_names[:main_attention.shape[0]],
                col_labels=feature_names[:main_attention.shape[1]],
                title=title
            )
        
        return {
            'top_features': importance[:10],
            'attention_stats': {
                'mean': float(main_attention.mean()),
                'max': float(main_attention.max()),
                'std': float(main_attention.std())
            },
            'visualization_path': heatmap_path
        }
    
    def _save_attention_data(
        self,
        attention_weights: np.ndarray,
        filename: str
    ) -> str:
        """Save attention data as JSON when visualization not available."""
        filepath = self.output_dir / filename
        
        data = {
            'shape': list(attention_weights.shape),
            'mean': float(attention_weights.mean()),
            'max': float(attention_weights.max()),
            'min': float(attention_weights.min()),
            'data': attention_weights.tolist() if attention_weights.size < 1000 else 'Too large to save'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(filepath)
    
    def compare_attention_patterns(
        self,
        attention_a: np.ndarray,
        attention_b: np.ndarray,
        name_a: str = "Model A",
        name_b: str = "Model B"
    ) -> Dict[str, Any]:
        """
        Compare attention patterns between two models/predictions.
        
        Returns:
            Comparison statistics
        """
        # Correlation
        corr = np.corrcoef(attention_a.flatten(), attention_b.flatten())[0, 1]
        
        # Difference stats
        diff = attention_a - attention_b
        
        return {
            'correlation': round(float(corr), 4),
            'mean_difference': round(float(diff.mean()), 4),
            'max_difference': round(float(np.abs(diff).max()), 4),
            'similar': corr > 0.8
        }


# Global instance
_visualizer: Optional[AttentionVisualizer] = None


def get_visualizer() -> AttentionVisualizer:
    """Get or create attention visualizer."""
    global _visualizer
    if _visualizer is None:
        _visualizer = AttentionVisualizer()
    return _visualizer


def visualize_attention(
    attention_weights: np.ndarray,
    labels: List[str] = None,
    title: str = "Attention Weights"
) -> Optional[str]:
    """Quick function to visualize attention weights."""
    return get_visualizer().visualize_attention_heatmap(
        attention_weights,
        row_labels=labels,
        col_labels=labels,
        title=title
    )


def get_attention_importance(
    attention_weights: np.ndarray,
    feature_names: List[str]
) -> List[Dict]:
    """Quick function to get feature importance from attention."""
    return get_visualizer().get_feature_importance_from_attention(
        attention_weights,
        feature_names
    )
