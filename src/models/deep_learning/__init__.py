"""Deep Learning Models Package."""

from .transformer_lstm import TransformerLSTMModel, get_model as get_transformer_lstm
from .temporal_cnn import TemporalCNNModel, get_model as get_temporal_cnn
from .attention_models import AttentionPredictor, get_model as get_attention_model

# Import existing graph neural network if available
try:
    from .graph_neural_network import GraphNeuralNetworkModel
except ImportError:
    GraphNeuralNetworkModel = None

__all__ = [
    'TransformerLSTMModel', 'get_transformer_lstm',
    'TemporalCNNModel', 'get_temporal_cnn',
    'AttentionPredictor', 'get_attention_model',
    'GraphNeuralNetworkModel'
]
