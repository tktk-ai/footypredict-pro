"""Features Engineering Package."""

# Core advanced features
from .advanced_features import AdvancedFeatureEngineer, create_advanced_features

# New feature modules
try:
    from .team_features import TeamFeatureGenerator, get_generator as get_team_generator
except ImportError:
    TeamFeatureGenerator = None
    get_team_generator = None

try:
    from .player_features import PlayerFeatureGenerator, get_generator as get_player_generator
except ImportError:
    PlayerFeatureGenerator = None
    get_player_generator = None

try:
    from .momentum_features import MomentumFeatureGenerator, get_generator as get_momentum_generator
except ImportError:
    MomentumFeatureGenerator = None
    get_momentum_generator = None

try:
    from .advanced_metrics import AdvancedMetrics, get_metrics
except ImportError:
    AdvancedMetrics = None
    get_metrics = None

try:
    from .embeddings import TeamEmbeddings, get_embeddings
except ImportError:
    TeamEmbeddings = None
    get_embeddings = None

__all__ = [
    'AdvancedFeatureEngineer', 'create_advanced_features',
    'TeamFeatureGenerator', 'get_team_generator',
    'PlayerFeatureGenerator', 'get_player_generator',
    'MomentumFeatureGenerator', 'get_momentum_generator',
    'AdvancedMetrics', 'get_metrics',
    'TeamEmbeddings', 'get_embeddings'
]

