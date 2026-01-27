"""Ensemble Models Package."""

from .model_combiner import ModelCombiner, get_combiner
from .meta_learner import MetaLearner, get_meta_learner

# Re-export ModelEnsemble from parent for compatibility
try:
    from src.models.ensemble_base import ModelEnsemble
except ImportError:
    try:
        # Create a simple alias if the original is at models.ensemble
        ModelEnsemble = None
    except:
        ModelEnsemble = None

__all__ = [
    'ModelCombiner', 'get_combiner',
    'MetaLearner', 'get_meta_learner',
    'ModelEnsemble'
]

