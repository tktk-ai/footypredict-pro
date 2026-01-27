"""Ensemble Models Package."""

from .model_combiner import ModelCombiner, get_combiner
from .meta_learner import MetaLearner, get_meta_learner

# Re-export ModelEnsemble and EnsemblePrediction from parent for compatibility
# These are defined in src/models/ensemble.py (the file, not this package)
try:
    # Import from the ensemble.py file in the models directory
    import importlib.util
    import os
    
    # Get the path to the ensemble.py file
    ensemble_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ensemble.py')
    
    if os.path.exists(ensemble_file):
        spec = importlib.util.spec_from_file_location("ensemble_module", ensemble_file)
        ensemble_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ensemble_module)
        ModelEnsemble = ensemble_module.ModelEnsemble
        EnsemblePrediction = ensemble_module.EnsemblePrediction
    else:
        ModelEnsemble = None
        EnsemblePrediction = None
except Exception:
    ModelEnsemble = None
    EnsemblePrediction = None

__all__ = [
    'ModelCombiner', 'get_combiner',
    'MetaLearner', 'get_meta_learner',
    'ModelEnsemble', 'EnsemblePrediction'
]


