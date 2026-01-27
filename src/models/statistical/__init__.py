"""Statistical Models Package."""

# Try to import existing models
try:
    from src.dixon_coles import DixonColesModel
except ImportError:
    DixonColesModel = None

try:
    from src.bivariate_poisson import BivariatePoissonModel
except ImportError:
    BivariatePoissonModel = None

from .dynamic_poisson import DynamicPoissonModel, get_model as get_dynamic_poisson
from .bayesian_hierarchical import BayesianHierarchicalModel, get_model as get_bayesian

__all__ = [
    'DixonColesModel',
    'BivariatePoissonModel', 
    'DynamicPoissonModel', 'get_dynamic_poisson',
    'BayesianHierarchicalModel', 'get_bayesian'
]
