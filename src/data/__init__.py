"""
Data Collection Module

Unified interface for collecting football data from multiple sources.
"""

from .kaggle_collector import KaggleDataCollector, collect_kaggle_data
from .huggingface_collector import HuggingFaceCollector, collect_huggingface_data
from .github_collector import GitHubCollector, collect_github_data
from .data_merger import DataMerger, merge_all_data

__all__ = [
    'KaggleDataCollector',
    'HuggingFaceCollector', 
    'GitHubCollector',
    'DataMerger',
    'collect_kaggle_data',
    'collect_huggingface_data',
    'collect_github_data',
    'merge_all_data'
]
