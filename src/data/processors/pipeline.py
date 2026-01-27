"""
Data Pipeline Orchestrator

Handles the flow of data from collection through processing to feature engineering.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    collectors: List[str] = field(default_factory=lambda: ['football_data', 'fbref'])
    cache_enabled: bool = True
    cache_path: Path = field(default_factory=lambda: Path('data/cache'))
    output_path: Path = field(default_factory=lambda: Path('data/processed'))
    validate_data: bool = True
    clean_data: bool = True


class DataPipeline:
    """
    Orchestrates the data flow from collection to feature engineering.
    
    Pipeline stages:
    1. Collection - Fetch from various sources
    2. Validation - Check data quality
    3. Cleaning - Standardize and clean
    4. Storage - Save processed data
    5. Feature Engineering - Create features
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self._collectors = {}
        self._processors = {}
        self._feature_engineers = {}
        self._cached_data = {}
        
    def register_collector(self, name: str, collector):
        """Register a data collector."""
        self._collectors[name] = collector
        logger.info(f"Registered collector: {name}")
        
    def register_processor(self, name: str, processor):
        """Register a data processor."""
        self._processors[name] = processor
        logger.info(f"Registered processor: {name}")
        
    def register_feature_engineer(self, name: str, engineer):
        """Register a feature engineer."""
        self._feature_engineers[name] = engineer
        logger.info(f"Registered feature engineer: {name}")
    
    def run(
        self,
        sources: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        leagues: List[str] = None
    ) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Args:
            sources: Data sources to use (defaults to config)
            start_date: Start date for data
            end_date: End date for data
            leagues: Leagues to include
            
        Returns:
            Processed DataFrame with features
        """
        sources = sources or self.config.collectors
        
        logger.info(f"Running pipeline with sources: {sources}")
        
        # Stage 1: Collection
        collected_data = self._collect_data(sources, start_date, end_date, leagues)
        
        if collected_data.empty:
            logger.warning("No data collected")
            return pd.DataFrame()
        
        # Stage 2: Validation
        if self.config.validate_data:
            collected_data = self._validate_data(collected_data)
        
        # Stage 3: Cleaning
        if self.config.clean_data:
            collected_data = self._clean_data(collected_data)
        
        # Stage 4: Feature Engineering
        processed_data = self._engineer_features(collected_data)
        
        # Stage 5: Storage
        self._save_data(processed_data)
        
        logger.info(f"Pipeline complete: {len(processed_data)} rows processed")
        return processed_data
    
    def _collect_data(
        self,
        sources: List[str],
        start_date: str,
        end_date: str,
        leagues: List[str]
    ) -> pd.DataFrame:
        """Collect data from specified sources."""
        all_data = []
        
        for source in sources:
            if source in self._collectors:
                try:
                    collector = self._collectors[source]
                    data = collector.collect(
                        start_date=start_date,
                        end_date=end_date,
                        leagues=leagues
                    )
                    if data is not None and len(data) > 0:
                        data['source'] = source
                        all_data.append(data)
                        logger.info(f"Collected {len(data)} rows from {source}")
                except Exception as e:
                    logger.error(f"Error collecting from {source}: {e}")
            else:
                logger.warning(f"Collector not registered: {source}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate collected data."""
        if 'validator' in self._processors:
            return self._processors['validator'].validate(df)
        
        # Basic validation
        initial_len = len(df)
        
        # Remove rows with missing critical columns
        critical_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
        existing_critical = [c for c in critical_cols if c in df.columns]
        
        if existing_critical:
            df = df.dropna(subset=existing_critical)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Validation removed {removed} rows")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data."""
        if 'cleaner' in self._processors:
            return self._processors['cleaner'].clean(df)
        
        # Basic cleaning
        # Standardize team names
        if 'home_team' in df.columns:
            df['home_team'] = df['home_team'].str.strip().str.title()
        if 'away_team' in df.columns:
            df['away_team'] = df['away_team'].str.strip().str.title()
        
        # Convert dates
        if 'match_date' in df.columns:
            df['match_date'] = pd.to_datetime(df['match_date'], errors='coerce')
        
        # Sort by date
        if 'match_date' in df.columns:
            df = df.sort_values('match_date').reset_index(drop=True)
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        for name, engineer in self._feature_engineers.items():
            try:
                df = engineer.create_features(df)
                logger.info(f"Applied feature engineering: {name}")
            except Exception as e:
                logger.error(f"Error in feature engineering {name}: {e}")
        
        return df
    
    def _save_data(self, df: pd.DataFrame):
        """Save processed data."""
        if not self.config.output_path.exists():
            self.config.output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.config.output_path / f'processed_data_{timestamp}.csv'
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
    
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if available."""
        if self.config.cache_enabled and key in self._cached_data:
            return self._cached_data[key]
        return None
    
    def cache_data(self, key: str, data: pd.DataFrame):
        """Cache data for reuse."""
        if self.config.cache_enabled:
            self._cached_data[key] = data


# Global pipeline instance
_pipeline: Optional[DataPipeline] = None


def get_pipeline() -> DataPipeline:
    """Get the global data pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
    return _pipeline


def run_pipeline(**kwargs) -> pd.DataFrame:
    """Convenience function to run the pipeline."""
    return get_pipeline().run(**kwargs)
