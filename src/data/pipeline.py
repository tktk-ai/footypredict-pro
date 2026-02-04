"""
Data Pipeline Module
Orchestrates data collection, processing, and storage.

Part of the complete blueprint implementation.
"""

import pandas as pd
from typing import Dict, List, Optional, Callable
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates the complete data pipeline:
    
    1. Collection from multiple sources
    2. Cleaning and standardization
    3. Validation
    4. Feature engineering
    5. Storage
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        cache_ttl_hours: int = 24
    ):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.features_dir = self.data_dir / "features"
        
        for d in [self.raw_dir, self.processed_dir, self.features_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.collectors = {}
        self.processors = []
        
    def register_collector(self, name: str, collector: Callable):
        """Register a data collector."""
        self.collectors[name] = collector
        logger.info(f"Registered collector: {name}")
    
    def register_processor(self, processor: Callable):
        """Register a data processor."""
        self.processors.append(processor)
        logger.info(f"Registered processor: {processor.__name__}")
    
    def collect(
        self,
        sources: List[str] = None,
        leagues: List[str] = None,
        seasons: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect data from specified sources.
        
        Args:
            sources: List of collector names to use
            leagues: Leagues to collect
            seasons: Seasons to collect
        """
        sources = sources or list(self.collectors.keys())
        collected = {}
        
        for source in sources:
            if source not in self.collectors:
                logger.warning(f"Unknown source: {source}")
                continue
            
            try:
                logger.info(f"Collecting from {source}...")
                collector = self.collectors[source]
                
                if callable(collector):
                    data = collector(leagues=leagues, seasons=seasons)
                else:
                    data = collector.fetch_all_leagues(seasons, leagues)
                
                if isinstance(data, pd.DataFrame) and not data.empty:
                    collected[source] = data
                    logger.info(f"Collected {len(data)} rows from {source}")
                    
            except Exception as e:
                logger.error(f"Collection failed for {source}: {e}")
        
        return collected
    
    def process(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process and combine data from multiple sources.
        """
        if not data:
            return pd.DataFrame()
        
        # Import processors
        try:
            from .processors.data_cleaner import get_cleaner
            from .processors.data_validator import get_validator
            
            cleaner = get_cleaner()
            validator = get_validator()
        except ImportError:
            cleaner = None
            validator = None
        
        processed_dfs = []
        
        for source, df in data.items():
            logger.info(f"Processing data from {source} ({len(df)} rows)")
            
            # Clean
            if cleaner:
                df = cleaner.clean_dataframe(df)
                df = cleaner.remove_duplicates(df)
            
            # Validate
            if validator:
                result = validator.validate(df)
                if not result.is_valid:
                    logger.warning(f"Validation errors for {source}: {result.errors}")
                if result.warnings:
                    logger.info(f"Validation warnings for {source}: {result.warnings[:3]}")
            
            df['data_source'] = source
            processed_dfs.append(df)
        
        if not processed_dfs:
            return pd.DataFrame()
        
        # Combine all sources
        combined = pd.concat(processed_dfs, ignore_index=True)
        
        # Run custom processors
        for processor in self.processors:
            try:
                combined = processor(combined)
            except Exception as e:
                logger.error(f"Processor {processor.__name__} failed: {e}")
        
        logger.info(f"Combined dataset: {len(combined)} rows")
        return combined
    
    def save(
        self,
        df: pd.DataFrame,
        name: str,
        directory: str = "processed",
        format: str = "parquet"
    ) -> Path:
        """Save DataFrame to disk."""
        dir_map = {
            'raw': self.raw_dir,
            'processed': self.processed_dir,
            'features': self.features_dir
        }
        
        save_dir = dir_map.get(directory, self.processed_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"{name}_{timestamp}.{format}"
        filepath = save_dir / filename
        
        if format == 'parquet':
            df.to_parquet(filepath, index=False)
        elif format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records')
        
        logger.info(f"Saved {len(df)} rows to {filepath}")
        return filepath
    
    def load(
        self,
        name: str,
        directory: str = "processed",
        latest: bool = True
    ) -> Optional[pd.DataFrame]:
        """Load DataFrame from disk."""
        dir_map = {
            'raw': self.raw_dir,
            'processed': self.processed_dir,
            'features': self.features_dir
        }
        
        load_dir = dir_map.get(directory, self.processed_dir)
        
        # Find matching files
        files = list(load_dir.glob(f"{name}*"))
        
        if not files:
            return None
        
        if latest:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        filepath = files[0]
        
        if filepath.suffix == '.parquet':
            return pd.read_parquet(filepath)
        elif filepath.suffix == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            return pd.read_json(filepath)
        
        return None
    
    def run_full_pipeline(
        self,
        sources: List[str] = None,
        leagues: List[str] = None,
        seasons: List[str] = None,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        1. Collect from sources
        2. Process and clean
        3. Validate
        4. Save
        
        Returns processed DataFrame.
        """
        logger.info("Starting full data pipeline...")
        
        # Collect
        raw_data = self.collect(sources, leagues, seasons)
        
        if not raw_data:
            logger.warning("No data collected")
            return pd.DataFrame()
        
        # Process
        processed = self.process(raw_data)
        
        if processed.empty:
            logger.warning("Processing resulted in empty DataFrame")
            return processed
        
        # Save
        if save_results:
            self.save(processed, "matches")
        
        logger.info(f"Pipeline complete: {len(processed)} matches")
        return processed
    
    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """Get the latest processed data."""
        return self.load("matches", "processed", latest=True)
    
    def update_incremental(self) -> pd.DataFrame:
        """Incrementally update with new data."""
        existing = self.get_latest_data()
        
        # Get latest date
        if existing is not None and 'match_date' in existing.columns:
            latest_date = existing['match_date'].max()
            logger.info(f"Existing data up to {latest_date}")
        else:
            latest_date = None
        
        # Collect new data
        new_data = self.run_full_pipeline(save_results=False)
        
        if new_data.empty:
            return existing if existing is not None else pd.DataFrame()
        
        # Filter to new matches only
        if latest_date and 'match_date' in new_data.columns:
            new_data = new_data[new_data['match_date'] > latest_date]
        
        # Combine
        if existing is not None and not new_data.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
            self.save(combined, "matches")
            return combined
        
        return new_data


# Global instance
_pipeline: Optional[DataPipeline] = None


def get_pipeline() -> DataPipeline:
    """Get or create data pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = DataPipeline()
        
        # Register collectors
        try:
            from .collectors.football_data import get_collector
            _pipeline.register_collector('football_data', get_collector())
        except ImportError:
            pass
        
        try:
            from .collectors.fbref_scraper import get_scraper
            _pipeline.register_collector('fbref', get_scraper())
        except ImportError:
            pass
    
    return _pipeline


def run_pipeline(
    sources: List[str] = None,
    leagues: List[str] = None
) -> pd.DataFrame:
    """Quick function to run data pipeline."""
    return get_pipeline().run_full_pipeline(sources, leagues)
