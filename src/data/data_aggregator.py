"""
Multi-Source Data Aggregator
=============================

Aggregates data from all scrapers and generates unified features
for training and predictions.

Data Sources:
1. SportyBet - Odds and fixtures
2. Predictz - Match predictions
3. SofaScore - Live stats, ratings
4. FootyStats - xG, team stats
5. SoccerStats - Goal timing
6. WhoScored - Player ratings
7. FotMob - Momentum, xG
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent


class DataAggregator:
    """Aggregates data from multiple sources into unified features."""
    
    def __init__(self):
        self.data_dir = project_root / "data"
        self.cache = {}
        
        # Initialize scrapers lazily
        self._sportybet = None
        self._multi_source = None
    
    @property
    def sportybet(self):
        """Lazy load SportyBet scraper."""
        if self._sportybet is None:
            from src.data.sportybet_scraper import SportyBetScraper
            self._sportybet = SportyBetScraper()
        return self._sportybet
    
    @property
    def multi_source(self):
        """Lazy load multi-source scraper."""
        if self._multi_source is None:
            from src.data.multi_source_scraper import MultiSourceScraper
            self._multi_source = MultiSourceScraper()
        return self._multi_source
    
    def collect_all_data(self, days: int = 7) -> Dict:
        """
        Collect data from all sources.
        
        Returns:
            Dictionary with data from each source
        """
        logger.info("="*60)
        logger.info("Collecting data from all sources...")
        logger.info("="*60)
        
        data = {
            'collected_at': datetime.now().isoformat(),
            'sources': {}
        }
        
        # 1. SportyBet - Primary source for odds
        logger.info("\n📊 Collecting from SportyBet...")
        try:
            fixtures = self.sportybet.get_all_fixtures(days=days)
            data['sources']['sportybet'] = {
                'fixtures': len(fixtures),
                'data': fixtures
            }
            logger.info(f"   ✓ {len(fixtures)} fixtures with odds")
        except Exception as e:
            logger.error(f"   ✗ SportyBet failed: {e}")
            data['sources']['sportybet'] = {'error': str(e)}
        
        # 2. FotMob - xG and match data
        logger.info("\n📱 Collecting from FotMob...")
        try:
            fotmob_matches = []
            for day in range(min(days, 3)):
                date = (datetime.now() + pd.Timedelta(days=day)).strftime("%Y%m%d")
                matches = self.multi_source.fotmob.get_matches(date)
                fotmob_matches.extend(matches)
            data['sources']['fotmob'] = {
                'matches': len(fotmob_matches),
                'data': fotmob_matches
            }
            logger.info(f"   ✓ {len(fotmob_matches)} matches")
        except Exception as e:
            logger.error(f"   ✗ FotMob failed: {e}")
            data['sources']['fotmob'] = {'error': str(e)}
        
        # 3. SofaScore - Team form and stats
        logger.info("\n📈 Collecting from SofaScore...")
        try:
            sofa_matches = self.multi_source.sofascore.get_upcoming_matches(days)
            data['sources']['sofascore'] = {
                'matches': len(sofa_matches),
                'data': sofa_matches
            }
            logger.info(f"   ✓ {len(sofa_matches)} matches")
        except Exception as e:
            logger.error(f"   ✗ SofaScore failed: {e}")
            data['sources']['sofascore'] = {'error': str(e)}
        
        # 4. Predictz - Match predictions
        logger.info("\n🎯 Collecting from Predictz...")
        try:
            predictions = self.multi_source.predictz.get_predictions("premier-league")
            data['sources']['predictz'] = {
                'predictions': len(predictions),
                'data': predictions
            }
            logger.info(f"   ✓ {len(predictions)} predictions")
        except Exception as e:
            logger.error(f"   ✗ Predictz failed: {e}")
            data['sources']['predictz'] = {'error': str(e)}
        
        logger.info("\n" + "="*60)
        logger.info("Data collection complete!")
        
        return data
    
    def merge_match_data(self, collected_data: Dict) -> pd.DataFrame:
        """
        Merge data from all sources into a unified DataFrame.
        
        Args:
            collected_data: Data from collect_all_data()
            
        Returns:
            Merged DataFrame with all available features
        """
        logger.info("Merging data from all sources...")
        
        # Start with SportyBet fixtures as base
        sportybet_data = collected_data.get('sources', {}).get('sportybet', {}).get('data', [])
        
        if not sportybet_data:
            logger.warning("No SportyBet data found, using other sources")
            # Try other sources
            sofascore_data = collected_data.get('sources', {}).get('sofascore', {}).get('data', [])
            if sofascore_data:
                sportybet_data = sofascore_data
        
        if not sportybet_data:
            logger.error("No fixture data available from any source")
            return pd.DataFrame()
        
        # Create base DataFrame
        records = []
        for fix in sportybet_data:
            record = {
                'event_id': fix.get('event_id', ''),
                'home_team': fix.get('home_team', ''),
                'away_team': fix.get('away_team', ''),
                'league': fix.get('league', ''),
                'date': fix.get('date', ''),
                'time': fix.get('time', ''),
            }
            
            # Add odds if available
            odds = fix.get('odds', {})
            for key, value in odds.items():
                if isinstance(value, (int, float)):
                    record[f'odds_{key}'] = value
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Merge with Predictz predictions
        predictz_data = collected_data.get('sources', {}).get('predictz', {}).get('data', [])
        if predictz_data:
            pred_df = pd.DataFrame(predictz_data)
            if not pred_df.empty:
                # Normalize team names for matching
                df['_home_key'] = df['home_team'].str.lower().str.strip()
                pred_df['_home_key'] = pred_df['home_team'].str.lower().str.strip()
                
                # Merge
                merge_cols = [col for col in ['prediction', 'predicted_score'] if col in pred_df.columns]
                if merge_cols:
                    for col in merge_cols:
                        pred_df[f'predictz_{col}'] = pred_df[col]
                    df = df.merge(
                        pred_df[['_home_key'] + [f'predictz_{c}' for c in merge_cols]],
                        on='_home_key',
                        how='left'
                    )
                
                df = df.drop(columns=['_home_key'], errors='ignore')
        
        # Merge with SofaScore data
        sofascore_data = collected_data.get('sources', {}).get('sofascore', {}).get('data', [])
        if sofascore_data:
            sofa_df = pd.DataFrame(sofascore_data)
            if not sofa_df.empty and 'home_team_id' in sofa_df.columns:
                # Add SofaScore IDs for future API calls
                df['_home_key'] = df['home_team'].str.lower().str.strip()
                sofa_df['_home_key'] = sofa_df['home_team'].str.lower().str.strip()
                
                if 'home_team_id' in sofa_df.columns:
                    df = df.merge(
                        sofa_df[['_home_key', 'home_team_id', 'away_team_id']].drop_duplicates(),
                        on='_home_key',
                        how='left'
                    )
                
                df = df.drop(columns=['_home_key'], errors='ignore')
        
        logger.info(f"Merged {len(df)} matches with {len(df.columns)} columns")
        
        return df
    
    def generate_features(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features from merged data using advanced feature generator.
        
        Args:
            merged_data: DataFrame from merge_match_data()
            
        Returns:
            DataFrame with all generated features
        """
        from src.features.advanced_sportybet_features import AdvancedSportyBetFeatures
        
        logger.info("Generating features from merged data...")
        
        generator = AdvancedSportyBetFeatures()
        
        all_features = []
        for idx, row in merged_data.iterrows():
            # Convert row to fixture format
            fixture = row.to_dict()
            
            # Extract odds into nested dict
            odds = {}
            for col in row.index:
                if col.startswith('odds_'):
                    odds[col.replace('odds_', '')] = row[col]
            fixture['odds'] = odds
            
            # Generate features
            features = generator.generate_all_features(fixture)
            
            # Add metadata
            features['event_id'] = row.get('event_id', '')
            features['home_team'] = row.get('home_team', '')
            features['away_team'] = row.get('away_team', '')
            features['league'] = row.get('league', '')
            features['date'] = row.get('date', '')
            
            # Add Predictz features if available
            if 'predictz_prediction' in row:
                pred = row['predictz_prediction']
                features['predictz_home'] = 1.0 if pred == '1' else 0.0
                features['predictz_draw'] = 1.0 if pred == 'X' else 0.0
                features['predictz_away'] = 1.0 if pred == '2' else 0.0
            
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} matches")
        
        return features_df
    
    def run_full_pipeline(self, days: int = 7, save: bool = True) -> pd.DataFrame:
        """
        Run the complete data collection and feature generation pipeline.
        
        Args:
            days: Number of days to collect
            save: Whether to save results to file
            
        Returns:
            DataFrame with all features ready for prediction/training
        """
        # Collect data
        collected_data = self.collect_all_data(days)
        
        # Merge sources
        merged_data = self.merge_match_data(collected_data)
        
        if merged_data.empty:
            logger.error("No data to process")
            return pd.DataFrame()
        
        # Generate features
        features_df = self.generate_features(merged_data)
        
        # Save if requested
        if save:
            output_dir = self.data_dir / "aggregated"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_path = output_dir / f"aggregated_features_{timestamp}.csv"
            features_df.to_csv(output_path, index=False)
            logger.info(f"Saved features to {output_path}")
            
            # Also save as latest
            latest_path = output_dir / "latest_features.csv"
            features_df.to_csv(latest_path, index=False)
        
        return features_df


def main():
    """CLI for data aggregator."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Multi-Source Data Aggregator')
    parser.add_argument('--days', type=int, default=7, help='Days ahead to collect')
    parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    args = parser.parse_args()
    
    aggregator = DataAggregator()
    features_df = aggregator.run_full_pipeline(
        days=args.days,
        save=not args.no_save
    )
    
    if not features_df.empty:
        print("\n" + "="*60)
        print("📊 Data Aggregation Summary")
        print("="*60)
        print(f"Total matches: {len(features_df)}")
        print(f"Total features: {len(features_df.columns)}")
        print(f"\nSample features:")
        for col in list(features_df.columns)[:10]:
            print(f"  - {col}")
        print("  ...")


if __name__ == "__main__":
    main()
