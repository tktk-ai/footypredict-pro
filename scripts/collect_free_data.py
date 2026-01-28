#!/usr/bin/env python3
"""
Data Collection Script - Free Sources Only
Collects historical match data from all free sources for V4.0 training
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import collectors
from src.data.enhanced_collectors import (
    FootballDataCollector,
    FBrefCollector,
    StatsBombCollector,
    ClubEloCollector,
    OpenLigaDBCollector,
    ESPNCollector,
    EnhancedDataAggregator,
)

# Configuration
LEAGUES = [
    'ENG-Premier League',
    'ESP-La Liga',
    'GER-Bundesliga',
    'ITA-Serie A',
    'FRA-Ligue 1',
    'ENG-Championship',
    'NED-Eredivisie',
    'POR-Primeira Liga',
]

SEASONS = ['2022-2023', '2023-2024', '2024-2025']

OUTPUT_DIR = Path('data/collected')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_football_data():
    """Collect from Football-Data.co.uk (main source)."""
    logger.info("=" * 60)
    logger.info("COLLECTING FROM FOOTBALL-DATA.CO.UK")
    logger.info("=" * 60)
    
    collector = FootballDataCollector()
    all_data = []
    
    for league in LEAGUES:
        for season in SEASONS:
            try:
                df = collector.collect(league, season)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  ✅ {league} {season}: {len(df)} matches")
                else:
                    logger.warning(f"  ⚠️ {league} {season}: No data")
            except Exception as e:
                logger.error(f"  ❌ {league} {season}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / 'football_data.parquet')
        logger.info(f"\n📊 Football-Data: {len(combined)} total matches")
        return combined
    
    return pd.DataFrame()


def collect_clubelo():
    """Collect Elo ratings from ClubElo."""
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTING ELO RATINGS FROM CLUBELO")
    logger.info("=" * 60)
    
    try:
        collector = ClubEloCollector()
        df = collector.collect()
        
        if not df.empty:
            df.to_parquet(OUTPUT_DIR / 'clubelo_ratings.parquet')
            logger.info(f"  ✅ Collected {len(df)} team Elo ratings")
            return df
    except Exception as e:
        logger.error(f"  ❌ ClubElo failed: {e}")
    
    return pd.DataFrame()


def collect_openligadb():
    """Collect from OpenLigaDB (German leagues)."""
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTING FROM OPENLIGADB")
    logger.info("=" * 60)
    
    collector = OpenLigaDBCollector()
    all_data = []
    
    german_leagues = ['GER-Bundesliga', 'GER-2. Bundesliga']
    
    for league in german_leagues:
        for season in SEASONS:
            try:
                df = collector.collect(league, season)
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"  ✅ {league} {season}: {len(df)} matches")
            except Exception as e:
                logger.error(f"  ❌ {league} {season}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / 'openligadb.parquet')
        logger.info(f"\n📊 OpenLigaDB: {len(combined)} total matches")
        return combined
    
    return pd.DataFrame()


def collect_statsbomb():
    """Collect from StatsBomb free data."""
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTING FROM STATSBOMB (FREE DATA)")
    logger.info("=" * 60)
    
    collector = StatsBombCollector()
    all_data = []
    
    for comp_name in collector.FREE_COMPETITIONS.keys():
        try:
            df = collector.collect(comp_name)
            if not df.empty:
                all_data.append(df)
                logger.info(f"  ✅ {comp_name}: {len(df)} matches")
        except Exception as e:
            logger.error(f"  ❌ {comp_name}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / 'statsbomb.parquet')
        logger.info(f"\n📊 StatsBomb: {len(combined)} total matches")
        return combined
    
    return pd.DataFrame()


def collect_espn():
    """Collect from ESPN API."""
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTING FROM ESPN")
    logger.info("=" * 60)
    
    collector = ESPNCollector()
    all_data = []
    
    for league in LEAGUES[:5]:  # Top 5 leagues
        try:
            df = collector.collect(league, '2024-2025')
            if not df.empty:
                all_data.append(df)
                logger.info(f"  ✅ {league}: {len(df)} events")
        except Exception as e:
            logger.error(f"  ❌ {league}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_parquet(OUTPUT_DIR / 'espn.parquet')
        logger.info(f"\n📊 ESPN: {len(combined)} total events")
        return combined
    
    return pd.DataFrame()


def merge_all_data():
    """Merge all collected data."""
    logger.info("\n" + "=" * 60)
    logger.info("MERGING ALL DATA SOURCES")
    logger.info("=" * 60)
    
    data_files = list(OUTPUT_DIR.glob('*.parquet'))
    logger.info(f"Found {len(data_files)} data files")
    
    # Use Football-Data as primary
    primary_file = OUTPUT_DIR / 'football_data.parquet'
    if primary_file.exists():
        merged = pd.read_parquet(primary_file)
        logger.info(f"Primary source: {len(merged)} matches")
        
        # Add Elo ratings if available
        elo_file = OUTPUT_DIR / 'clubelo_ratings.parquet'
        if elo_file.exists():
            elo_df = pd.read_parquet(elo_file)
            logger.info(f"Elo ratings available for {len(elo_df)} teams")
            # Could merge Elo here if needed
        
        # Save merged data
        merged.to_parquet(OUTPUT_DIR / 'merged_training_data.parquet')
        logger.info(f"\n✅ Merged data saved: {len(merged)} matches")
        
        return merged
    
    return pd.DataFrame()


def generate_features(data: pd.DataFrame):
    """Generate 698 features from collected data."""
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING 698 FEATURES")
    logger.info("=" * 60)
    
    try:
        from src.features.enhanced_engineering import EnhancedFeatureGenerator
        
        generator = EnhancedFeatureGenerator(data)
        features = generator.generate_all_features()
        
        # Combine features with target variables
        if 'home_goals' in data.columns and 'away_goals' in data.columns:
            features['home_goals'] = data['home_goals'].values
            features['away_goals'] = data['away_goals'].values
            features['total_goals'] = features['home_goals'] + features['away_goals']
            features['result'] = np.where(
                features['home_goals'] > features['away_goals'], 0,  # Home win
                np.where(features['home_goals'] < features['away_goals'], 2, 1)  # Away win or Draw
            )
        
        # Save features
        features.to_parquet(OUTPUT_DIR / 'training_features.parquet')
        logger.info(f"✅ Generated {len(features.columns)} features for {len(features)} matches")
        
        return features
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def main():
    """Main data collection pipeline."""
    print("=" * 70)
    print("V4.0 DATA COLLECTION - FREE SOURCES")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Leagues: {len(LEAGUES)}")
    print(f"Seasons: {SEASONS}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    
    # 1. Collect from all free sources
    fd_data = collect_football_data()
    elo_data = collect_clubelo()
    openliga_data = collect_openligadb()
    sb_data = collect_statsbomb()
    espn_data = collect_espn()
    
    # 2. Merge data
    merged_data = merge_all_data()
    
    # 3. Generate features (if we have enough data)
    if len(merged_data) >= 100:
        features = generate_features(merged_data)
    else:
        logger.warning("Not enough data for feature generation")
        features = pd.DataFrame()
    
    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Football-Data: {len(fd_data)} matches")
    print(f"ClubElo: {len(elo_data)} ratings")
    print(f"OpenLigaDB: {len(openliga_data)} matches")
    print(f"StatsBomb: {len(sb_data)} matches")
    print(f"ESPN: {len(espn_data)} events")
    print(f"Merged: {len(merged_data)} matches")
    print(f"Features: {len(features.columns) if not features.empty else 0} columns")
    print(f"End time: {datetime.now()}")
    print("=" * 70)
    
    return merged_data, features


if __name__ == "__main__":
    data, features = main()
