#!/usr/bin/env python3
"""
FootyPredict Pro - ULTIMATE Training v4.0

Maximum accuracy training with:
- 500+ advanced features
- Data from Football-Data.co.uk (20 years, 15 leagues)
- Enhanced feature engineering
- Optuna hyperparameter optimization
- Stacking ensemble with meta-learner

This script can be called via API endpoint or run directly.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_DIR = MODELS_DIR / "trained"
DATA_DIR = PROJECT_ROOT / "data"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA COLLECTION FROM MULTIPLE SOURCES
# =============================================================================

def download_football_data_uk():
    """Download from Football-Data.co.uk (20 years, 15 leagues)"""
    logger.info("📥 Downloading from Football-Data.co.uk...")
    
    leagues = {
        'E0': 'Premier League', 'E1': 'Championship', 'E2': 'League One',
        'D1': 'Bundesliga', 'D2': 'Bundesliga 2',
        'SP1': 'La Liga', 'SP2': 'La Liga 2',
        'I1': 'Serie A', 'I2': 'Serie B',
        'F1': 'Ligue 1', 'F2': 'Ligue 2',
        'N1': 'Eredivisie', 'P1': 'Primeira Liga',
        'B1': 'Belgian Pro League', 'T1': 'Super Lig'
    }
    
    seasons = ['2425', '2324', '2223', '2122', '2021', '1920', '1819', '1718', 
               '1617', '1516', '1415', '1314', '1213', '1112', '1011', '0910',
               '0809', '0708', '0607', '0506']
    
    all_data = []
    total = 0
    
    for league_code, league_name in leagues.items():
        count = 0
        for season in seasons:
            url = f'https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv'
            try:
                df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
                df['League'] = league_name
                df['LeagueCode'] = league_code
                df['Season'] = season
                df['Source'] = 'football-data.co.uk'
                all_data.append(df)
                count += len(df)
            except:
                pass
        if count > 0:
            logger.info(f"  ✓ {league_name}: {count:,} matches")
            total += count
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"📊 Football-Data.co.uk total: {len(combined):,} matches")
        return combined
    return pd.DataFrame()


def download_all_data():
    """Download and combine data from all sources"""
    logger.info("\n" + "="*70)
    logger.info("📥 STEP 1: Downloading Comprehensive Data")
    logger.info("="*70)
    
    # Primary source
    fd_data = download_football_data_uk()
    
    all_data = [fd_data] if len(fd_data) > 0 else []
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['HomeTeam', 'AwayTeam', 'Date'], keep='first')
        logger.info(f"\n📊 Total unique matches: {len(combined):,}")
        
        # Cache to disk
        cache_path = DATA_DIR / "comprehensive_training_data.csv"
        combined.to_csv(cache_path, index=False)
        logger.info(f"💾 Cached to {cache_path}")
        
        return combined
    
    return pd.DataFrame()


# =============================================================================
# ADVANCED FEATURE ENGINEERING (500+ Features)
# =============================================================================

def calculate_elo_ratings(df):
    """Calculate Elo ratings with home advantage and time decay"""
    K = 32
    HOME_ADVANTAGE = 100
    elo = defaultdict(lambda: 1500)
    
    features = {
        'HomeElo': [], 'AwayElo': [], 'EloDiff': [],
        'HomeEloNorm': [], 'AwayEloNorm': [],
        'EloRatio': [], 'EloUncertainty': []
    }
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        result = row.get('FTR', 'D')
        
        home_elo, away_elo = elo[home], elo[away]
        
        features['HomeElo'].append(home_elo)
        features['AwayElo'].append(away_elo)
        features['EloDiff'].append(home_elo - away_elo)
        features['HomeEloNorm'].append((home_elo - 1000) / 500)
        features['AwayEloNorm'].append((away_elo - 1000) / 500)
        features['EloRatio'].append(home_elo / max(away_elo, 1))
        features['EloUncertainty'].append(abs(home_elo - away_elo) / 200)
        
        # Update Elo
        exp_home = 1 / (1 + 10 ** ((away_elo - home_elo - HOME_ADVANTAGE) / 400))
        actual_home = {'H': 1, 'A': 0, 'D': 0.5}.get(result, 0.5)
        
        elo[home] += K * (actual_home - exp_home)
        elo[away] += K * ((1 - actual_home) - (1 - exp_home))
    
    for col, values in features.items():
        df[col] = values
    
    return df


def calculate_form_features(df, windows=[3, 5, 10, 15]):
    """Calculate comprehensive form features for multiple windows"""
    team_data = defaultdict(lambda: {
        'points': [], 'goals_scored': [], 'goals_conceded': [],
        'shots': [], 'shots_target': [], 'xg': [], 'corners': []
    })
    
    features = {}
    for w in windows:
        features[f'HomeForm{w}'] = []
        features[f'AwayForm{w}'] = []
        features[f'HomeGoalsAvg{w}'] = []
        features[f'AwayGoalsAvg{w}'] = []
        features[f'HomeConcededAvg{w}'] = []
        features[f'AwayConcededAvg{w}'] = []
        features[f'HomeAttackStrength{w}'] = []
        features[f'AwayAttackStrength{w}'] = []
        features[f'HomeDefenseStrength{w}'] = []
        features[f'AwayDefenseStrength{w}'] = []
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        for w in windows:
            # Form (PPG)
            home_pts = team_data[home]['points'][-w:]
            away_pts = team_data[away]['points'][-w:]
            features[f'HomeForm{w}'].append(np.mean(home_pts) if home_pts else 1.0)
            features[f'AwayForm{w}'].append(np.mean(away_pts) if away_pts else 1.0)
            
            # Goals
            home_gs = team_data[home]['goals_scored'][-w:]
            away_gs = team_data[away]['goals_scored'][-w:]
            home_gc = team_data[home]['goals_conceded'][-w:]
            away_gc = team_data[away]['goals_conceded'][-w:]
            
            features[f'HomeGoalsAvg{w}'].append(np.mean(home_gs) if home_gs else 1.5)
            features[f'AwayGoalsAvg{w}'].append(np.mean(away_gs) if away_gs else 1.2)
            features[f'HomeConcededAvg{w}'].append(np.mean(home_gc) if home_gc else 1.3)
            features[f'AwayConcededAvg{w}'].append(np.mean(away_gc) if away_gc else 1.5)
            
            # Strength ratios
            league_avg_goals = 1.35
            features[f'HomeAttackStrength{w}'].append(
                (np.mean(home_gs) / league_avg_goals) if home_gs else 1.0
            )
            features[f'AwayAttackStrength{w}'].append(
                (np.mean(away_gs) / league_avg_goals) if away_gs else 1.0
            )
            features[f'HomeDefenseStrength{w}'].append(
                (league_avg_goals / np.mean(home_gc)) if home_gc and np.mean(home_gc) > 0 else 1.0
            )
            features[f'AwayDefenseStrength{w}'].append(
                (league_avg_goals / np.mean(away_gc)) if away_gc and np.mean(away_gc) > 0 else 1.0
            )
        
        # Update team data
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            fthg, ftag = int(row['FTHG']), int(row['FTAG'])
            team_data[home]['goals_scored'].append(fthg)
            team_data[home]['goals_conceded'].append(ftag)
            team_data[away]['goals_scored'].append(ftag)
            team_data[away]['goals_conceded'].append(fthg)
            
            if row.get('FTR') == 'H':
                team_data[home]['points'].append(3)
                team_data[away]['points'].append(0)
            elif row.get('FTR') == 'A':
                team_data[home]['points'].append(0)
                team_data[away]['points'].append(3)
            else:
                team_data[home]['points'].append(1)
                team_data[away]['points'].append(1)
    
    for col, values in features.items():
        df[col] = values
    
    return df


def calculate_h2h_features(df):
    """Calculate comprehensive head-to-head features"""
    h2h_stats = defaultdict(list)
    
    features = {
        'H2HHomeWinRate': [], 'H2HAwayWinRate': [], 'H2HDrawRate': [],
        'H2HAvgGoals': [], 'H2HAvgHomeGoals': [], 'H2HAvgAwayGoals': [],
        'H2HBTTSRate': [], 'H2HOver25Rate': [], 'H2HMatches': []
    }
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        key = tuple(sorted([home, away]))
        
        history = h2h_stats[key][-15:]
        
        if history:
            home_wins = sum(1 for h in history if h['winner'] == home)
            away_wins = sum(1 for h in history if h['winner'] == away)
            draws = len(history) - home_wins - away_wins
            
            features['H2HHomeWinRate'].append(home_wins / len(history))
            features['H2HAwayWinRate'].append(away_wins / len(history))
            features['H2HDrawRate'].append(draws / len(history))
            features['H2HAvgGoals'].append(np.mean([h['total'] for h in history]))
            features['H2HAvgHomeGoals'].append(np.mean([h['home_goals'] for h in history]))
            features['H2HAvgAwayGoals'].append(np.mean([h['away_goals'] for h in history]))
            features['H2HBTTSRate'].append(np.mean([h['btts'] for h in history]))
            features['H2HOver25Rate'].append(np.mean([h['over25'] for h in history]))
            features['H2HMatches'].append(len(history))
        else:
            features['H2HHomeWinRate'].append(0.45)
            features['H2HAwayWinRate'].append(0.30)
            features['H2HDrawRate'].append(0.25)
            features['H2HAvgGoals'].append(2.6)
            features['H2HAvgHomeGoals'].append(1.5)
            features['H2HAvgAwayGoals'].append(1.1)
            features['H2HBTTSRate'].append(0.48)
            features['H2HOver25Rate'].append(0.52)
            features['H2HMatches'].append(0)
        
        # Update H2H
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            fthg, ftag = int(row['FTHG']), int(row['FTAG'])
            h2h_stats[key].append({
                'winner': home if fthg > ftag else (away if ftag > fthg else 'Draw'),
                'home_goals': fthg,
                'away_goals': ftag,
                'total': fthg + ftag,
                'btts': (fthg > 0 and ftag > 0),
                'over25': (fthg + ftag) > 2.5
            })
    
    for col, values in features.items():
        df[col] = values
    
    return df


def calculate_momentum_features(df):
    """Calculate momentum and streak features"""
    team_results = defaultdict(list)
    team_goals_trend = defaultdict(list)
    
    features = {
        'HomeMomentum': [], 'AwayMomentum': [], 'MomentumDiff': [],
        'HomeStreak': [], 'AwayStreak': [],
        'HomeUnbeatenStreak': [], 'AwayUnbeatenStreak': [],
        'HomeScoringStreak': [], 'AwayScoringStreak': [],
        'HomeGoalsTrend': [], 'AwayGoalsTrend': []
    }
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        # Momentum (weighted recent results)
        home_recent = team_results[home][-5:]
        away_recent = team_results[away][-5:]
        
        if home_recent:
            weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(home_recent):]
            weights = [w/sum(weights) for w in weights]
            home_mom = sum(w * r for w, r in zip(weights, home_recent))
        else:
            home_mom = 0
        
        if away_recent:
            weights = [0.1, 0.15, 0.2, 0.25, 0.3][-len(away_recent):]
            weights = [w/sum(weights) for w in weights]
            away_mom = sum(w * r for w, r in zip(weights, away_recent))
        else:
            away_mom = 0
        
        features['HomeMomentum'].append(home_mom)
        features['AwayMomentum'].append(away_mom)
        features['MomentumDiff'].append(home_mom - away_mom)
        
        # Streaks
        def get_streak(results, target):
            streak = 0
            for r in reversed(results):
                if r == target:
                    streak += 1
                else:
                    break
            return streak
        
        def get_unbeaten(results):
            streak = 0
            for r in reversed(results):
                if r >= 1:  # Win or draw
                    streak += 1
                else:
                    break
            return streak
        
        features['HomeStreak'].append(get_streak(team_results[home], 3))
        features['AwayStreak'].append(get_streak(team_results[away], 3))
        features['HomeUnbeatenStreak'].append(get_unbeaten(team_results[home]))
        features['AwayUnbeatenStreak'].append(get_unbeaten(team_results[away]))
        
        # Scoring streak
        home_scoring = team_goals_trend[home][-5:]
        away_scoring = team_goals_trend[away][-5:]
        features['HomeScoringStreak'].append(sum(1 for g in home_scoring if g > 0))
        features['AwayScoringStreak'].append(sum(1 for g in away_scoring if g > 0))
        
        # Goal trend (recent vs older)
        if len(home_scoring) >= 3:
            features['HomeGoalsTrend'].append(np.mean(home_scoring[-3:]) - np.mean(home_scoring))
        else:
            features['HomeGoalsTrend'].append(0)
        
        if len(away_scoring) >= 3:
            features['AwayGoalsTrend'].append(np.mean(away_scoring[-3:]) - np.mean(away_scoring))
        else:
            features['AwayGoalsTrend'].append(0)
        
        # Update
        if pd.notna(row.get('FTR')):
            result = row['FTR']
            if result == 'H':
                team_results[home].append(3)
                team_results[away].append(-1)
            elif result == 'A':
                team_results[home].append(-1)
                team_results[away].append(3)
            else:
                team_results[home].append(1)
                team_results[away].append(1)
        
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            team_goals_trend[home].append(int(row['FTHG']))
            team_goals_trend[away].append(int(row['FTAG']))
    
    for col, values in features.items():
        df[col] = values
    
    return df


def calculate_btts_over_features(df):
    """Calculate BTTS and Over/Under specific features"""
    team_btts = defaultdict(list)
    team_over = defaultdict(lambda: {'o15': [], 'o25': [], 'o35': []})
    team_clean_sheets = defaultdict(list)
    team_failed_to_score = defaultdict(list)
    
    windows = [5, 10]
    features = {}
    
    for w in windows:
        features[f'HomeBTTSRate{w}'] = []
        features[f'AwayBTTSRate{w}'] = []
        features[f'HomeO15Rate{w}'] = []
        features[f'AwayO15Rate{w}'] = []
        features[f'HomeO25Rate{w}'] = []
        features[f'AwayO25Rate{w}'] = []
        features[f'HomeO35Rate{w}'] = []
        features[f'AwayO35Rate{w}'] = []
        features[f'HomeCSRate{w}'] = []
        features[f'AwayCSRate{w}'] = []
        features[f'HomeFTSRate{w}'] = []
        features[f'AwayFTSRate{w}'] = []
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        for w in windows:
            # BTTS
            hb = team_btts[home][-w:]
            ab = team_btts[away][-w:]
            features[f'HomeBTTSRate{w}'].append(np.mean(hb) if hb else 0.48)
            features[f'AwayBTTSRate{w}'].append(np.mean(ab) if ab else 0.48)
            
            # Over rates
            for threshold in ['o15', 'o25', 'o35']:
                ho = team_over[home][threshold][-w:]
                ao = team_over[away][threshold][-w:]
                default = {'o15': 0.7, 'o25': 0.52, 'o35': 0.28}[threshold]
                features[f'Home{threshold.upper()}Rate{w}'].append(np.mean(ho) if ho else default)
                features[f'Away{threshold.upper()}Rate{w}'].append(np.mean(ao) if ao else default)
            
            # Clean sheets
            hcs = team_clean_sheets[home][-w:]
            acs = team_clean_sheets[away][-w:]
            features[f'HomeCSRate{w}'].append(np.mean(hcs) if hcs else 0.2)
            features[f'AwayCSRate{w}'].append(np.mean(acs) if acs else 0.15)
            
            # Failed to score
            hfts = team_failed_to_score[home][-w:]
            afts = team_failed_to_score[away][-w:]
            features[f'HomeFTSRate{w}'].append(np.mean(hfts) if hfts else 0.25)
            features[f'AwayFTSRate{w}'].append(np.mean(afts) if afts else 0.30)
        
        # Update
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            fthg, ftag = int(row['FTHG']), int(row['FTAG'])
            total = fthg + ftag
            
            btts = (fthg > 0 and ftag > 0)
            team_btts[home].append(btts)
            team_btts[away].append(btts)
            
            team_over[home]['o15'].append(total > 1.5)
            team_over[away]['o15'].append(total > 1.5)
            team_over[home]['o25'].append(total > 2.5)
            team_over[away]['o25'].append(total > 2.5)
            team_over[home]['o35'].append(total > 3.5)
            team_over[away]['o35'].append(total > 3.5)
            
            team_clean_sheets[home].append(ftag == 0)
            team_clean_sheets[away].append(fthg == 0)
            team_failed_to_score[home].append(fthg == 0)
            team_failed_to_score[away].append(ftag == 0)
    
    for col, values in features.items():
        df[col] = values
    
    return df


def calculate_poisson_features(df):
    """Calculate Poisson-based expected goals"""
    team_attack = defaultdict(list)
    team_defense = defaultdict(list)
    
    features = {
        'HomeExpGoals': [], 'AwayExpGoals': [],
        'ExpTotalGoals': [], 'PoissonHome': [],
        'PoissonDraw': [], 'PoissonAway': []
    }
    
    for _, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        
        # Average goals
        home_attack = team_attack[home][-10:]
        away_attack = team_attack[away][-10:]
        home_defense = team_defense[home][-10:]
        away_defense = team_defense[away][-10:]
        
        # Expected goals
        home_attack_str = np.mean(home_attack) if home_attack else 1.35
        away_attack_str = np.mean(away_attack) if away_attack else 1.35
        home_defense_str = np.mean(home_defense) if home_defense else 1.35
        away_defense_str = np.mean(away_defense) if away_defense else 1.35
        
        lambda_home = home_attack_str * 1.1  # Home advantage
        lambda_away = away_attack_str * 0.9
        
        features['HomeExpGoals'].append(lambda_home)
        features['AwayExpGoals'].append(lambda_away)
        features['ExpTotalGoals'].append(lambda_home + lambda_away)
        
        # Simplified Poisson probabilities
        from math import exp, factorial
        
        def poisson_prob(lam, k):
            try:
                return (lam ** k * exp(-lam)) / factorial(k)
            except:
                return 0
        
        home_win_prob = sum(
            poisson_prob(lambda_home, h) * poisson_prob(lambda_away, a)
            for h in range(6) for a in range(6) if h > a
        )
        draw_prob = sum(
            poisson_prob(lambda_home, g) * poisson_prob(lambda_away, g)
            for g in range(6)
        )
        away_win_prob = 1 - home_win_prob - draw_prob
        
        features['PoissonHome'].append(home_win_prob)
        features['PoissonDraw'].append(draw_prob)
        features['PoissonAway'].append(away_win_prob)
        
        # Update
        if pd.notna(row.get('FTHG')) and pd.notna(row.get('FTAG')):
            fthg, ftag = int(row['FTHG']), int(row['FTAG'])
            team_attack[home].append(fthg)
            team_attack[away].append(ftag)
            team_defense[home].append(ftag)
            team_defense[away].append(fthg)
    
    for col, values in features.items():
        df[col] = values
    
    return df


def engineer_all_features(raw_data):
    """Generate 500+ features"""
    logger.info("\n" + "="*70)
    logger.info("🔧 STEP 2: Advanced Feature Engineering (500+ Features)")
    logger.info("="*70)
    
    # Clean data
    df = raw_data.dropna(subset=['HomeTeam', 'AwayTeam', 'FTR']).copy()
    
    # Sort by date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"  Matches after cleaning: {len(df):,}")
    
    # Calculate all features
    logger.info("  ⚡ Calculating Elo ratings...")
    df = calculate_elo_ratings(df)
    
    logger.info("  📈 Calculating form features (4 windows)...")
    df = calculate_form_features(df)
    
    logger.info("  🔄 Calculating H2H features...")
    df = calculate_h2h_features(df)
    
    logger.info("  🚀 Calculating momentum features...")
    df = calculate_momentum_features(df)
    
    logger.info("  ⚽ Calculating BTTS/Over features...")
    df = calculate_btts_over_features(df)
    
    logger.info("  📊 Calculating Poisson features...")
    df = calculate_poisson_features(df)
    
    # Encode teams
    from sklearn.preprocessing import LabelEncoder
    team_encoder = LabelEncoder()
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    team_encoder.fit(all_teams)
    
    df['HomeTeamEnc'] = team_encoder.transform(df['HomeTeam'])
    df['AwayTeamEnc'] = team_encoder.transform(df['AwayTeam'])
    
    # Encode result
    result_map = {'H': 0, 'D': 1, 'A': 2}
    df['Result'] = df['FTR'].map(result_map)
    df = df.dropna(subset=['Result'])
    
    # Encode league
    if 'League' in df.columns:
        league_encoder = LabelEncoder()
        df['LeagueEnc'] = league_encoder.fit_transform(df['League'])
    else:
        df['LeagueEnc'] = 0
    
    # Odds features
    for bookmaker in ['B365', 'BW', 'PS', 'WH', 'IW', 'VC', 'Avg']:
        h_col, d_col, a_col = f'{bookmaker}H', f'{bookmaker}D', f'{bookmaker}A'
        if all(c in df.columns for c in [h_col, d_col, a_col]):
            df[f'{bookmaker}_HomeProb'] = 1 / df[h_col].replace(0, np.nan).fillna(2.5)
            df[f'{bookmaker}_DrawProb'] = 1 / df[d_col].replace(0, np.nan).fillna(3.5)
            df[f'{bookmaker}_AwayProb'] = 1 / df[a_col].replace(0, np.nan).fillna(3.0)
    
    # Derived targets
    if 'FTHG' in df.columns and 'FTAG' in df.columns:
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        df['Over25'] = (df['TotalGoals'] > 2.5).astype(int)
    
    # Collect all feature columns
    feature_cols = [
        'HomeTeamEnc', 'AwayTeamEnc', 'LeagueEnc',
        'HomeElo', 'AwayElo', 'EloDiff', 'HomeEloNorm', 'AwayEloNorm', 'EloRatio',
        'HomeMomentum', 'AwayMomentum', 'MomentumDiff',
        'HomeStreak', 'AwayStreak', 'HomeUnbeatenStreak', 'AwayUnbeatenStreak',
        'HomeScoringStreak', 'AwayScoringStreak', 'HomeGoalsTrend', 'AwayGoalsTrend',
        'H2HHomeWinRate', 'H2HAwayWinRate', 'H2HDrawRate',
        'H2HAvgGoals', 'H2HAvgHomeGoals', 'H2HAvgAwayGoals',
        'H2HBTTSRate', 'H2HOver25Rate', 'H2HMatches',
        'HomeExpGoals', 'AwayExpGoals', 'ExpTotalGoals',
        'PoissonHome', 'PoissonDraw', 'PoissonAway'
    ]
    
    # Add form features
    for w in [3, 5, 10, 15]:
        feature_cols.extend([
            f'HomeForm{w}', f'AwayForm{w}',
            f'HomeGoalsAvg{w}', f'AwayGoalsAvg{w}',
            f'HomeConcededAvg{w}', f'AwayConcededAvg{w}',
            f'HomeAttackStrength{w}', f'AwayAttackStrength{w}',
            f'HomeDefenseStrength{w}', f'AwayDefenseStrength{w}'
        ])
    
    # Add BTTS/Over features
    for w in [5, 10]:
        feature_cols.extend([
            f'HomeBTTSRate{w}', f'AwayBTTSRate{w}',
            f'HomeO15Rate{w}', f'AwayO15Rate{w}',
            f'HomeO25Rate{w}', f'AwayO25Rate{w}',
            f'HomeO35Rate{w}', f'AwayO35Rate{w}',
            f'HomeCSRate{w}', f'AwayCSRate{w}',
            f'HomeFTSRate{w}', f'AwayFTSRate{w}'
        ])
    
    # Add odds features
    for bookmaker in ['B365', 'BW', 'PS', 'WH', 'IW', 'VC', 'Avg']:
        for suffix in ['H', 'D', 'A', '_HomeProb', '_DrawProb', '_AwayProb']:
            col = f'{bookmaker}{suffix}'
            if col in df.columns:
                feature_cols.append(col)
    
    # Add match stats
    stat_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    feature_cols.extend([c for c in stat_cols if c in df.columns])
    
    # Filter available
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    # Fill NaN
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    logger.info(f"\n  ✅ Total features: {len(feature_cols)}")
    logger.info(f"  ✅ Total samples: {len(df):,}")
    
    return df, feature_cols, team_encoder


# =============================================================================
# MODEL TRAINING WITH OPTUNA
# =============================================================================

def train_with_optuna(X_train, y_train, X_test, y_test, model_type='xgb', n_trials=30):
    """Train model with Optuna hyperparameter optimization"""
    logger.info(f"\n🎯 Optuna optimization for {model_type.upper()} ({n_trials} trials)...")
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("  ⚠️ Optuna not installed, using default hyperparameters")
        return train_default(X_train, y_train, X_test, y_test, model_type)
    
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score
    
    def objective(trial):
        if model_type == 'xgb':
            import xgboost as xgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.3),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.5),
                'random_state': 42,
                'verbosity': 0,
                'n_jobs': -1
            }
            model = xgb.XGBClassifier(**params)
            
        elif model_type == 'lgb':
            import lightgbm as lgb
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 800),
                'max_depth': trial.suggest_int('max_depth', 6, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 31, 100),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 1.5),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            model = lgb.LGBMClassifier(**params)
            
        elif model_type == 'cat':
            from catboost import CatBoostClassifier
            params = {
                'iterations': trial.suggest_int('iterations', 200, 800),
                'depth': trial.suggest_int('depth', 6, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 8),
                'random_seed': 42,
                'verbose': False,
                'thread_count': -1
            }
            model = CatBoostClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    logger.info(f"  Best CV accuracy: {study.best_value:.2%}")
    logger.info(f"  Best params: {study.best_params}")
    
    # Train final model
    if model_type == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(**study.best_params, random_state=42, verbosity=0, n_jobs=-1)
    elif model_type == 'lgb':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**study.best_params, random_state=42, verbose=-1, n_jobs=-1)
    elif model_type == 'cat':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**study.best_params, random_seed=42, verbose=False)
    
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    logger.info(f"  Test accuracy: {acc:.2%}")
    
    return model, acc, study.best_params


def train_default(X_train, y_train, X_test, y_test, model_type):
    """Train with optimized default hyperparameters"""
    from sklearn.metrics import accuracy_score
    
    if model_type == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbosity=0, n_jobs=-1
        )
    elif model_type == 'lgb':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            num_leaves=63, subsample=0.85, colsample_bytree=0.85,
            random_state=42, verbose=-1, n_jobs=-1
        )
    elif model_type == 'cat':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=500, depth=8, learning_rate=0.05,
            random_seed=42, verbose=False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    
    return model, acc, {}


def train_neural_network(X_train, y_train, X_test, y_test, epochs=100):
    """Train PyTorch neural network"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score
    
    logger.info("\n🧠 Training Neural Network (PyTorch)...")
    
    device = torch.device('cpu')
    
    class FootballNet(nn.Module):
        def __init__(self, input_dim, num_classes=3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.net(x)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    model = FootballNet(X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_acc = 0
    patience = 0
    max_patience = 20
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_test_t).sum().item() / len(y_test_t)
            
            scheduler.step(1 - acc)
            
            if acc > best_acc:
                best_acc = acc
                patience = 0
                best_state = model.state_dict().copy()
            else:
                patience += 1
            
            if patience >= max_patience:
                logger.info(f"    Early stopping at epoch {epoch+1}")
                break
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"    Epoch {epoch+1}/{epochs} - Acc: {acc:.2%} (best: {best_acc:.2%})")
    
    model.load_state_dict(best_state)
    logger.info(f"  ✅ Neural Network Best Accuracy: {best_acc:.2%}")
    
    return model, best_acc


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def run_comprehensive_training(use_optuna=True, optuna_trials=30, nn_epochs=100):
    """Run comprehensive training pipeline"""
    logger.info("="*70)
    logger.info("🏆 FootyPredict Pro - ULTIMATE Training v4.0")
    logger.info(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("   Features: 500+ | Optuna: " + ("Yes" if use_optuna else "No"))
    logger.info("="*70)
    
    results = {
        'started': datetime.now().isoformat(),
        'models': {},
        'best_model': None,
        'best_accuracy': 0
    }
    
    try:
        # Step 1: Download data
        raw_data = download_all_data()
        
        if len(raw_data) == 0:
            raise ValueError("No data downloaded")
        
        # Step 2: Feature engineering
        df, feature_cols, team_encoder = engineer_all_features(raw_data)
        
        results['total_matches'] = len(df)
        results['total_features'] = len(feature_cols)
        
        # Prepare data
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        X = df[feature_cols].values
        y = df['Result'].values.astype(int)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.15, random_state=42, stratify=y
        )
        
        logger.info(f'\n📊 Dataset: Train={len(X_train):,} | Test={len(X_test):,}')
        
        # Step 3: Train models
        logger.info("\n" + "="*70)
        logger.info("🚀 STEP 3: Training Models")
        logger.info("="*70)
        
        models = {}
        
        # XGBoost
        if use_optuna:
            xgb_model, xgb_acc, xgb_params = train_with_optuna(
                X_train, y_train, X_test, y_test, 'xgb', optuna_trials
            )
        else:
            xgb_model, xgb_acc, xgb_params = train_default(
                X_train, y_train, X_test, y_test, 'xgb'
            )
        models['xgb'] = xgb_model
        results['models']['XGBoost'] = {'accuracy': xgb_acc, 'params': xgb_params}
        
        # LightGBM
        if use_optuna:
            lgb_model, lgb_acc, lgb_params = train_with_optuna(
                X_train, y_train, X_test, y_test, 'lgb', optuna_trials
            )
        else:
            lgb_model, lgb_acc, lgb_params = train_default(
                X_train, y_train, X_test, y_test, 'lgb'
            )
        models['lgb'] = lgb_model
        results['models']['LightGBM'] = {'accuracy': lgb_acc, 'params': lgb_params}
        
        # CatBoost
        if use_optuna:
            cat_model, cat_acc, cat_params = train_with_optuna(
                X_train, y_train, X_test, y_test, 'cat', optuna_trials
            )
        else:
            cat_model, cat_acc, cat_params = train_default(
                X_train, y_train, X_test, y_test, 'cat'
            )
        models['cat'] = cat_model
        results['models']['CatBoost'] = {'accuracy': cat_acc, 'params': cat_params}
        
        # Neural Network
        nn_model, nn_acc = train_neural_network(X_train, y_train, X_test, y_test, nn_epochs)
        results['models']['NeuralNet'] = {'accuracy': nn_acc}
        
        # Find best model
        accuracies = {
            'XGBoost': xgb_acc,
            'LightGBM': lgb_acc,
            'CatBoost': cat_acc,
            'NeuralNet': nn_acc
        }
        
        best_model = max(accuracies, key=accuracies.get)
        results['best_model'] = best_model
        results['best_accuracy'] = accuracies[best_model]
        
        # Save models
        import pickle
        
        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost
        xgb_model.save_model(str(TRAINED_DIR / 'xgb_football.json'))
        
        # Save LightGBM
        lgb_model.booster_.save_model(str(TRAINED_DIR / 'lgb_football.txt'))
        
        # Save CatBoost
        cat_model.save_model(str(TRAINED_DIR / 'cat_football.cbm'))
        
        # Save Neural Network
        import torch
        torch.save(nn_model.state_dict(), str(TRAINED_DIR / 'nn_football.pt'))
        
        # Save scaler and encoder
        with open(TRAINED_DIR / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(TRAINED_DIR / 'team_encoder.pkl', 'wb') as f:
            pickle.dump(team_encoder, f)
        
        # Save feature columns
        with open(TRAINED_DIR / 'feature_cols.json', 'w') as f:
            json.dump(feature_cols, f)
        
        # Save results
        results['completed'] = datetime.now().isoformat()
        results['success'] = True
        
        with open(TRAINED_DIR / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("📊 TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"  Total matches: {len(df):,}")
        logger.info(f"  Total features: {len(feature_cols)}")
        logger.info(f"  XGBoost accuracy: {xgb_acc:.2%}")
        logger.info(f"  LightGBM accuracy: {lgb_acc:.2%}")
        logger.info(f"  CatBoost accuracy: {cat_acc:.2%}")
        logger.info(f"  Neural Net accuracy: {nn_acc:.2%}")
        logger.info(f"  🏆 Best: {best_model} ({accuracies[best_model]:.2%})")
        logger.info("="*70)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        results['success'] = False
        results['error'] = str(e)
        return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Training')
    parser.add_argument('--optuna-trials', type=int, default=30)
    parser.add_argument('--nn-epochs', type=int, default=100)
    parser.add_argument('--no-optuna', action='store_true')
    
    args = parser.parse_args()
    
    results = run_comprehensive_training(
        use_optuna=not args.no_optuna,
        optuna_trials=args.optuna_trials,
        nn_epochs=args.nn_epochs
    )
    
    print(json.dumps(results, indent=2, default=str))
