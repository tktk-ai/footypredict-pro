#!/usr/bin/env python3
"""
Training Data Enhancement Script
================================

Adds new features to improve prediction accuracy:
1. Win/Loss Streaks - Momentum indicators
2. xG Estimates - From shots data
3. Form Derivatives - Rolling averages
4. Home/Away Specific Stats

Run before Kaggle training for better results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


def calculate_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss/draw streaks for each team.
    
    Adds:
    - home_win_streak: Consecutive home wins
    - away_win_streak: Consecutive away wins
    - home_unbeaten_streak: Games without loss at home
    - away_scoring_streak: Games scoring away
    """
    logger.info("Calculating win/loss streaks...")
    
    # Sort by date
    df = df.sort_values('Date').copy()
    
    # Initialize streak columns
    df['home_win_streak'] = 0
    df['away_win_streak'] = 0
    df['home_unbeaten_streak'] = 0
    df['away_unbeaten_streak'] = 0
    df['home_scoring_streak'] = 0
    df['away_scoring_streak'] = 0
    
    # Track streaks per team
    team_streaks = {}
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row.get('FTR', '')
        home_goals = row.get('FTHG', 0) or 0
        away_goals = row.get('FTAG', 0) or 0
        
        # Initialize team if not seen
        for team in [home, away]:
            if team not in team_streaks:
                team_streaks[team] = {
                    'home_wins': 0, 'away_wins': 0,
                    'home_unbeaten': 0, 'away_unbeaten': 0,
                    'home_scoring': 0, 'away_scoring': 0
                }
        
        # Set current streaks before updating
        df.at[idx, 'home_win_streak'] = team_streaks[home]['home_wins']
        df.at[idx, 'away_win_streak'] = team_streaks[away]['away_wins']
        df.at[idx, 'home_unbeaten_streak'] = team_streaks[home]['home_unbeaten']
        df.at[idx, 'away_unbeaten_streak'] = team_streaks[away]['away_unbeaten']
        df.at[idx, 'home_scoring_streak'] = team_streaks[home]['home_scoring']
        df.at[idx, 'away_scoring_streak'] = team_streaks[away]['away_scoring']
        
        # Update streaks based on result
        if result == 'H':
            team_streaks[home]['home_wins'] += 1
            team_streaks[home]['home_unbeaten'] += 1
            team_streaks[away]['away_wins'] = 0
            team_streaks[away]['away_unbeaten'] = 0
        elif result == 'A':
            team_streaks[away]['away_wins'] += 1
            team_streaks[away]['away_unbeaten'] += 1
            team_streaks[home]['home_wins'] = 0
            team_streaks[home]['home_unbeaten'] = 0
        else:  # Draw
            team_streaks[home]['home_wins'] = 0
            team_streaks[away]['away_wins'] = 0
            team_streaks[home]['home_unbeaten'] += 1
            team_streaks[away]['away_unbeaten'] += 1
        
        # Scoring streaks
        if home_goals > 0:
            team_streaks[home]['home_scoring'] += 1
        else:
            team_streaks[home]['home_scoring'] = 0
            
        if away_goals > 0:
            team_streaks[away]['away_scoring'] += 1
        else:
            team_streaks[away]['away_scoring'] = 0
    
    logger.info(f"Added 6 streak features")
    return df


def estimate_xg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate xG from shots data.
    
    Simple xG model:
    - Shots on target: ~0.3 xG each
    - Shots off target: ~0.05 xG each
    
    This is a rough estimate, but better than nothing.
    """
    logger.info("Estimating xG from shots data...")
    
    # xG conversion rates (empirical averages)
    XG_PER_SOT = 0.32  # Shots on target
    XG_PER_SHOT = 0.08  # All shots (including off target)
    
    df['home_xg_est'] = 0.0
    df['away_xg_est'] = 0.0
    
    # Calculate xG where shots data available
    has_shots = df['HS'].notna() & df['HST'].notna()
    
    df.loc[has_shots, 'home_xg_est'] = (
        df.loc[has_shots, 'HST'] * XG_PER_SOT + 
        (df.loc[has_shots, 'HS'] - df.loc[has_shots, 'HST']) * (XG_PER_SHOT - XG_PER_SOT * 0.3)
    ).clip(lower=0)
    
    df.loc[has_shots, 'away_xg_est'] = (
        df.loc[has_shots, 'AST'] * XG_PER_SOT + 
        (df.loc[has_shots, 'AS'] - df.loc[has_shots, 'AST']) * (XG_PER_SHOT - XG_PER_SOT * 0.3)
    ).clip(lower=0)
    
    # For matches without shots, use goals as proxy
    no_shots = ~has_shots
    df.loc[no_shots, 'home_xg_est'] = df.loc[no_shots, 'FTHG'].fillna(1.3)
    df.loc[no_shots, 'away_xg_est'] = df.loc[no_shots, 'FTAG'].fillna(1.0)
    
    # xG difference
    df['xg_diff'] = df['home_xg_est'] - df['away_xg_est']
    
    logger.info(f"Added 3 xG features (home_xg_est, away_xg_est, xg_diff)")
    return df


def add_form_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced form features:
    - Points per game (last 5)
    - Goals scored/conceded trends
    - Clean sheet rate
    - BTTS rate
    """
    logger.info("Calculating form derivatives...")
    
    df = df.sort_values('Date').copy()
    
    # Initialize columns
    df['home_ppg_5'] = 1.0  # Points per game (last 5)
    df['away_ppg_5'] = 1.0
    df['home_goals_trend'] = 0.0  # Recent vs overall
    df['away_goals_trend'] = 0.0
    df['home_clean_sheet_rate'] = 0.2
    df['away_clean_sheet_rate'] = 0.2
    df['match_btts_likelihood'] = 0.5
    
    # Track per-team stats
    team_history = {}
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        result = row.get('FTR', 'D')
        hg = row.get('FTHG', 0) or 0
        ag = row.get('FTAG', 0) or 0
        
        # Initialize teams
        for team in [home, away]:
            if team not in team_history:
                team_history[team] = {
                    'results': [],  # List of points
                    'goals_for': [],
                    'goals_against': [],
                    'clean_sheets': []
                }
        
        # Calculate current values before updating
        home_hist = team_history[home]
        away_hist = team_history[away]
        
        if len(home_hist['results']) >= 5:
            df.at[idx, 'home_ppg_5'] = np.mean(home_hist['results'][-5:])
            df.at[idx, 'home_clean_sheet_rate'] = np.mean(home_hist['clean_sheets'][-5:])
        
        if len(away_hist['results']) >= 5:
            df.at[idx, 'away_ppg_5'] = np.mean(away_hist['results'][-5:])
            df.at[idx, 'away_clean_sheet_rate'] = np.mean(away_hist['clean_sheets'][-5:])
        
        # Goals trend (last 3 vs last 10)
        if len(home_hist['goals_for']) >= 10:
            recent = np.mean(home_hist['goals_for'][-3:])
            overall = np.mean(home_hist['goals_for'][-10:])
            df.at[idx, 'home_goals_trend'] = recent - overall
            
        if len(away_hist['goals_for']) >= 10:
            recent = np.mean(away_hist['goals_for'][-3:])
            overall = np.mean(away_hist['goals_for'][-10:])
            df.at[idx, 'away_goals_trend'] = recent - overall
        
        # BTTS likelihood
        home_scores = np.mean(home_hist['goals_for'][-5:]) if len(home_hist['goals_for']) >= 5 else 1.3
        away_scores = np.mean(away_hist['goals_for'][-5:]) if len(away_hist['goals_for']) >= 5 else 1.0
        home_concedes = np.mean(home_hist['goals_against'][-5:]) if len(home_hist['goals_against']) >= 5 else 1.2
        away_concedes = np.mean(away_hist['goals_against'][-5:]) if len(away_hist['goals_against']) >= 5 else 1.3
        
        # P(home scores) * P(away scores)
        p_home = 1 - np.exp(-((home_scores + away_concedes) / 2))
        p_away = 1 - np.exp(-((away_scores + home_concedes) / 2))
        df.at[idx, 'match_btts_likelihood'] = p_home * p_away
        
        # Update history
        points = 3 if result == 'H' else (1 if result == 'D' else 0)
        away_points = 3 if result == 'A' else (1 if result == 'D' else 0)
        
        team_history[home]['results'].append(points)
        team_history[home]['goals_for'].append(hg)
        team_history[home]['goals_against'].append(ag)
        team_history[home]['clean_sheets'].append(1 if ag == 0 else 0)
        
        team_history[away]['results'].append(away_points)
        team_history[away]['goals_for'].append(ag)
        team_history[away]['goals_against'].append(hg)
        team_history[away]['clean_sheets'].append(1 if hg == 0 else 0)
    
    logger.info("Added 7 form derivative features")
    return df


def add_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add odds-derived features.
    """
    logger.info("Adding odds-derived features...")
    
    # Odds ratio (market assessment of strength)
    if 'B365H' in df.columns and 'B365A' in df.columns:
        df['odds_ratio'] = df['B365A'] / df['B365H'].replace(0, np.nan)
        df['odds_ratio'] = df['odds_ratio'].fillna(1.0)
        
        # Implied total goals from over/under
        if 'B365>2.5' in df.columns:
            # Higher over odds = lower expected goals
            df['implied_goals'] = 2.5 + np.log(df['B365<2.5'].fillna(1.9) / df['B365>2.5'].fillna(1.9))
        else:
            df['implied_goals'] = 2.5
        
        # Market confidence (inverse of favorite odds)
        df['market_confidence'] = 1 / df[['B365H', 'B365D', 'B365A']].min(axis=1)
    
    logger.info("Added 3 odds-derived features")
    return df


def enhance_training_data():
    """Main function to enhance training data."""
    
    input_file = DATA_DIR / "comprehensive_training_data.csv"
    output_file = DATA_DIR / "enhanced_training_data.csv"
    
    logger.info(f"Loading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    
    original_cols = len(df.columns)
    logger.info(f"Original: {len(df):,} rows, {original_cols} columns")
    
    # Parse dates properly
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    
    # Apply all enhancements
    df = calculate_streaks(df)
    df = estimate_xg(df)
    df = add_form_derivatives(df)
    df = add_odds_features(df)
    
    new_cols = len(df.columns)
    logger.info(f"Enhanced: {len(df):,} rows, {new_cols} columns (+{new_cols - original_cols} features)")
    
    # Save enhanced data
    df.to_csv(output_file, index=False)
    logger.info(f"Saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ENHANCEMENT COMPLETE")
    print("=" * 60)
    print(f"Original features: {original_cols}")
    print(f"New features: {new_cols}")
    print(f"Added: {new_cols - original_cols} features")
    print()
    print("New features added:")
    new_feat = [
        'home_win_streak', 'away_win_streak', 
        'home_unbeaten_streak', 'away_unbeaten_streak',
        'home_scoring_streak', 'away_scoring_streak',
        'home_xg_est', 'away_xg_est', 'xg_diff',
        'home_ppg_5', 'away_ppg_5',
        'home_goals_trend', 'away_goals_trend',
        'home_clean_sheet_rate', 'away_clean_sheet_rate',
        'match_btts_likelihood',
        'odds_ratio', 'implied_goals', 'market_confidence'
    ]
    for f in new_feat:
        print(f"  ✅ {f}")
    
    print(f"\nOutput: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    return df


if __name__ == "__main__":
    enhance_training_data()
