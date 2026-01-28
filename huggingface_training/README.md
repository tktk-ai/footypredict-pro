---
title: FootyPredict V4 Training
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# FootyPredict V4 Training

GPU-accelerated training for the V4.0 football prediction model.

## Features

- **698 engineered features** from 27 categories
- **15 data sources** including Understat, ClubElo, OpenLigaDB
- **Ensemble models**: XGBoost, LightGBM, CatBoost
- **Optuna** hyperparameter optimization

## Usage

1. Upload your training data (CSV or Parquet)
2. Configure training parameters
3. Click "Start Training"
4. Download trained models

## Data Format

Your training data should include:

- `Date`: Match date
- `HomeTeam`, `AwayTeam`: Team names
- `FTHG`, `FTAG`: Full-time goals
- `FTR`: Full-time result (H/D/A)
