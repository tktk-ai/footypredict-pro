---
title: FootyPredict Pro
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# FootyPredict Pro ⚽

AI-powered football match prediction with real fixtures and advanced ML models.

## 🚀 Live Demo

This app is deployed on Hugging Face Spaces with full ML capabilities including PyTorch for reinforcement learning models.

## Features

- 🏆 **35+ leagues** (Premier League, La Liga, Bundesliga, Serie A, etc.)
- 📊 **ELO-based predictions** with ML ensemble (XGBoost, LightGBM, CatBoost)
- 🤖 **RL Models** - PyTorch-powered reinforcement learning predictions
- ⚽ **Goal predictions** (xG, O2.5, BTTS)
- 💰 **Value bet detection** & Kelly Criterion
- 🎰 **Accumulator generator** with smart filtering
- 📱 **PWA mobile app** support
- 🤖 **Telegram & WhatsApp bots**

## API Endpoints

```
GET /api/fixtures?league=premier_league&days=7
GET /api/predict?home=Bayern&away=Dortmund
GET /api/standings?league=premier_league
GET /api/leagues
GET /api/h2h?home=Bayern&away=Dortmund
GET /api/accumulators
```

## Tech Stack

- **Backend**: Flask + Gunicorn
- **ML Models**: XGBoost, LightGBM, CatBoost, PyTorch (RL)
- **Data**: Real-time fixtures from multiple free APIs

## Memory Requirements

This app requires ~1.5GB RAM to load all ML models. Hugging Face Spaces free tier (2GB) provides sufficient memory.

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000
