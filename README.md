# Soccer Prediction System

AI-powered football match prediction with real fixtures and ML models.

## Quick Start

```bash
cd /home/netboss/Desktop/pers_bus/soccer
source venv/bin/activate
python app.py
```

Open http://localhost:5000

## Features

- 🏆 11+ leagues (Bundesliga, Premier League, La Liga, Serie A, etc.)
- 📊 ELO-based predictions with ML ensemble
- ⚽ Goal predictions (xG, O2.5, BTTS)
- 💰 Value bet detection & Kelly Criterion
- 🎰 Accumulator generator
- 📱 PWA mobile app
- 🤖 Telegram & WhatsApp bots

## API Endpoints

```
GET /api/fixtures?league=bundesliga&days=7
GET /api/predict?home=Bayern&away=Dortmund
GET /api/standings?league=bundesliga
GET /api/leagues
GET /api/h2h?home=Bayern&away=Dortmund
```
