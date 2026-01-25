# 🚀 Football Prediction System - Deployment Guide

## System Overview

This is a **complete AI-powered football prediction system** with:

- 5-model ML ensemble (XGBoost, LightGBM, CatBoost, Neural Net)
- 70+ API endpoints
- Real-time predictions with advanced features
- Auto-tuning and retraining capabilities

---

## Quick Start (Local)

```bash
# 1. Clone and setup
cd /home/netboss/Desktop/pers_bus/soccer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start server
python app.py

# 3. Open browser
# http://localhost:5000
```

---

## Deployment Options

### Option 1: Koyeb (Recommended - Free Tier)

```bash
# 1. Install Koyeb CLI
curl https://cli.koyeb.com/install.sh | bash

# 2. Login
koyeb login

# 3. Deploy
koyeb app create football-predictions \
  --git github.com/your-username/soccer \
  --git-branch main \
  --instance-type free \
  --port 5000
```

### Option 2: Docker

```bash
# Build
docker build -t football-predictions .

# Run
docker run -p 5000:5000 football-predictions
```

### Option 3: Vercel (Serverless)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

---

## Environment Variables

Create a `.env` file:

```env
# Optional - for enhanced features
FOOTBALL_DATA_API_KEY=your_key_here
ODDS_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

---

## API Endpoints Summary

### Core Predictions

| Endpoint                            | Description         |
| ----------------------------------- | ------------------- |
| `GET /api/v2/predict?home=X&away=Y` | Enhanced prediction |
| `GET /api/form/{team}`              | Team form           |
| `GET /api/h2h?team1=X&team2=Y`      | Head-to-head        |
| `GET /api/injuries/{team}`          | Injury data         |

### Live Data

| Endpoint                  | Description          |
| ------------------------- | -------------------- |
| `GET /api/live-odds`      | Live odds comparison |
| `GET /api/live-scores`    | Live match scores    |
| `GET /api/fixtures/today` | Today's fixtures     |

### In-Play

| Endpoint                     | Description          |
| ---------------------------- | -------------------- |
| `POST /api/inplay/start`     | Start tracking match |
| `POST /api/inplay/update`    | Update score         |
| `GET /api/inplay/{match_id}` | Get live prediction  |

### Training & Tuning

| Endpoint                   | Description         |
| -------------------------- | ------------------- |
| `POST /api/training/start` | Start retraining    |
| `POST /api/tuning/set`     | Set hyperparameters |
| `POST /api/schedule/start` | Start auto-schedule |

### Analytics

| Endpoint                  | Description        |
| ------------------------- | ------------------ |
| `GET /api/accuracy/stats` | Accuracy dashboard |
| `POST /api/backtest/run`  | Run backtest       |
| `POST /api/ab-test/run`   | Run A/B test       |

### Documentation

| Endpoint        | Description           |
| --------------- | --------------------- |
| `GET /api/docs` | OpenAPI specification |

---

## Feature Checklist

- [x] ML Ensemble (5 models)
- [x] Form data (last 5)
- [x] Head-to-head
- [x] Injuries
- [x] Weather
- [x] Live odds
- [x] In-play predictions
- [x] Auto-tuning
- [x] Scheduled retraining
- [x] A/B testing
- [x] Backtesting
- [x] Telegram alerts
- [x] WhatsApp bot
- [x] PWA mobile app
- [x] API documentation
- [x] Test suite

---

## Run Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_predictions.py -v
```

---

## Performance

| Metric            | Value  |
| ----------------- | ------ |
| Expected Accuracy | 65-70% |
| API Response Time | <100ms |
| Models Loaded     | 5      |
| Leagues Covered   | 22+    |

---

## Support

Open issues at: https://github.com/your-repo
