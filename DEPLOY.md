# FootyPredict Pro - Production Architecture & Deployment

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [GitHub]              [Kaggle]           [HuggingFace]        │
│   Code Repo          GPU Training         Model Serving          │
│      │                    │                    │                 │
│      └────────┬───────────┴────────────────────┘                 │
│               │                                                  │
│               ▼                                                  │
│   ┌───────────────────────────────────────────┐                 │
│   │          CLOUDFLARE (FREE)                 │                 │
│   ├───────────────────────────────────────────┤                 │
│   │  Pages: Frontend UI                        │                 │
│   │  footypredict-ui.pages.dev                │                 │
│   │                                            │                 │
│   │  Workers: API Backend                      │                 │
│   │  footypredict-api.workers.dev             │                 │
│   └───────────────────────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 💰 Cost Summary

| Service                | Purpose                          | Cost                |
| ---------------------- | -------------------------------- | ------------------- |
| **GitHub**             | Code repository, CI/CD           | FREE                |
| **Kaggle**             | GPU model training (30 hrs/week) | FREE                |
| **HuggingFace Spaces** | ML model serving                 | FREE                |
| **Cloudflare Pages**   | Frontend hosting                 | FREE                |
| **Cloudflare Workers** | API backend                      | FREE (100K req/day) |
| **Total**              |                                  | **$0/month**        |

---

## 📍 Live URLs

| Service            | URL                                              |
| ------------------ | ------------------------------------------------ |
| Frontend           | https://footypredict-ui.pages.dev                |
| API                | https://footypredict-api.tirene857.workers.dev   |
| Sure Bets          | https://footypredict-ui.pages.dev/sure-bets.html |
| HuggingFace Models | https://nananie143-footypredict-pro.hf.space     |

---

## 🚀 Deployment Guide

### Step 1: GitHub Setup

1. Push code to GitHub repository
2. Configure secrets (optional for API keys):
   ```
   KAGGLE_USERNAME, KAGGLE_KEY (for training)
   RAPIDAPI_KEY (for fixtures)
   ```

### Step 2: Cloudflare Pages (Frontend)

```bash
cd cloudflare-pages
npx wrangler pages deploy . --project-name=footypredict-ui
```

### Step 3: Cloudflare Workers (API)

```bash
cd cloudflare-worker
npx wrangler deploy --name footypredict-api
```

### Step 4: HuggingFace Space (ML Models)

1. Create Space at https://huggingface.co/spaces
2. Upload `huggingface_training/` files
3. Space auto-deploys and serves `/api/predict` endpoint

---

## 🧠 Training Pipeline (Kaggle)

### Setup Kaggle Training

1. Create Kaggle account at https://kaggle.com
2. Go to Settings → API → Create New Token
3. Upload training notebook:
   ```
   kaggle_training/footypredict_training.ipynb
   ```
4. Enable GPU accelerator (T4 GPU, 30 hrs/week free)

### Training Data Location

```
data/processed/training_data_unified.csv (76,268 matches)
```

### Trained Models (V4)

Location: `models/v4_fixed/`

- `result_model.pkl` - Match result (57% accuracy)
- `over25_model.pkl` - Over 2.5 goals (67.9% accuracy)
- `over15_model.pkl` - Over 1.5 goals (75.6% accuracy)
- `btts_model.pkl` - Both teams to score (64.5% accuracy)

### Deploy Models to HuggingFace

After training on Kaggle:

```bash
# Upload models to HuggingFace Space
cd huggingface_training
git add .
git commit -m "Update trained models"
git push huggingface main
```

---

## 🔄 Data Flow

```
1. User visits footypredict-ui.pages.dev
                    │
                    ▼
2. Frontend calls footypredict-api.workers.dev
                    │
                    ▼
3. Worker fetches fixtures from:
   - API-Football (RapidAPI)
   - TheSportsDB (free)
   - The Odds API (free tier)
                    │
                    ▼
4. Worker calls HuggingFace for ML predictions:
   https://nananie143-footypredict-pro.hf.space/api/predict
                    │
                    ▼
5. Worker combines data + predictions → Returns JSON
                    │
                    ▼
6. Frontend renders predictions to user
```

---

## 🔧 Common Commands

### Deploy Frontend

```bash
cd cloudflare-pages && npx wrangler pages deploy . --project-name=footypredict-ui
```

### Deploy Worker

```bash
cd cloudflare-worker && npx wrangler deploy
```

### Test API

```bash
curl https://footypredict-api.tirene857.workers.dev/sure-bets | jq
```

### Test HuggingFace

```bash
curl "https://nananie143-footypredict-pro.hf.space/api/predict?home=Liverpool&away=Arsenal"
```

---

## 📊 Model Accuracy (V4)

| Market             | Accuracy  | CV Score |
| ------------------ | --------- | -------- |
| Match Result       | 57.0%     | 56.6%    |
| Over 2.5 Goals     | 67.9%     | 67.4%    |
| **Over 1.5 Goals** | **75.6%** | 75.4%    |
| BTTS               | 64.5%     | 64.6%    |

---

## 🛡️ Fallback Strategy

The system has multiple fallback layers:

1. **Primary**: HuggingFace ML API (trained models)
2. **Secondary**: Odds-based calculation (from betting markets)
3. **Tertiary**: Team strength algorithm (Elo-based statistics)

If HuggingFace is down, predictions still work using market odds or statistical methods.
