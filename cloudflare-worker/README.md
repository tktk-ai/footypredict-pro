# FootyPredict API - Cloudflare Worker

Lightweight serverless prediction API powered by ML ensemble models trained on Kaggle GPU.

## Endpoints

| Endpoint       | Method | Description                 |
| -------------- | ------ | --------------------------- |
| `/`            | GET    | API info and documentation  |
| `/predict`     | POST   | Get single match prediction |
| `/batch`       | POST   | Get batch predictions       |
| `/health`      | GET    | Health check                |
| `/models/info` | GET    | Model metadata              |

## Usage

### Single Prediction

```bash
curl -X POST https://footypredict-api.YOUR_SUBDOMAIN.workers.dev/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "league": "Premier League"
  }'
```

### Batch Predictions

```bash
curl -X POST https://footypredict-api.YOUR_SUBDOMAIN.workers.dev/batch \
  -H "Content-Type: application/json" \
  -d '{
    "matches": [
      {"home_team": "Arsenal", "away_team": "Chelsea"},
      {"home_team": "Liverpool", "away_team": "Man City"}
    ]
  }'
```

## Deployment

### Prerequisites

1. Install Wrangler CLI:

   ```bash
   npm install -g wrangler
   ```

2. Login to Cloudflare:
   ```bash
   wrangler login
   ```

### Deploy

```bash
cd cloudflare-worker
wrangler deploy
```

### Local Development

```bash
cd cloudflare-worker
wrangler dev
```

## Configuration

Edit `wrangler.toml` to:

- Set custom domain routes
- Configure KV namespaces for caching
- Add R2 bucket for model storage

## GitHub Secrets Required

| Secret            | Description                                   |
| ----------------- | --------------------------------------------- |
| `CF_ACCOUNT_ID`   | Cloudflare Account ID                         |
| `CF_API_TOKEN`    | Cloudflare API Token with Workers permissions |
| `KAGGLE_USERNAME` | Kaggle username                               |
| `KAGGLE_KEY`      | Kaggle API key                                |
