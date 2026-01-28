# FootyPredict Pro - Production Deployment Guide

## Architecture Overview

```
Hetzner VPS (€3.49/mo) + Kaggle (FREE) + GitHub Actions (FREE) + Cloudflare (FREE)
Total: ~$4.50/month
```

## Quick Start

### Prerequisites

- Hetzner account (https://hetzner.com)
- Kaggle account with API token (https://kaggle.com/settings)
- GitHub repository with secrets configured
- Domain name (optional but recommended)

---

## Step 1: Kaggle Setup (10 min)

### 1.1 Create Kaggle API Token

1. Go to https://kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`

### 1.2 Create Dataset

1. Go to https://kaggle.com/datasets
2. Click "+ New Dataset"
3. Name: `footypredict-data`
4. Upload: `data/merged_training_data.parquet`

### 1.3 Test Notebook

1. Upload `kaggle_training/footypredict_training.ipynb` to Kaggle
2. Enable GPU accelerator
3. Run once manually to verify it works

---

## Step 2: Hetzner VPS Setup (30 min)

### 2.1 Create VPS

1. Go to https://console.hetzner.cloud
2. Create project "FootyPredict"
3. Add server:
   - **Type**: CX22 (€3.49/mo)
   - **Image**: Ubuntu 22.04
   - **Location**: Your nearest
   - **SSH Key**: Add your public key

### 2.2 Deploy Application

```bash
# SSH into server
ssh root@YOUR_SERVER_IP

# Download and run deployment script
curl -sSL https://raw.githubusercontent.com/YOUR_USER/soccer/main/scripts/deploy_hetzner.sh | bash
```

### 2.3 Configure

1. Update domain in Nginx config
2. Update PostgreSQL password
3. Configure environment variables

---

## Step 3: GitHub Secrets (5 min)

Add these secrets to your GitHub repository:

| Secret            | Description              | Example         |
| ----------------- | ------------------------ | --------------- |
| `KAGGLE_USERNAME` | Your Kaggle username     | `nananie143`    |
| `KAGGLE_KEY`      | API key from kaggle.json | `abc123...`     |
| `VPS_HOST`        | Hetzner server IP        | `123.45.67.89`  |
| `VPS_USER`        | SSH user                 | `footypredict`  |
| `VPS_SSH_KEY`     | Private SSH key          | `-----BEGIN...` |

---

## Step 4: Cloudflare Setup (10 min)

### 4.1 Add Domain

1. Go to https://dash.cloudflare.com
2. Add site → Enter your domain
3. Follow nameserver instructions

### 4.2 Configure DNS

```
Type    Name    Content           Proxy
A       @       YOUR_SERVER_IP    Proxied (orange)
CNAME   www     @                 Proxied (orange)
```

### 4.3 SSL Settings

- SSL/TLS → Full (strict)
- Enable "Always Use HTTPS"

---

## Automated Training

Training runs automatically on the 1st and 15th of each month.

### Manual Trigger

```bash
# Option 1: GitHub Actions
gh workflow run retrain.yml

# Option 2: Local script
./scripts/trigger_training.sh
```

### Monitor Training

- GitHub: Actions tab → "Bi-Weekly Model Retraining"
- Kaggle: https://kaggle.com/code/YOUR_USER/footypredict-training

---

## Maintenance

### View Logs

```bash
ssh footypredict@YOUR_SERVER
sudo journalctl -u footypredict -f
```

### Restart App

```bash
sudo systemctl restart footypredict
```

### Update Code

```bash
cd /app/soccer
git pull
sudo systemctl restart footypredict
```

---

## Cost Summary

| Service        | Monthly Cost       |
| -------------- | ------------------ |
| Hetzner CX22   | €3.49 (~$4)        |
| Kaggle GPU     | FREE (30 hrs/week) |
| GitHub Actions | FREE (2000 min/mo) |
| Cloudflare CDN | FREE               |
| **Total**      | **~$4.50/month**   |
