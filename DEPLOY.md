# 🚀 Koyeb Deployment Guide

## Quick Deploy (3 Steps)

### 1. Push to GitHub

```bash
git add .
git commit -m "Add Koyeb deployment files"
git push origin main
```

### 2. Deploy on Koyeb

1. Go to [koyeb.com](https://app.koyeb.com) and sign up (free)
2. Click **"Create App"** → **"GitHub"**
3. Connect your GitHub account
4. Select your repository: `soccer`
5. Set these options:
   - **Builder**: Dockerfile
   - **Region**: Frankfurt (free)
   - **Instance**: Free tier

### 3. Add Environment Variables

In Koyeb dashboard → Settings → Environment Variables:

| Variable                | Value                   |
| ----------------------- | ----------------------- |
| `FOOTBALL_DATA_API_KEY` | Your API key            |
| `THE_ODDS_API_KEY`      | Your API key            |
| `API_FOOTBALL_KEY`      | Your API key (optional) |
| `FLASK_ENV`             | production              |

---

## Files Created

- `Procfile` - Gunicorn web server command
- `runtime.txt` - Python version
- `Dockerfile` - Container build instructions
- `koyeb.yaml` - App configuration

## Free Tier Limits

- ✅ 512MB RAM
- ✅ 2GB Storage
- ✅ 0.1 vCPU
- ✅ Never sleeps!
- ✅ Free PostgreSQL available

## Your App URL

After deployment: `https://your-app-name.koyeb.app`

## Troubleshooting

- Check logs: Koyeb Dashboard → Logs
- Rebuild: Settings → Redeploy

---

Built with ❤️ for Football Predictions
