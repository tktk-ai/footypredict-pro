#!/bin/bash
# =========================================================
# Football Prediction System - Deployment Script
# =========================================================

echo "🚀 Football Prediction System - Deployment"
echo "==========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: Run this script from the soccer project directory"
    exit 1
fi

# Step 1: Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << 'EOF'
# Football Prediction System - Environment Variables
# ==================================================

# Optional: Football-Data.org API key (for club football data)
# Get free key at: https://www.football-data.org/client/register
FOOTBALL_DATA_API_KEY=

# Optional: The Odds API key (for live odds)
# Get free key at: https://the-odds-api.com/
ODDS_API_KEY=

# Telegram Bot (for alerts)
# 1. Message @BotFather on Telegram
# 2. Create a new bot: /newbot
# 3. Copy the token here
TELEGRAM_BOT_TOKEN=

# Telegram Chat ID
# 1. Message @userinfobot on Telegram
# 2. It will reply with your chat ID
TELEGRAM_CHAT_ID=

# WhatsApp (Twilio)
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=

# Flask settings
FLASK_ENV=production
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
EOF
    echo -e "${GREEN}✅ Created .env file - please edit with your API keys${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Step 2: Create requirements.txt for production
echo -e "${YELLOW}Updating requirements.txt...${NC}"
cat > requirements.txt << 'EOF'
flask>=2.0.0
requests>=2.25.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
tensorflow>=2.8.0
aiohttp>=3.8.0
apscheduler>=3.9.0
python-dotenv>=0.19.0
gunicorn>=20.1.0
twilio>=7.0.0
stripe>=2.0.0
EOF
echo -e "${GREEN}✅ Updated requirements.txt${NC}"

# Step 3: Create Procfile for Heroku/Koyeb
echo -e "${YELLOW}Creating Procfile...${NC}"
cat > Procfile << 'EOF'
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
EOF
echo -e "${GREEN}✅ Created Procfile${NC}"

# Step 4: Update koyeb.yaml
echo -e "${YELLOW}Creating koyeb.yaml...${NC}"
cat > koyeb.yaml << 'EOF'
name: football-predictions
type: web
instance_types:
  - type: free
regions:
  - fra
ports:
  - port: 5000
    protocol: http
routes:
  - path: /
env:
  - key: FLASK_ENV
    value: production
  - key: PORT
    value: "5000"
build:
  type: buildpack
run: gunicorn app:app --bind 0.0.0.0:5000 --workers 2
health_checks:
  - type: http
    path: /api/health
    port: 5000
    interval_seconds: 60
    timeout_seconds: 20
EOF
echo -e "${GREEN}✅ Created koyeb.yaml${NC}"

# Step 5: Show deployment options
echo ""
echo "==========================================="
echo "Deployment Options:"
echo "==========================================="
echo ""
echo "Option 1: Koyeb (Recommended - Free)"
echo "  1. Install CLI: curl https://cli.koyeb.com/install.sh | bash"
echo "  2. Login: koyeb login"
echo "  3. Deploy: koyeb app create football-predictions --git . --port 5000"
echo ""
echo "Option 2: Vercel"
echo "  1. Install: npm i -g vercel"
echo "  2. Deploy: vercel --prod"
echo ""
echo "Option 3: Docker"
echo "  1. Build: docker build -t football-pred ."
echo "  2. Run: docker run -p 5000:5000 football-pred"
echo ""
echo "Option 4: Local (for testing)"
echo "  python app.py"
echo ""
echo -e "${GREEN}✅ Deployment files ready!${NC}"
