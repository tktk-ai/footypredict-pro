#!/bin/bash
# =========================================================
# Telegram Bot Setup Script
# =========================================================

echo "📱 Telegram Bot Setup"
echo "====================="
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env file first..."
    touch .env
fi

# Get the token
echo "Step 1: Create Your Bot"
echo "------------------------"
echo "1. Open Telegram and search for @BotFather"
echo "2. Send: /newbot"
echo "3. Follow the prompts to create your bot"
echo "4. Copy the API token that BotFather gives you"
echo ""
read -p "Paste your bot token here: " BOT_TOKEN

# Get chat ID
echo ""
echo "Step 2: Get Your Chat ID"
echo "------------------------"
echo "1. Open Telegram and search for @userinfobot"
echo "2. Start a chat with it"
echo "3. It will reply with your chat ID"
echo ""
read -p "Paste your chat ID here: " CHAT_ID

# Update .env file
if grep -q "TELEGRAM_BOT_TOKEN=" .env; then
    sed -i "s/TELEGRAM_BOT_TOKEN=.*/TELEGRAM_BOT_TOKEN=$BOT_TOKEN/" .env
else
    echo "TELEGRAM_BOT_TOKEN=$BOT_TOKEN" >> .env
fi

if grep -q "TELEGRAM_CHAT_ID=" .env; then
    sed -i "s/TELEGRAM_CHAT_ID=.*/TELEGRAM_CHAT_ID=$CHAT_ID/" .env
else
    echo "TELEGRAM_CHAT_ID=$CHAT_ID" >> .env
fi

echo ""
echo "✅ Telegram credentials saved to .env"
echo ""

# Test the bot
echo "Testing bot connection..."
source venv/bin/activate 2>/dev/null || true

python3 << EOF
import os
import requests
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('TELEGRAM_BOT_TOKEN', '$BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID', '$CHAT_ID')

if token and chat_id:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(url, json={
        'chat_id': chat_id,
        'text': '🎉 Football Prediction Bot Connected!\n\nYou will now receive:\n• Daily predictions at 9 AM\n• Sure win alerts\n• Value bet notifications\n• Weekly accuracy reports',
        'parse_mode': 'HTML'
    })
    print(f"Bot test: {'✅ Success!' if response.status_code == 200 else '❌ Failed'}")
else:
    print("❌ Missing token or chat ID")
EOF

echo ""
echo "Setup complete! Start the server and enable alerts:"
echo "  curl -X POST http://localhost:5000/api/cron/start"
