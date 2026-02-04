#!/bin/bash
# =============================================================================
# FootyPredict Weekly Automation Setup
# =============================================================================
#
# This script sets up cron jobs for automated:
# - Weekly model retraining (Sundays 2 AM)
# - Daily fixture refresh (6 AM)
# - Hourly odds updates (every hour)
#
# Usage:
#   chmod +x scripts/cron_setup.sh
#   ./scripts/cron_setup.sh          # Install cron jobs
#   ./scripts/cron_setup.sh --remove # Remove cron jobs
#
# =============================================================================

set -e

PROJECT_DIR="/home/netboss/Desktop/pers_bus/soccer"
PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
LOG_DIR="${PROJECT_DIR}/logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}FootyPredict Cron Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if --remove flag is passed
if [[ "$1" == "--remove" ]]; then
    echo -e "${YELLOW}Removing FootyPredict cron jobs...${NC}"
    crontab -l 2>/dev/null | grep -v "FootyPredict" | grep -v "weekly_training" | grep -v "sportybet_scraper" | crontab -
    echo -e "${GREEN}✓ Cron jobs removed${NC}"
    exit 0
fi

# Verify paths exist
if [[ ! -f "$PYTHON_BIN" ]]; then
    echo -e "${YELLOW}Warning: Virtual env not found at $PYTHON_BIN${NC}"
    echo "Using system python3 instead"
    PYTHON_BIN=$(which python3)
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo -e "${RED}Error: Project directory not found: $PROJECT_DIR${NC}"
    exit 1
fi

echo -e "\nProject: ${PROJECT_DIR}"
echo -e "Python:  ${PYTHON_BIN}"
echo -e "Logs:    ${LOG_DIR}\n"

# Define cron jobs
read -r -d '' CRON_JOBS << 'EOF' || true
# ============================================
# FootyPredict Automated Tasks
# ============================================

# Weekly model retraining (Every Sunday at 2:00 AM)
0 2 * * 0 cd /home/netboss/Desktop/pers_bus/soccer && /home/netboss/Desktop/pers_bus/soccer/venv/bin/python -m src.training.weekly_training_pipeline >> /home/netboss/Desktop/pers_bus/soccer/logs/weekly_training.log 2>&1

# Daily fixture collection (Every day at 6:00 AM)
0 6 * * * cd /home/netboss/Desktop/pers_bus/soccer && /home/netboss/Desktop/pers_bus/soccer/venv/bin/python -m src.data.sportybet_scraper --days 7 --save >> /home/netboss/Desktop/pers_bus/soccer/logs/daily_fixtures.log 2>&1

# Hourly odds update for today's matches (Every hour at :30)
30 * * * * cd /home/netboss/Desktop/pers_bus/soccer && /home/netboss/Desktop/pers_bus/soccer/venv/bin/python -c "from src.data.sportybet_scraper import SportyBetScraper; s=SportyBetScraper(); s.save_fixtures_to_csv(s.get_todays_fixtures(), 'today_odds.csv')" >> /home/netboss/Desktop/pers_bus/soccer/logs/hourly_odds.log 2>&1

# Clear old logs weekly (Sundays at 1:00 AM, keep last 30 days)
0 1 * * 0 find /home/netboss/Desktop/pers_bus/soccer/logs -name "*.log" -mtime +30 -delete

EOF

# Get existing crontab (if any)
EXISTING_CRON=$(crontab -l 2>/dev/null || echo "")

# Check if our jobs already exist
if echo "$EXISTING_CRON" | grep -q "FootyPredict"; then
    echo -e "${YELLOW}FootyPredict cron jobs already exist.${NC}"
    read -p "Do you want to reinstall them? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing cron jobs."
        exit 0
    fi
    # Remove existing FootyPredict jobs
    EXISTING_CRON=$(echo "$EXISTING_CRON" | grep -v "FootyPredict" | grep -v "weekly_training" | grep -v "sportybet_scraper")
fi

# Add new cron jobs
echo "$EXISTING_CRON" | cat - <(echo "$CRON_JOBS") | crontab -

echo -e "${GREEN}✓ Cron jobs installed successfully!${NC}\n"
echo "Scheduled tasks:"
echo "  • Weekly training: Sundays at 2:00 AM"
echo "  • Daily fixtures:  Every day at 6:00 AM"
echo "  • Hourly odds:     Every hour at :30"
echo ""
echo "View scheduled jobs with: crontab -l"
echo "View logs at: $LOG_DIR"
