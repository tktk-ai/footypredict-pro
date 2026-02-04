#!/bin/bash
# FootyPredict Pro - Hetzner VPS Deployment Script
# Run this on your VPS after initial Ubuntu 22.04 setup

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== FootyPredict Pro VPS Setup ===${NC}"

# 1. System Update
echo -e "${YELLOW}[1/8] Updating system...${NC}"
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget nginx certbot python3-certbot-nginx \
    python3-pip python3-venv postgresql postgresql-contrib \
    supervisor ufw

# 2. Create app user
echo -e "${YELLOW}[2/8] Creating app user...${NC}"
sudo useradd -m -s /bin/bash footypredict || true
sudo mkdir -p /app
sudo chown footypredict:footypredict /app

# 3. Setup PostgreSQL
echo -e "${YELLOW}[3/8] Setting up PostgreSQL...${NC}"
sudo -u postgres psql -c "CREATE USER footypredict WITH PASSWORD 'change_this_password';" || true
sudo -u postgres psql -c "CREATE DATABASE footypredict OWNER footypredict;" || true

# 4. Clone repository
echo -e "${YELLOW}[4/8] Cloning repository...${NC}"
sudo -u footypredict git clone https://github.com/YOUR_USERNAME/soccer.git /app/soccer || \
    (cd /app/soccer && sudo -u footypredict git pull)

# 5. Setup Python environment
echo -e "${YELLOW}[5/8] Setting up Python environment...${NC}"
cd /app/soccer
sudo -u footypredict python3 -m venv venv
sudo -u footypredict ./venv/bin/pip install --upgrade pip
sudo -u footypredict ./venv/bin/pip install -r requirements.txt
sudo -u footypredict ./venv/bin/pip install gunicorn

# 6. Create systemd service
echo -e "${YELLOW}[6/8] Creating systemd service...${NC}"
sudo tee /etc/systemd/system/footypredict.service > /dev/null << 'EOF'
[Unit]
Description=FootyPredict Pro Flask Application
After=network.target postgresql.service

[Service]
User=footypredict
Group=footypredict
WorkingDirectory=/app/soccer
Environment="PATH=/app/soccer/venv/bin"
Environment="FLASK_ENV=production"
Environment="DATABASE_URL=postgresql://footypredict:change_this_password@localhost/footypredict"
ExecStart=/app/soccer/venv/bin/gunicorn --workers 4 --bind 127.0.0.1:5000 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable footypredict
sudo systemctl start footypredict

# 7. Configure Nginx
echo -e "${YELLOW}[7/8] Configuring Nginx...${NC}"
sudo tee /etc/nginx/sites-available/footypredict > /dev/null << 'EOF'
server {
    listen 80;
    server_name YOUR_DOMAIN.com www.YOUR_DOMAIN.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    location /static {
        alias /app/soccer/static;
        expires 30d;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/footypredict /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx

# 8. Configure Firewall
echo -e "${YELLOW}[8/8] Configuring firewall...${NC}"
sudo ufw allow ssh
sudo ufw allow 'Nginx Full'
sudo ufw --force enable

echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Next steps:"
echo "1. Update domain in /etc/nginx/sites-available/footypredict"
echo "2. Update PostgreSQL password in /etc/systemd/system/footypredict.service"
echo "3. Point your domain DNS to this server IP"
echo "4. Run: sudo certbot --nginx -d YOUR_DOMAIN.com"
echo "5. Setup GitHub secrets for automated deployments"
echo ""
echo "Check status: sudo systemctl status footypredict"
echo "View logs: sudo journalctl -u footypredict -f"
