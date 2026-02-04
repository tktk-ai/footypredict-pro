#!/bin/bash
# Deploy to HuggingFace Spaces

# Configuration
HF_SPACE="netboss/footypredict-v4-training"
HF_DIR="/home/netboss/Desktop/pers_bus/soccer/huggingface_training"

echo "======================================================"
echo "   FootyPredict V4 - HuggingFace Deployment"
echo "======================================================"

# Check if already logged in
if ! huggingface-cli whoami &>/dev/null; then
    echo ""
    echo "📋 Please login to HuggingFace first:"
    echo "   huggingface-cli login"
    echo ""
    exit 1
fi

echo "✅ Logged into HuggingFace"

# Create space if it doesn't exist
echo ""
echo "Creating HuggingFace Space..."
huggingface-cli repo create $HF_SPACE --type space --space_sdk gradio 2>/dev/null || echo "Space already exists"

# Clone and prepare
TEMP_DIR="/tmp/hf_deploy_$$"
rm -rf $TEMP_DIR
git clone https://huggingface.co/spaces/$HF_SPACE $TEMP_DIR 2>/dev/null || mkdir -p $TEMP_DIR

# Copy files
echo ""
echo "Copying files..."
cp -r $HF_DIR/* $TEMP_DIR/
cp $HF_DIR/README.md $TEMP_DIR/

echo ""
echo "Files to deploy:"
ls -la $TEMP_DIR/

# Push to HuggingFace
echo ""
echo "Pushing to HuggingFace Spaces..."
cd $TEMP_DIR
git lfs install
git add -A
git commit -m "V4.0 Training with 698 features - $(date +%Y-%m-%d)" || echo "No changes"
git push origin main || git push -u origin main

echo ""
echo "======================================================"
echo "✅ Deployment complete!"
echo ""
echo "🌐 View your Space: https://huggingface.co/spaces/$HF_SPACE"
echo ""
echo "📝 Notes:"
echo "   - Enable GPU in Settings > Hardware (T4 recommended)"
echo "   - First build takes ~5 minutes"
echo "======================================================"

# Cleanup
rm -rf $TEMP_DIR
