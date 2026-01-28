#!/bin/bash
# Local script to manually trigger Kaggle training
# Usage: ./trigger_training.sh

set -e

KAGGLE_USER="${KAGGLE_USERNAME:-your_username}"
KERNEL="$KAGGLE_USER/footypredict-training"
DATASET="$KAGGLE_USER/footypredict-data"

echo "=== FootyPredict Manual Training Trigger ==="
echo "Kaggle User: $KAGGLE_USER"

# Check Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "Error: kaggle CLI not installed"
    echo "Install with: pip install kaggle"
    exit 1
fi

# 1. Upload latest data
echo ""
echo "[1/4] Uploading latest training data..."
cd "$(dirname "$0")/.."

mkdir -p kaggle_data
cp data/merged_training_data.parquet kaggle_data/ 2>/dev/null || echo "No new data to upload"

if [ -f kaggle_data/merged_training_data.parquet ]; then
    cat > kaggle_data/dataset-metadata.json << EOF
{
  "title": "FootyPredict Training Data",
  "id": "$DATASET",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
    kaggle datasets version -p kaggle_data -m "Manual update $(date +%Y-%m-%d)" || \
    kaggle datasets create -p kaggle_data
fi

# 2. Push notebook
echo ""
echo "[2/4] Pushing training notebook..."
mkdir -p kaggle_notebook
cp kaggle_training/footypredict_training.ipynb kaggle_notebook/

cat > kaggle_notebook/kernel-metadata.json << EOF
{
  "id": "$KERNEL",
  "title": "FootyPredict V4 Training",
  "code_file": "footypredict_training.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["$DATASET"],
  "kernel_sources": []
}
EOF

kaggle kernels push -p kaggle_notebook
echo "Notebook triggered!"

# 3. Wait for completion
echo ""
echo "[3/4] Waiting for training to complete..."
echo "This may take 20-40 minutes. Check status at:"
echo "https://www.kaggle.com/code/$KERNEL"

MAX_WAIT=3600
INTERVAL=60
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(kaggle kernels status $KERNEL 2>&1 || echo "unknown")
    echo "[$(date +%H:%M:%S)] Status: $STATUS"
    
    if echo "$STATUS" | grep -q "complete"; then
        echo "✅ Training completed!"
        break
    elif echo "$STATUS" | grep -q "error\|failed"; then
        echo "❌ Training failed!"
        exit 1
    fi
    
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

# 4. Download models
echo ""
echo "[4/4] Downloading trained models..."
mkdir -p models/v4
kaggle kernels output $KERNEL -p models/v4

echo ""
echo "=== Training Complete ==="
echo "Models saved to: models/v4/"
ls -la models/v4/

# Cleanup
rm -rf kaggle_data kaggle_notebook
