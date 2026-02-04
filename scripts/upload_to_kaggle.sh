#!/bin/bash
# Upload training data to Kaggle for Quantum Model training
# Run this BEFORE starting the Kaggle kernel

set -e

echo "📦 Preparing data for Kaggle upload..."

# Check kaggle is configured
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "❌ Kaggle API not configured!"
    echo "Please run: kaggle configure"
    exit 1
fi

# Create dataset directory
DATASET_DIR="kaggle_dataset"
mkdir -p $DATASET_DIR

# Copy training data
echo "📁 Copying training data..."
cp data/comprehensive_training_data.csv $DATASET_DIR/training_data.csv

# Create dataset metadata
cat > $DATASET_DIR/dataset-metadata.json << 'EOF'
{
    "title": "Football Prediction Training Data",
    "id": "YOUR_USERNAME/football-training-data",
    "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo ""
echo "⚠️  IMPORTANT: Edit dataset-metadata.json and replace YOUR_USERNAME"
echo ""
echo "Then run:"
echo "  cd kaggle_dataset"
echo "  kaggle datasets create -p ."
echo ""
echo "After upload, add the dataset to your Kaggle notebook and run!"
