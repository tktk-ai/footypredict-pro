#!/bin/bash
# Auto-push collected training data to Kaggle Dataset
# This runs as part of the bi-weekly automation or can be run manually
#
# Usage: ./sync_data_to_kaggle.sh
#
# Prerequisites:
# - Kaggle API installed: pip install kaggle
# - API token at ~/.kaggle/kaggle.json

set -e

# Configuration
KAGGLE_USER="${KAGGLE_USERNAME:-your_username}"
DATASET_NAME="footypredict-data"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
TEMP_DIR="/tmp/kaggle_upload_$$"

echo "=== FootyPredict Data Sync to Kaggle ==="
echo "User: $KAGGLE_USER"
echo "Data dir: $DATA_DIR"

# Create temp directory
mkdir -p "$TEMP_DIR"

# Function to merge all data sources
merge_data() {
    echo "[1/4] Merging training data from all sources..."
    
    python3 << 'PYTHON_SCRIPT'
import pandas as pd
from pathlib import Path
import sys

data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/kaggle_upload")

dfs = []

# 1. Comprehensive training data (main source)
comp_path = data_dir / "comprehensive_training_data.csv"
if comp_path.exists():
    df = pd.read_csv(comp_path)
    print(f"  Loaded {len(df)} matches from comprehensive_training_data.csv")
    dfs.append(df)

# 2. Collected parquet files
collected_dir = data_dir / "collected"
if collected_dir.exists():
    for f in collected_dir.glob("*.parquet"):
        if "merged" not in f.name:  # Skip merged file
            df = pd.read_parquet(f)
            print(f"  Loaded {len(df)} matches from {f.name}")
            dfs.append(df)

# 3. Any CSV in data dir
for f in data_dir.glob("*.csv"):
    if f.name not in ["comprehensive_training_data.csv", "training_data.csv"]:
        try:
            df = pd.read_csv(f)
            if len(df) > 100:
                print(f"  Loaded {len(df)} matches from {f.name}")
                dfs.append(df)
        except:
            pass

# Merge all (concat with dedup by key columns if available)
if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    
    # Try to deduplicate
    if all(col in merged.columns for col in ["Date", "HomeTeam", "AwayTeam"]):
        before = len(merged)
        merged = merged.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
        print(f"  Deduplicated: {before} -> {len(merged)} matches")
    
    # Save
    output_path = output_dir / "merged_training_data.parquet"
    merged.to_parquet(output_path, index=False)
    print(f"\n✅ Saved {len(merged)} matches to {output_path}")
    
    # Also save CSV for Kaggle compatibility
    csv_path = output_dir / "merged_training_data.csv"
    merged.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV version to {csv_path}")
else:
    print("❌ No data found!")
    sys.exit(1)
PYTHON_SCRIPT
}

# Merge data
cd "$PROJECT_ROOT"
python3 -c "
import pandas as pd
from pathlib import Path
import sys

data_dir = Path('$DATA_DIR')
output_dir = Path('$TEMP_DIR')

dfs = []

# Load comprehensive data
comp_path = data_dir / 'comprehensive_training_data.csv'
if comp_path.exists():
    df = pd.read_csv(comp_path)
    print(f'  Loaded {len(df)} matches from comprehensive_training_data.csv')
    dfs.append(df)

# Load collected parquets
collected_dir = data_dir / 'collected'
if collected_dir.exists():
    for f in collected_dir.glob('*.parquet'):
        if 'merged' not in f.name:
            df = pd.read_parquet(f)
            print(f'  Loaded {len(df)} matches from {f.name}')
            dfs.append(df)

# Merge
if dfs:
    merged = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate if possible
    key_cols = ['Date', 'HomeTeam', 'AwayTeam']
    if all(col in merged.columns for col in key_cols):
        before = len(merged)
        merged = merged.drop_duplicates(subset=key_cols)
        print(f'  Deduplicated: {before} -> {len(merged)} matches')
    
    # Save both formats
    merged.to_parquet(output_dir / 'merged_training_data.parquet', index=False)
    merged.to_csv(output_dir / 'merged_training_data.csv', index=False)
    print(f'\\n✅ Merged {len(merged)} total matches')
else:
    print('❌ No data found')
    sys.exit(1)
"

echo ""
echo "[2/4] Creating Kaggle dataset metadata..."
cat > "$TEMP_DIR/dataset-metadata.json" << EOF
{
  "title": "FootyPredict Training Data",
  "id": "$KAGGLE_USER/$DATASET_NAME",
  "licenses": [{"name": "CC0-1.0"}],
  "keywords": ["football", "soccer", "prediction", "machine-learning"]
}
EOF

echo ""
echo "[3/4] Uploading to Kaggle..."
cd "$TEMP_DIR"

# Check if dataset exists
if kaggle datasets status "$KAGGLE_USER/$DATASET_NAME" &>/dev/null; then
    echo "  Updating existing dataset..."
    kaggle datasets version -p . -m "Auto-sync $(date +%Y-%m-%d_%H:%M)"
else
    echo "  Creating new dataset..."
    kaggle datasets create -p . --public
fi

echo ""
echo "[4/4] Cleanup..."
rm -rf "$TEMP_DIR"

echo ""
echo "=== Sync Complete ==="
echo "Dataset: https://kaggle.com/datasets/$KAGGLE_USER/$DATASET_NAME"
