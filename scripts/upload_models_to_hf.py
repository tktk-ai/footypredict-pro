#!/usr/bin/env python3
"""
Upload models to HuggingFace Hub
================================

This script uploads trained model files to the HuggingFace Hub repository.
Run this to sync your local models with the cloud.

Usage:
    python scripts/upload_models_to_hf.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_REPO = os.getenv("HUGGINGFACE_REPO", "nananie143/footypredict-models")
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# Models to upload (priority order - most important first)
MODELS_TO_UPLOAD = [
    # SportyBet Ensemble (main production models)
    "sportybet_ensemble/result_1x2_ensemble.pkl",
    "sportybet_ensemble/btts_ensemble.pkl",
    "sportybet_ensemble/over_under_25_ensemble.pkl",
    "sportybet_ensemble/over_under_15_ensemble.pkl",
    "sportybet_ensemble/double_chance_1x_ensemble.pkl",
    "sportybet_ensemble/double_chance_x2_ensemble.pkl",
    "sportybet_ensemble/home_win_ensemble.pkl",
    "sportybet_ensemble/away_win_ensemble.pkl",
    "sportybet_ensemble/first_half_over_05_ensemble.pkl",
    "sportybet_ensemble/training_results.json",
    
    # Trained models (general)
    "trained/scaler.pkl",
    "trained/team_encoder.pkl",
    "trained/xgb_football_76features.json",
    "trained/nn_football.pt",
    "trained/training_metadata.json",
    
    # SportyBet XGBoost (lightweight)
    "trained/sportybet/btts_model.json",
    "trained/sportybet/btts_scaler.pkl",
    "trained/sportybet/over_15_model.json",
    "trained/sportybet/over_15_scaler.pkl",
    "trained/sportybet/over_25_model.json",
    "trained/sportybet/over_25_scaler.pkl",
    "trained/sportybet/dc_1x_model.json",
    "trained/sportybet/dc_1x_scaler.pkl",
    "trained/sportybet/dc_x2_model.json",
    "trained/sportybet/dc_x2_scaler.pkl",
    "trained/sportybet/dc_12_model.json",
    "trained/sportybet/dc_12_scaler.pkl",
    
    # Advanced models
    "advanced/result_1x2_model.joblib",
    "advanced/btts_model.joblib",
    "advanced/goals_over25_model.joblib",
    "advanced/calibrator.pkl",
]


def upload_models():
    """Upload all models to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("❌ huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import HfApi, login
    
    if not HF_TOKEN:
        print("❌ HUGGINGFACE_TOKEN not found in .env")
        print("   Please add: HUGGINGFACE_TOKEN=your_token_here")
        sys.exit(1)
    
    # Login to HuggingFace
    print(f"🔐 Logging in to HuggingFace...")
    login(token=HF_TOKEN)
    
    api = HfApi()
    
    # Check if repo exists
    print(f"📦 Repository: {HF_REPO}")
    
    uploaded = 0
    skipped = 0
    failed = 0
    
    for model_path in MODELS_TO_UPLOAD:
        local_path = MODELS_DIR / model_path
        
        if not local_path.exists():
            print(f"⚠️  Skip (not found): {model_path}")
            skipped += 1
            continue
        
        size_mb = local_path.stat().st_size / 1024 / 1024
        print(f"📤 Uploading: {model_path} ({size_mb:.1f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=model_path,
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"   ✅ Uploaded successfully")
            uploaded += 1
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"📊 Upload Summary:")
    print(f"   ✅ Uploaded: {uploaded}")
    print(f"   ⚠️  Skipped: {skipped}")
    print(f"   ❌ Failed: {failed}")
    print(f"{'='*50}")
    
    if uploaded > 0:
        print(f"\n🎉 Models available at: https://huggingface.co/{HF_REPO}")
        print(f"\n💡 To use in your app:")
        print(f"   from huggingface_hub import hf_hub_download")
        print(f"   model_path = hf_hub_download('{HF_REPO}', 'sportybet_ensemble/btts_ensemble.pkl')")


if __name__ == "__main__":
    upload_models()
