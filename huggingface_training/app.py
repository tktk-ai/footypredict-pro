#!/usr/bin/env python3
"""
HuggingFace Space Training App
Trains V4.0 ensemble models with 698 features using GPU
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU Detection
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_NAME = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "None"
except:
    GPU_AVAILABLE = False
    GPU_NAME = "None"

# Import training components
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_training import (
    TrainingConfig, 
    SelfImprovingTrainer,
    EnhancedTrainingPipeline
)
from enhanced_engineering import EnhancedFeatureGenerator


def load_data():
    """Load training data from uploaded files or sample."""
    data_path = Path("data/merged_training_data.parquet")
    if data_path.exists():
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} matches from storage")
        return df
    
    logger.warning("No data found, using sample data")
    return None


def generate_features(data):
    """Generate 698 features from raw data."""
    logger.info("Generating features...")
    generator = EnhancedFeatureGenerator(data)
    features = generator.generate_all_features()
    logger.info(f"Generated {len(features.columns)} features")
    return features


def train_models(features_df, data_df, config, markets):
    """Train V4.0 ensemble models."""
    logger.info("Starting model training...")
    
    pipeline = EnhancedTrainingPipeline(config)
    results = pipeline.train_all_markets(data_df, features_df, markets)
    
    return results


def format_results(results):
    """Format training results for display."""
    if not results:
        return "No results yet"
    
    output = "## Training Results\n\n"
    for market, result in results.items():
        output += f"### {market.upper()}\n"
        output += f"- **Validation Accuracy**: {result.val_accuracy:.2%}\n"
        output += f"- **Test Accuracy**: {result.test_accuracy:.2%}\n"
        output += f"- **Log Loss**: {result.log_loss_val:.4f}\n"
        output += f"- **Training Time**: {result.training_time_seconds:.1f}s\n"
        if result.suggestions:
            output += f"- **Suggestions**: {result.suggestions[0]}\n"
        output += "\n"
    
    return output


def run_training(file, n_trials, n_splits, markets):
    """Main training function for Gradio."""
    progress_log = []
    
    def log(msg):
        progress_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        return "\n".join(progress_log[-20:])
    
    yield log("🚀 Starting V4.0 Training Pipeline"), ""
    
    # Load data
    if file is not None:
        yield log(f"📁 Loading data from uploaded file..."), ""
        data = pd.read_parquet(file.name) if file.name.endswith('.parquet') else pd.read_csv(file.name)
        yield log(f"✅ Loaded {len(data)} matches"), ""
    else:
        yield log("📁 Loading stored data..."), ""
        data = load_data()
        if data is None:
            yield log("❌ No data available"), ""
            return
        yield log(f"✅ Loaded {len(data)} matches"), ""
    
    # Generate features
    yield log("🔧 Generating 698 features..."), ""
    try:
        features = generate_features(data)
        yield log(f"✅ Generated {len(features.columns)} features"), ""
    except Exception as e:
        yield log(f"❌ Feature generation failed: {e}"), ""
        return
    
    # Configure training
    config = TrainingConfig(
        n_optuna_trials=int(n_trials),
        n_splits=int(n_splits),
    )
    
    market_list = [m.strip().lower() for m in markets.split(",")]
    
    yield log(f"⚙️ Config: {n_trials} trials, {n_splits} splits, GPU: {GPU_AVAILABLE}"), ""
    yield log(f"📊 Markets: {market_list}"), ""
    
    # Train
    yield log("🎯 Training ensemble models..."), ""
    try:
        results = train_models(features, data, config, market_list)
        yield log("✅ Training complete!"), format_results(results)
    except Exception as e:
        yield log(f"❌ Training failed: {e}"), ""
        import traceback
        yield log(traceback.format_exc()), ""


# Gradio Interface
with gr.Blocks(title="FootyPredict V4 Training") as demo:
    gr.Markdown(f"""
    # ⚽ FootyPredict V4.0 - GPU Training
    
    **GPU:** {GPU_NAME} {"✅" if GPU_AVAILABLE else "❌"}
    **Features:** 698 engineered features
    **Models:** XGBoost, LightGBM, CatBoost ensemble
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Training Data (Parquet/CSV)", type="filepath")
            n_trials = gr.Slider(5, 50, value=20, step=5, label="Optuna Trials")
            n_splits = gr.Slider(2, 5, value=3, step=1, label="CV Splits")
            markets = gr.Textbox(value="result,over25,btts", label="Markets (comma-separated)")
            train_btn = gr.Button("🚀 Start Training", variant="primary")
        
        with gr.Column():
            progress = gr.Textbox(label="Progress Log", lines=15, interactive=False)
            results = gr.Markdown(label="Results")
    
    train_btn.click(
        run_training,
        inputs=[file_input, n_trials, n_splits, markets],
        outputs=[progress, results]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
