"""
HuggingFace Data Collector

Fetches football datasets and pre-trained models from HuggingFace Hub:
- JulienDelavande/soccer_stats - Betting optimization dataset
- Pre-trained soccer prediction models
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Base paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "huggingface"
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "huggingface"


class HuggingFaceCollector:
    """Fetches soccer datasets from HuggingFace Hub"""
    
    DATASETS = {
        "JulienDelavande/soccer_stats": {
            "name": "Soccer Stats for Betting",
            "description": "Historical matches, team stats, betting odds, model inference",
            "split": "train"
        }
    }
    
    MODELS = {
        "Nickel5HF/podos_soccer_model": {
            "name": "Podos Soccer Predictor",
            "description": "Transformer model for match outcome prediction",
            "input_features": 23
        }
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_id: str, split: str = "train") -> pd.DataFrame:
        """Load a dataset from HuggingFace Hub"""
        try:
            from datasets import load_dataset
            
            logger.info(f"Loading dataset: {dataset_id}")
            dataset = load_dataset(dataset_id, split=split)
            
            # Convert to pandas
            df = dataset.to_pandas()
            
            # Save locally
            output_file = self.output_dir / f"{dataset_id.replace('/', '_')}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"✓ Saved {len(df)} rows to {output_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}: {e}")
            return pd.DataFrame()
    
    def load_soccer_stats(self) -> pd.DataFrame:
        """Load the JulienDelavande/soccer_stats dataset"""
        return self.load_dataset("JulienDelavande/soccer_stats")
    
    def download_pretrained_model(self, model_id: str) -> Optional[Path]:
        """Download a pre-trained model from HuggingFace"""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading model: {model_id}")
            
            model_dir = self.models_dir / model_id.replace("/", "_")
            snapshot_download(
                repo_id=model_id,
                local_dir=model_dir,
                local_dir_use_symlinks=False
            )
            
            logger.info(f"✓ Downloaded model to {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return None
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all configured datasets"""
        results = {}
        
        for dataset_id, info in self.DATASETS.items():
            split = info.get("split", "train")
            df = self.load_dataset(dataset_id, split)
            if not df.empty:
                results[dataset_id] = df
        
        return results
    
    def download_all_models(self) -> Dict[str, Optional[Path]]:
        """Download all configured pre-trained models"""
        results = {}
        
        for model_id in self.MODELS:
            results[model_id] = self.download_pretrained_model(model_id)
        
        return results
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get all HuggingFace data combined"""
        all_dfs = []
        
        for csv_file in self.output_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {csv_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load {csv_file}: {e}")
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()


# Convenience functions
def collect_huggingface_data() -> pd.DataFrame:
    """Download and return all HuggingFace football data"""
    collector = HuggingFaceCollector()
    collector.load_all_datasets()
    return collector.get_combined_data()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    collector = HuggingFaceCollector()
    
    print("Loading HuggingFace datasets...")
    datasets = collector.load_all_datasets()
    
    for name, df in datasets.items():
        print(f"  {name}: {len(df)} rows, {len(df.columns)} columns")
    
    print("\nDownloading pre-trained models...")
    models = collector.download_all_models()
    
    for name, path in models.items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}")
