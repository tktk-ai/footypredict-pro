"""
Enhanced Quantum Sequential Training Pipeline

ENHANCED VERSION with:
- Real-time progress dashboard with progress bars
- Memory-efficient batch processing
- Hardware detection (GPU/CPU)
- Cross-validation support
- Model versioning and best model tracking
- Rich console output

Author: FootyPredict Pro
"""

import numpy as np
import pandas as pd
import json
import pickle
import gc
import psutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Console Progress Display
# =============================================================================

class ProgressDisplay:
    """Rich console progress display"""
    
    def __init__(self, total_steps: int, description: str = ""):
        self.total = total_steps
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: int = 1, status: str = ""):
        self.current += step
        self._render(status)
    
    def _render(self, status: str = ""):
        percent = min(100, (self.current / self.total) * 100)
        filled = int(percent // 2)
        bar = "█" * filled + "░" * (50 - filled)
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eta = (elapsed / self.current * (self.total - self.current)) if self.current > 0 else 0
        
        line = f"\r  [{bar}] {percent:5.1f}% | {self.current}/{self.total}"
        if status:
            line += f" | {status}"
        if eta > 0:
            line += f" | ETA: {eta:.0f}s"
        
        sys.stdout.write(line + " " * 10)
        sys.stdout.flush()
    
    def finish(self):
        self.current = self.total
        self._render("Done!")
        print()


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_hardware_info() -> Dict[str, Any]:
    """Detect available hardware"""
    import torch
    
    info = {
        'cpu_count': psutil.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / 1024**3,
        'memory_available_gb': psutil.virtual_memory().available / 1024**3,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return info


# =============================================================================
# Enhanced Sequential Trainer
# =============================================================================

class QuantumSequentialTrainer:
    """
    Enhanced training orchestrator with:
    - Progress visualization
    - Cross-validation
    - Memory management
    - Hardware detection
    """
    
    MODELS = [
        ('q_xgb', 'QuantumXGBoost', '🔄 Quantum XGBoost (Pauli Gates + VQE)'),
        ('q_lgb', 'QuantumLightGBM', '📊 Quantum LightGBM (Amplitude Encoding)'),
        ('q_cat', 'QuantumCatBoost', '⚛️ Quantum CatBoost (Quantum Kernels)'),
        ('neat', 'NEATFootball', '🧬 NEAT Neuroevolution (Dynamic Topology)'),
        ('deep_nn', 'DeepFootballNet', '🧠 Deep NN (Residual + Attention)'),
    ]
    
    def __init__(self, 
                 output_dir: str = './models/quantum_trained',
                 use_evolution: bool = True,
                 evolution_generations: int = 10,
                 use_cv: bool = False,
                 n_folds: int = 5,
                 batch_size: int = 10000):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_evolution = use_evolution
        self.evolution_generations = evolution_generations
        self.use_cv = use_cv
        self.n_folds = n_folds
        self.batch_size = batch_size
        
        self.trained_models: Dict[str, Any] = {}
        self.training_results: Dict[str, Dict] = {}
        self.checkpoint_file = self.output_dir / 'checkpoint.json'
        self.completed_models: List[str] = []
        self.model_versions: Dict[str, int] = {}
        
        self._detect_hardware()
        self._load_checkpoint()
    
    def _detect_hardware(self):
        """Detect and log hardware capabilities"""
        try:
            self.hardware = get_hardware_info()
            logger.info("🖥️ Hardware Detection:")
            logger.info(f"   CPUs: {self.hardware['cpu_count']}")
            logger.info(f"   RAM: {self.hardware['memory_total_gb']:.1f} GB (available: {self.hardware['memory_available_gb']:.1f} GB)")
            if self.hardware['cuda_available']:
                logger.info(f"   GPU: {self.hardware['cuda_device_name']} ({self.hardware['cuda_device_count']} device(s))")
            else:
                logger.info("   GPU: Not available (CPU training)")
        except Exception as e:
            logger.warning(f"Hardware detection failed: {e}")
            self.hardware = {}
    
    def _load_checkpoint(self):
        """Load training checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            self.completed_models = checkpoint.get('completed_models', [])
            self.training_results = checkpoint.get('results', {})
            self.model_versions = checkpoint.get('versions', {})
            logger.info(f"📂 Checkpoint loaded: {len(self.completed_models)} models completed")
    
    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            'completed_models': self.completed_models,
            'results': self.training_results,
            'versions': self.model_versions,
            'timestamp': datetime.now().isoformat(),
            'hardware': self.hardware,
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare training data with memory efficiency"""
        from train_comprehensive import download_comprehensive_data, engineer_all_features
        
        logger.info("\n📥 Loading and preparing data...")
        
        raw_data = download_comprehensive_data()
        df, feature_cols, team_encoder = engineer_all_features(raw_data)
        
        X = df[feature_cols].values.astype(np.float32)  # Memory efficient
        y = df['Result'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        memory_used = get_memory_usage()
        logger.info(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
        logger.info(f"   Features: {X.shape[1]} | Memory: {memory_used:.0f} MB")
        
        return X_train, X_test, y_train, y_test
    
    def _get_model_class(self, model_name: str):
        """Get model class by name"""
        if model_name == 'QuantumXGBoost':
            from .quantum_models import QuantumXGBoost
            return QuantumXGBoost
        elif model_name == 'QuantumLightGBM':
            from .quantum_models import QuantumLightGBM
            return QuantumLightGBM
        elif model_name == 'QuantumCatBoost':
            from .quantum_models import QuantumCatBoost
            return QuantumCatBoost
        elif model_name == 'NEATFootball':
            from .neat_model import NEATFootball
            return NEATFootball
        elif model_name == 'DeepFootballNet':
            from .deep_nn import DeepFootballNet
            return DeepFootballNet
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def train_with_cv(self, model_key: str, ModelClass: Any,
                      X: np.ndarray, y: np.ndarray,
                      params: Dict = None) -> Dict:
        """Train with cross-validation"""
        logger.info(f"   Running {self.n_folds}-fold cross-validation...")
        
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        scores = []
        
        progress = ProgressDisplay(self.n_folds, f"{model_key} CV")
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = ModelClass(**(params or {}))
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            score = (preds == y_val).mean()
            scores.append(score)
            
            progress.update(1, f"Fold {fold+1}: {score:.2%}")
            
            # Memory cleanup
            del model
            gc.collect()
        
        progress.finish()
        
        return {
            'cv_scores': scores,
            'cv_mean': np.mean(scores),
            'cv_std': np.std(scores),
        }
    
    def train_single_model(self, model_key: str, model_class_name: str,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           params: Dict = None) -> Dict:
        """Train a single model with enhanced tracking"""
        
        logger.info(f"\n{'─'*60}")
        logger.info(f"🔧 Training {model_key}")
        logger.info(f"{'─'*60}")
        
        start_time = datetime.now()
        start_memory = get_memory_usage()
        
        try:
            ModelClass = self._get_model_class(model_class_name)
            
            # Cross-validation (optional)
            cv_result = {}
            if self.use_cv:
                cv_result = self.train_with_cv(model_key, ModelClass, 
                                                np.vstack([X_train, X_test]),
                                                np.hstack([y_train, y_test]),
                                                params)
            
            # Final training
            logger.info("   Training on full dataset...")
            model = ModelClass(**(params or {}))
            model.fit(X_train, y_train)
            
            # Evaluate
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            train_acc = (train_preds == y_train).mean()
            test_acc = (test_preds == y_test).mean()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            memory_delta = get_memory_usage() - start_memory
            
            # Version tracking
            version = self.model_versions.get(model_key, 0) + 1
            self.model_versions[model_key] = version
            
            # Save model
            model_path = self.output_dir / f"{model_key}_model_v{version}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Also save as latest
            latest_path = self.output_dir / f"{model_key}_model.pkl"
            with open(latest_path, 'wb') as f:
                pickle.dump(model, f)
            
            result = {
                'model_key': model_key,
                'version': version,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'duration_seconds': duration,
                'memory_delta_mb': memory_delta,
                'model_path': str(latest_path),
                'versioned_path': str(model_path),
                'trained_at': end_time.isoformat(),
                'params': params or {},
                **cv_result,
            }
            
            # Display results
            logger.info(f"   ✅ Train: {train_acc:.2%} | Test: {test_acc:.2%}")
            if cv_result:
                logger.info(f"   📊 CV: {cv_result['cv_mean']:.2%} ± {cv_result['cv_std']:.2%}")
            logger.info(f"   ⏱️ Duration: {duration:.1f}s | Memory: +{memory_delta:.0f} MB")
            logger.info(f"   💾 Saved: {model_path.name}")
            
            self.trained_models[model_key] = model
            self.training_results[model_key] = result
            
            if model_key not in self.completed_models:
                self.completed_models.append(model_key)
            self._save_checkpoint()
            
            return result
            
        except Exception as e:
            logger.error(f"   ❌ Training failed: {e}")
            return {'error': str(e), 'model_key': model_key}
    
    def train_all_sequential(self, skip_completed: bool = True) -> Dict[str, Dict]:
        """Train all models sequentially with progress tracking"""
        
        print("\n" + "="*70)
        print("🚀 QUANTUM SEQUENTIAL TRAINING PIPELINE (ENHANCED)")
        print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Overall progress
        models_to_train = [m for m in self.MODELS if not (skip_completed and m[0] in self.completed_models)]
        total_models = len(models_to_train)
        
        print(f"\n📋 Training {total_models} models ({len(self.completed_models)} already completed)")
        
        for idx, (model_key, model_class, description) in enumerate(self.MODELS):
            if skip_completed and model_key in self.completed_models:
                print(f"\n⏭️ Skipping {model_key} (completed)")
                continue
            
            print(f"\n[{idx+1}/{len(self.MODELS)}] {description}")
            
            try:
                result = self.train_single_model(
                    model_key, model_class,
                    X_train, y_train,
                    X_test, y_test
                )
            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue
            
            # Memory cleanup
            gc.collect()
        
        # Self-evolution (optional)
        if self.use_evolution:
            self._run_evolution(X_train, y_train, X_test, y_test)
        
        # Summary
        self._print_summary()
        
        return self.training_results
    
    def _run_evolution(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray):
        """Run self-evolution optimization"""
        from .evolution_engine import SelfEvolutionEngine
        
        print("\n" + "="*60)
        print("🧬 SELF-EVOLUTION PHASE")
        print("="*60)
        
        engine = SelfEvolutionEngine(
            population_size=20,
            generations=self.evolution_generations,
            initial_mutation_rate=0.3
        )
        
        best_params = engine.run_all_models(
            X_train, y_train, X_val, y_val,
            model_types=['q_xgb', 'q_lgb', 'q_cat']
        )
        
        evolution_path = self.output_dir / 'evolved_params.json'
        engine.save(str(evolution_path))
        
        logger.info(f"   ✅ Evolved parameters saved: {evolution_path.name}")
    
    def _print_summary(self):
        """Print training summary dashboard"""
        print("\n" + "="*70)
        print("📊 TRAINING SUMMARY DASHBOARD")
        print("="*70)
        
        print(f"{'Model':<20} {'Train':>8} {'Test':>8} {'CV':>12} {'Time':>8} {'Version':>8}")
        print("-"*70)
        
        for model_key, result in self.training_results.items():
            if 'error' in result:
                print(f"{model_key:<20} {'FAILED':>8}")
            else:
                cv_str = f"{result.get('cv_mean', 0):.1%}±{result.get('cv_std', 0):.1%}" if 'cv_mean' in result else "N/A"
                print(f"{model_key:<20} {result['train_accuracy']:>7.1%} {result['test_accuracy']:>7.1%} {cv_str:>12} {result['duration_seconds']:>7.0f}s v{result.get('version', 1):>3}")
        
        print("-"*70)
        
        # Best model
        valid_results = {k: v for k, v in self.training_results.items() if 'test_accuracy' in v}
        if valid_results:
            best_key = max(valid_results, key=lambda k: valid_results[k]['test_accuracy'])
            print(f"🏆 Best: {best_key} ({valid_results[best_key]['test_accuracy']:.2%})")
        
        print("="*70)
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """Calculate ensemble weights based on performance"""
        valid_results = {k: v for k, v in self.training_results.items() if 'test_accuracy' in v}
        total_acc = sum(r['test_accuracy'] for r in valid_results.values())
        
        return {k: v['test_accuracy'] / total_acc for k, v in valid_results.items()}
    
    def predict_ensemble(self, X: np.ndarray,
                         weights: Dict[str, float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble prediction"""
        if weights is None:
            weights = self.get_ensemble_weights()
        
        if not self.trained_models:
            for model_key, result in self.training_results.items():
                if 'model_path' in result:
                    with open(result['model_path'], 'rb') as f:
                        self.trained_models[model_key] = pickle.load(f)
        
        all_probs = []
        all_weights = []
        
        for model_key, model in self.trained_models.items():
            if model_key in weights:
                try:
                    probs = model.predict_proba(X)
                    all_probs.append(probs)
                    all_weights.append(weights[model_key])
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_key}: {e}")
        
        if not all_probs:
            raise ValueError("No models available")
        
        all_weights = np.array(all_weights) / sum(all_weights)
        ensemble_probs = sum(w * p for w, p in zip(all_weights, all_probs))
        predictions = ensemble_probs.argmax(axis=1)
        
        return predictions, ensemble_probs


# =============================================================================
# Entry Point
# =============================================================================

def run_quantum_training(skip_completed: bool = True,
                         use_evolution: bool = False,
                         use_cv: bool = False) -> Dict:
    """Main entry point for quantum training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    trainer = QuantumSequentialTrainer(
        use_evolution=use_evolution,
        use_cv=use_cv
    )
    
    return trainer.train_all_sequential(skip_completed=skip_completed)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Training Pipeline')
    parser.add_argument('--fresh', action='store_true', help='Start fresh (ignore checkpoint)')
    parser.add_argument('--evolve', action='store_true', help='Run self-evolution')
    parser.add_argument('--cv', action='store_true', help='Use cross-validation')
    args = parser.parse_args()
    
    results = run_quantum_training(
        skip_completed=not args.fresh,
        use_evolution=args.evolve,
        use_cv=args.cv
    )
