# Part 4: Complete System & Main Execution

# ================================================================================
# SECTION 7: COMPLETE SYSTEM
# ================================================================================

class UltraAdvancedFootballPredictor:
    """ULTRA-ADVANCED FOOTBALL PREDICTION SYSTEM v3.0"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*80)
        print("INITIALIZING ULTRA-ADVANCED QUANTUM FOOTBALL PREDICTION SYSTEM v3.0")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Input dimensions: {config.input_dim}")
        print(f"Quantum qubits: {config.n_qubits}")
        print("="*80)
        
        self._init_preprocessors()
        self._init_quantum_components()
        self._init_neural_components()
        self._init_ensemble_components()
        
        print("\nSystem initialized successfully!")
        
    def _init_preprocessors(self):
        print("\n[1/4] Initializing preprocessors...")
        self.odds_processor = AdvancedOddsProcessor()
        self.scaler = RobustScaler()
        
    def _init_quantum_components(self):
        print("[2/4] Initializing QUANTUM components (CORE)...")
        self.quantum_transformer = HybridQuantumTransformer(
            input_dim=self.config.input_dim, d_model=128, n_heads=8, n_layers=4,
            n_qubits=self.config.n_qubits, n_quantum_layers=self.config.n_quantum_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.quantum_classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(64, self.config.n_classes)
        ).to(self.device)
        
    def _init_neural_components(self):
        print("[3/4] Initializing neural components...")
        self.deep_cross = nn.Sequential(
            DeepCrossNetwork(self.config.input_dim, n_cross_layers=4),
            nn.Linear(self.config.input_dim, 128), nn.GELU(), nn.Dropout(self.config.dropout)
        ).to(self.device)
        
        self.moe = MixtureOfExperts(input_dim=128, output_dim=64, n_experts=8, top_k=2).to(self.device)
        
        self.deep_ensemble = DeepEnsemble(
            input_dim=self.config.input_dim, hidden_dims=[256, 128, 64],
            n_classes=self.config.n_classes, n_networks=5, dropout=self.config.dropout
        ).to(self.device)
        
    def _init_ensemble_components(self):
        print("[4/4] Initializing gradient boosting ensemble...")
        self.gb_ensemble = EnhancedGBEnsemble(n_seeds=self.config.n_seeds)
        
        self.meta_learner = nn.Sequential(
            nn.Linear(self.config.n_classes * 4, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, self.config.n_classes)
        ).to(self.device)
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        print("\nPreprocessing data...")
        target_cols = ['home_goals', 'away_goals']
        feature_cols = [c for c in df.columns if c not in target_cols]
        
        processed = self.odds_processor.process_all_features(df[feature_cols])
        
        if 'home_goals' in df.columns and 'away_goals' in df.columns:
            y = np.where(df['home_goals'] > df['away_goals'], 0,
                        np.where(df['home_goals'] == df['away_goals'], 1, 2))
        else:
            y = None
        
        X = self.scaler.fit_transform(processed.values)
        self.feature_names = processed.columns.tolist()
        print(f"Processed features: {X.shape[1]}")
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        print("\n" + "="*80)
        print("TRAINING ULTRA-ADVANCED QUANTUM SYSTEM")
        print("="*80)
        print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Features: {X_train.shape[1]}")
        
        results = {}
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.LongTensor(y_val).to(self.device)
        
        class_counts = np.bincount(y_train)
        sample_weights = (1.0 / class_counts)[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(y_train), replacement=True)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, sampler=sampler)
        
        # PHASE 1: Gradient Boosting
        print("\n" + "-"*40 + "\nPHASE 1: Training Gradient Boosting\n" + "-"*40)
        self.gb_ensemble.fit(X_train, y_train, X_val, y_val)
        gb_val_pred = self.gb_ensemble.predict_proba(X_val)
        results['gb_accuracy'] = accuracy_score(y_val, gb_val_pred.argmax(axis=1))
        print(f"\nGB Ensemble Accuracy: {results['gb_accuracy']:.4f}")
        
        # PHASE 2: Quantum Neural Network
        print("\n" + "-"*40 + "\nPHASE 2: Training QUANTUM NN (CORE)\n" + "-"*40)
        quantum_params = list(self.quantum_transformer.parameters()) + list(self.quantum_classifier.parameters())
        optimizer = torch.optim.AdamW(quantum_params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(1.0 / class_counts).to(self.device))
        
        best_quantum_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            self.quantum_transformer.train()
            self.quantum_classifier.train()
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                features = self.quantum_transformer(batch_x)
                outputs = self.quantum_classifier(features)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(quantum_params, max_norm=1.0)
                optimizer.step()
            scheduler.step()
            
            self.quantum_transformer.eval()
            self.quantum_classifier.eval()
            with torch.no_grad():
                val_features = self.quantum_transformer(X_val_t)
                val_outputs = self.quantum_classifier(val_features)
                val_acc = (val_outputs.argmax(dim=1) == y_val_t).float().mean().item()
            
            if val_acc > best_quantum_acc:
                best_quantum_acc = val_acc
                torch.save({'transformer': self.quantum_transformer.state_dict(), 'classifier': self.quantum_classifier.state_dict()}, 'best_quantum_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs} - Acc: {val_acc:.4f} - Best: {best_quantum_acc:.4f}")
            
            if patience_counter >= 30: break
        
        checkpoint = torch.load('best_quantum_model.pt')
        self.quantum_transformer.load_state_dict(checkpoint['transformer'])
        self.quantum_classifier.load_state_dict(checkpoint['classifier'])
        results['quantum_accuracy'] = best_quantum_acc
        
        # PHASE 3: Deep Ensemble
        print("\n" + "-"*40 + "\nPHASE 3: Training Deep Ensemble\n" + "-"*40)
        ensemble_optimizer = torch.optim.AdamW(self.deep_ensemble.parameters(), lr=self.config.learning_rate)
        best_ensemble_acc = 0
        
        for epoch in range(50):
            self.deep_ensemble.train()
            for batch_x, batch_y in train_loader:
                ensemble_optimizer.zero_grad()
                pred, _ = self.deep_ensemble(batch_x, mc_samples=1)
                loss = F.cross_entropy(torch.log(pred + 1e-8), batch_y)
                loss.backward()
                ensemble_optimizer.step()
            
            self.deep_ensemble.eval()
            with torch.no_grad():
                val_pred, _ = self.deep_ensemble(X_val_t, mc_samples=5)
                val_acc = (val_pred.argmax(dim=1) == y_val_t).float().mean().item()
            
            if val_acc > best_ensemble_acc:
                best_ensemble_acc = val_acc
                torch.save(self.deep_ensemble.state_dict(), 'best_deep_ensemble.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/50 - Acc: {val_acc:.4f}")
        
        self.deep_ensemble.load_state_dict(torch.load('best_deep_ensemble.pt'))
        results['deep_ensemble_accuracy'] = best_ensemble_acc
        
        # PHASE 4: Meta-Learner
        print("\n" + "-"*40 + "\nPHASE 4: Training Meta-Learner\n" + "-"*40)
        self.quantum_transformer.eval()
        self.quantum_classifier.eval()
        self.deep_ensemble.eval()
        
        with torch.no_grad():
            q_pred = F.softmax(self.quantum_classifier(self.quantum_transformer(X_train_t)), dim=1)
            q_val_pred = F.softmax(self.quantum_classifier(self.quantum_transformer(X_val_t)), dim=1)
            de_pred, _ = self.deep_ensemble(X_train_t, mc_samples=5)
            de_val_pred, _ = self.deep_ensemble(X_val_t, mc_samples=5)
        
        gb_train_pred = torch.FloatTensor(self.gb_ensemble.predict_proba(X_train)).to(self.device)
        gb_val_pred_t = torch.FloatTensor(gb_val_pred).to(self.device)
        
        avg_train = (q_pred + de_pred + gb_train_pred) / 3
        avg_val = (q_val_pred + de_val_pred + gb_val_pred_t) / 3
        
        meta_train_input = torch.cat([q_pred, de_pred, gb_train_pred, avg_train], dim=1)
        meta_val_input = torch.cat([q_val_pred, de_val_pred, gb_val_pred_t, avg_val], dim=1)
        
        meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=1e-3)
        
        for epoch in range(100):
            self.meta_learner.train()
            meta_optimizer.zero_grad()
            outputs = self.meta_learner(meta_train_input)
            loss = F.cross_entropy(outputs, y_train_t)
            loss.backward()
            meta_optimizer.step()
        
        self.meta_learner.eval()
        with torch.no_grad():
            final_outputs = self.meta_learner(meta_val_input)
            final_probs = F.softmax(final_outputs, dim=1)
            final_pred = final_outputs.argmax(dim=1)
            results['meta_accuracy'] = (final_pred == y_val_t).float().mean().item()
        
        # Confidence Analysis
        print("\n" + "-"*40 + "\nConfidence Analysis\n" + "-"*40)
        confidence_scores = final_probs.max(dim=1)[0].cpu().numpy()
        
        for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            mask = confidence_scores >= thresh
            if mask.sum() > 0:
                acc = accuracy_score(y_val[mask], final_pred.cpu().numpy()[mask])
                print(f"Threshold >= {thresh:.2f}: Acc = {acc:.4f} (Coverage: {mask.mean()*100:.1f}%)")
                if thresh == self.config.confidence_threshold:
                    results['high_conf_accuracy'] = acc
                    results['high_conf_coverage'] = mask.mean() * 100
        
        print("\n" + "="*80 + "\nTRAINING COMPLETE\n" + "="*80)
        print(f"GB: {results['gb_accuracy']:.4f} | Quantum: {results['quantum_accuracy']:.4f}")
        print(f"Ensemble: {results['deep_ensemble_accuracy']:.4f} | Meta: {results['meta_accuracy']:.4f}")
        
        return results
    
    def predict(self, X: np.ndarray, return_confidence: bool = True):
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.quantum_transformer.eval()
        self.quantum_classifier.eval()
        self.deep_ensemble.eval()
        self.meta_learner.eval()
        
        with torch.no_grad():
            q_pred = F.softmax(self.quantum_classifier(self.quantum_transformer(X_t)), dim=1)
            de_pred, _ = self.deep_ensemble(X_t, mc_samples=10)
            gb_pred = torch.FloatTensor(self.gb_ensemble.predict_proba(X)).to(self.device)
            avg_pred = (q_pred + de_pred + gb_pred) / 3
            
            meta_input = torch.cat([q_pred, de_pred, gb_pred, avg_pred], dim=1)
            final_probs = F.softmax(self.meta_learner(meta_input), dim=1)
        
        probabilities = final_probs.cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        confidence = probabilities.max(axis=1)
        
        return (predictions, probabilities, confidence) if return_confidence else predictions


# ================================================================================
# SECTION 8: MAIN EXECUTION FOR KAGGLE
# ================================================================================

def main():
    print("="*80)
    print("ULTRA-ADVANCED QUANTUM FOOTBALL PREDICTION SYSTEM v3.0")
    print("="*80)
    
    # Load Kaggle data
    df = pd.read_csv('/kaggle/input/football-match-prediction-features/kaggle_training_data.csv')
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    config = ModelConfig(
        input_dim=200, n_classes=3, n_qubits=10, n_quantum_layers=4,
        batch_size=256, epochs=150, learning_rate=1e-3,
        confidence_threshold=0.55, n_folds=5, n_seeds=3
    )
    
    predictor = UltraAdvancedFootballPredictor(config)
    X, y = predictor.preprocess_data(df)
    
    # Update input_dim
    config.input_dim = X.shape[1]
    predictor = UltraAdvancedFootballPredictor(config)
    predictor.scaler = RobustScaler()
    X = predictor.scaler.fit_transform(X)
    
    # Temporal split
    split_idx = int(0.8 * len(X))
    X_train_full, X_test = X[:split_idx], X[split_idx:]
    y_train_full, y_test = y[:split_idx], y[split_idx:]
    
    val_split = int(0.9 * len(X_train_full))
    X_train, X_val = X_train_full[:val_split], X_train_full[val_split:]
    y_train, y_val = y_train_full[:val_split], y_train_full[val_split:]
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Train
    results = predictor.train(X_train, y_train, X_val, y_val)
    
    # Test
    print("\n" + "="*80 + "\nTEST SET EVALUATION\n" + "="*80)
    test_pred, test_probs, test_conf = predictor.predict(X_test)
    
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro')
    print(f"Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f}")
    
    conf_mask = test_conf >= config.confidence_threshold
    if conf_mask.sum() > 0:
        print(f"High Confidence (>={config.confidence_threshold:.0%}): {accuracy_score(y_test[conf_mask], test_pred[conf_mask]):.4f} ({conf_mask.mean()*100:.1f}%)")
    
    # Save model
    torch.save({
        'quantum_transformer': predictor.quantum_transformer.state_dict(),
        'quantum_classifier': predictor.quantum_classifier.state_dict(),
        'deep_ensemble': predictor.deep_ensemble.state_dict(),
        'meta_learner': predictor.meta_learner.state_dict(),
        'config': config
    }, 'quantum_football_predictor.pt')
    
    return predictor, results

if __name__ == "__main__":
    predictor, results = main()
