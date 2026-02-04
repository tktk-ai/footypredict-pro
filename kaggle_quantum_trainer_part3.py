# Part 3: Gradient Boosting Ensemble & Statistical Models

# ================================================================================
# SECTION 5: GRADIENT BOOSTING ENSEMBLE (ENHANCED)
# ================================================================================

class EnhancedGBEnsemble:
    """Enhanced Gradient Boosting Ensemble"""
    
    def __init__(self, n_seeds: int = 3):
        self.n_seeds = n_seeds
        self.models = {}
        self.calibrators = {}
        self.feature_importance = None
        
    def _get_catboost(self, seed: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=2000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
            random_strength=0.5, bagging_temperature=0.5, border_count=128,
            loss_function='MultiClass', eval_metric='TotalF1:average=Macro',
            early_stopping_rounds=200, verbose=False, random_state=seed,
            task_type='GPU' if torch.cuda.is_available() else 'CPU'
        )
    
    def _get_xgboost(self, seed: int) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=8, subsample=0.8,
            colsample_bytree=0.8, colsample_bylevel=0.8, reg_alpha=0.5, reg_lambda=1.5,
            min_child_weight=3, gamma=0.1, early_stopping_rounds=200, eval_metric='mlogloss',
            use_label_encoder=False, tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
            random_state=seed
        )
    
    def _get_lightgbm(self, seed: int) -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=8, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5,
            min_child_samples=20, verbose=-1, random_state=seed,
            device='gpu' if torch.cuda.is_available() else 'cpu'
        )
    
    def fit(self, X_train, y_train, X_val, y_val):
        print("Training Enhanced Gradient Boosting Ensemble...")
        all_importance = []
        
        for seed in range(self.n_seeds):
            print(f"\n  Seed {seed + 1}/{self.n_seeds}")
            
            for name, get_model in [('catboost', self._get_catboost), ('xgboost', self._get_xgboost), ('lightgbm', self._get_lightgbm)]:
                model_key = f"{name}_seed{seed}"
                model = get_model(seed)
                
                print(f"    Training {name}...", end=" ")
                
                if name == 'catboost':
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                else:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                all_importance.append(model.feature_importances_)
                
                self.calibrators[model_key] = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
                self.calibrators[model_key].fit(X_val, y_val)
                self.models[model_key] = model
                
                val_pred = self.calibrators[model_key].predict(X_val)
                acc = accuracy_score(y_val, val_pred)
                print(f"Accuracy: {acc:.4f}")
        
        self.feature_importance = np.mean(all_importance, axis=0)
        
    def predict_proba(self, X) -> np.ndarray:
        predictions = [calibrator.predict_proba(X) for calibrator in self.calibrators.values()]
        return np.mean(predictions, axis=0)
    
    def get_top_features(self, feature_names: List[str], top_n: int = 30) -> pd.DataFrame:
        return pd.DataFrame({'feature': feature_names, 'importance': self.feature_importance}).sort_values('importance', ascending=False).head(top_n)


# ================================================================================
# SECTION 6: DIXON-COLES POISSON MODEL
# ================================================================================

class DixonColesModel:
    """Dixon-Coles Model for Goal Prediction"""
    
    def __init__(self, rho: float = -0.13):
        self.rho = rho
        self.teams = {}
        self.home_advantage = 0.25
        
    def tau(self, home_goals: int, away_goals: int, lambda_h: float, mu_a: float) -> float:
        if home_goals == 0 and away_goals == 0: return 1 - lambda_h * mu_a * self.rho
        elif home_goals == 0 and away_goals == 1: return 1 + lambda_h * self.rho
        elif home_goals == 1 and away_goals == 0: return 1 + mu_a * self.rho
        elif home_goals == 1 and away_goals == 1: return 1 - self.rho
        return 1.0
    
    def fit(self, df: pd.DataFrame, time_weight: bool = True):
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
        for team in teams:
            home_scored = df[df['home_team'] == team]['home_goals'].mean()
            home_conceded = df[df['home_team'] == team]['away_goals'].mean()
            away_scored = df[df['away_team'] == team]['away_goals'].mean()
            away_conceded = df[df['away_team'] == team]['home_goals'].mean()
            self.teams[team] = {
                'attack': (home_scored + away_scored) / 2 if not np.isnan(home_scored) else 1.3,
                'defense': (home_conceded + away_conceded) / 2 if not np.isnan(home_conceded) else 1.3
            }
        self.avg_goals = df[['home_goals', 'away_goals']].mean().mean()
        self.home_advantage = (df['home_goals'].mean() - df['away_goals'].mean()) / self.avg_goals
        
    def predict(self, home_team: str, away_team: str, max_goals: int = 8) -> Dict:
        home_attack = self.teams.get(home_team, {}).get('attack', 1.0)
        away_defense = self.teams.get(away_team, {}).get('defense', 1.0)
        away_attack = self.teams.get(away_team, {}).get('attack', 1.0)
        home_defense = self.teams.get(home_team, {}).get('defense', 1.0)
        
        lambda_h = home_attack * away_defense * self.avg_goals * (1 + self.home_advantage * 0.5)
        mu_a = away_attack * home_defense * self.avg_goals * (1 - self.home_advantage * 0.25)
        
        probs = np.zeros((max_goals, max_goals))
        for h in range(max_goals):
            for a in range(max_goals):
                p_base = poisson.pmf(h, lambda_h) * poisson.pmf(a, mu_a)
                probs[h, a] = p_base * self.tau(h, a, lambda_h, mu_a)
        probs /= probs.sum()
        
        return {
            'home_win': np.triu(probs, k=1).sum(),
            'draw': np.trace(probs),
            'away_win': np.tril(probs, k=-1).sum(),
            'over_25': (probs * (np.array([[h + a for a in range(max_goals)] for h in range(max_goals)]) > 2.5)).sum(),
            'btts_yes': probs[1:, 1:].sum(),
            'home_xg': lambda_h,
            'away_xg': mu_a
        }
