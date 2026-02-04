"""
Enhanced Self-Evolution Engine for Football Prediction Models

ENHANCED VERSION with:
- Adaptive mutation rate (decreases as fitness improves)
- Diversity preservation via crowding mechanism
- Early convergence detection and population restart
- Multi-objective optimization (accuracy vs complexity)
- Parallel fitness evaluation using multiprocessing
- Fitness sharing within similar genomes

Author: FootyPredict Pro
"""

import numpy as np
import random
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Enhanced Genome Definition
# =============================================================================

@dataclass
class ModelGenome:
    """
    Enhanced genome with complexity tracking.
    """
    model_type: str
    genes: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0.0
    adjusted_fitness: float = 0.0  # After sharing
    complexity: float = 0.0  # Model complexity score
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    id: int = 0
    stagnant_generations: int = 0
    
    def compute_complexity(self) -> float:
        """Compute model complexity based on hyperparameters"""
        complexity = 0.0
        
        # Estimators/iterations add complexity
        if 'n_estimators' in self.genes:
            complexity += self.genes['n_estimators'] / 1000
        if 'iterations' in self.genes:
            complexity += self.genes['iterations'] / 1000
        
        # Depth adds complexity
        if 'max_depth' in self.genes:
            complexity += self.genes['max_depth'] / 10
        if 'depth' in self.genes:
            complexity += self.genes['depth'] / 10
        
        # Hidden units add complexity
        if 'hidden_units' in self.genes:
            complexity += self.genes['hidden_units'] / 256
        
        self.complexity = complexity
        return complexity


# Gene ranges for each model type
GENE_RANGES = {
    'q_xgb': {
        'n_estimators': (100, 1500, 'int'),
        'max_depth': (3, 15, 'int'),
        'learning_rate': (0.001, 0.3, 'float'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
        'n_rotation_layers': (1, 5, 'int'),
        'use_vqe': (0, 1, 'bool'),
    },
    'q_lgb': {
        'n_estimators': (100, 1500, 'int'),
        'max_depth': (3, 20, 'int'),
        'learning_rate': (0.001, 0.3, 'float'),
        'num_leaves': (20, 200, 'int'),
        'subsample': (0.5, 1.0, 'float'),
        'colsample_bytree': (0.5, 1.0, 'float'),
    },
    'q_cat': {
        'iterations': (100, 1500, 'int'),
        'depth': (4, 12, 'int'),
        'learning_rate': (0.001, 0.3, 'float'),
        'l2_leaf_reg': (1.0, 10.0, 'float'),
        'n_kernel_samples': (50, 200, 'int'),
        'n_projections': (30, 100, 'int'),
        'use_reuploading': (0, 1, 'bool'),
    },
    'neat': {
        'population_size': (50, 300, 'int'),
        'generations': (20, 150, 'int'),
        'mutation_rate': (0.5, 0.95, 'float'),
        'add_node_rate': (0.01, 0.1, 'float'),
        'add_connection_rate': (0.02, 0.15, 'float'),
    },
    'deep_nn': {
        'hidden_layers': (2, 6, 'int'),
        'hidden_units': (64, 512, 'int'),
        'dropout': (0.1, 0.5, 'float'),
        'learning_rate': (0.0001, 0.01, 'float'),
        'use_attention': (0, 1, 'bool'),
        'use_residual': (0, 1, 'bool'),
    }
}


# =============================================================================
# Diversity Metrics
# =============================================================================

def genome_distance(g1: ModelGenome, g2: ModelGenome) -> float:
    """
    Compute distance between two genomes.
    Used for crowding and fitness sharing.
    """
    if g1.model_type != g2.model_type:
        return float('inf')
    
    dist = 0.0
    gene_ranges = GENE_RANGES[g1.model_type]
    
    for gene_name, (min_val, max_val, dtype) in gene_ranges.items():
        if gene_name in g1.genes and gene_name in g2.genes:
            range_size = max_val - min_val + 1e-10
            diff = abs(g1.genes[gene_name] - g2.genes[gene_name]) / range_size
            dist += diff ** 2
    
    return np.sqrt(dist)


def sharing_function(distance: float, sigma: float = 0.5) -> float:
    """
    Fitness sharing function.
    Reduces fitness for genomes that are too similar.
    """
    if distance < sigma:
        return 1 - (distance / sigma) ** 2
    return 0.0


# =============================================================================
# Enhanced Self-Evolution Engine
# =============================================================================

class SelfEvolutionEngine:
    """
    Enhanced genetic algorithm with:
    - Adaptive mutation rate
    - Diversity preservation
    - Early stopping
    - Pareto optimization
    - Parallel evaluation
    """
    
    def __init__(self,
                 population_size: int = 30,
                 generations: int = 20,
                 elite_size: int = 3,
                 initial_mutation_rate: float = 0.3,
                 min_mutation_rate: float = 0.05,
                 crossover_rate: float = 0.7,
                 tournament_size: int = 3,
                 stagnation_limit: int = 5,
                 diversity_threshold: float = 0.1,
                 use_parallel: bool = False,
                 n_workers: int = 4):
        
        self.population_size = population_size
        self.generations = generations
        self.elite_size = elite_size
        self.initial_mutation_rate = initial_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.stagnation_limit = stagnation_limit
        self.diversity_threshold = diversity_threshold
        self.use_parallel = use_parallel
        self.n_workers = n_workers
        
        # Adaptive mutation tracking
        self.current_mutation_rate = initial_mutation_rate
        self.best_fitness_seen = 0.0
        self.stagnant_generations = 0
        
        self.population: Dict[str, List[ModelGenome]] = {}
        self.best_genomes: Dict[str, ModelGenome] = {}
        self.pareto_front: Dict[str, List[ModelGenome]] = {}
        self.history: List[Dict] = []
        self.next_id = 0
    
    def _random_gene_value(self, gene_range: Tuple) -> float:
        """Generate random value within gene range"""
        min_val, max_val, dtype = gene_range
        
        if dtype == 'int':
            return random.randint(int(min_val), int(max_val))
        elif dtype == 'bool':
            return random.choice([0, 1])
        else:
            return random.uniform(min_val, max_val)
    
    def _mutate_gene(self, value: float, gene_range: Tuple) -> float:
        """Apply adaptive Gaussian mutation"""
        min_val, max_val, dtype = gene_range
        
        if dtype == 'bool':
            # Flip with probability based on mutation rate
            if random.random() < self.current_mutation_rate:
                return 1 - value
            return value
        
        # Gaussian perturbation scaled by mutation rate
        range_size = max_val - min_val
        mutation = random.gauss(0, range_size * 0.1 * (self.current_mutation_rate / 0.3))
        new_value = value + mutation
        
        # Clamp to range
        new_value = max(min_val, min(max_val, new_value))
        
        if dtype == 'int':
            return int(round(new_value))
        return new_value
    
    def _update_adaptive_mutation(self, current_best: float):
        """Update mutation rate based on progress"""
        if current_best > self.best_fitness_seen + 0.001:
            # Improvement - decrease mutation
            self.current_mutation_rate = max(
                self.min_mutation_rate,
                self.current_mutation_rate * 0.9
            )
            self.best_fitness_seen = current_best
            self.stagnant_generations = 0
        else:
            # Stagnation - increase mutation
            self.stagnant_generations += 1
            if self.stagnant_generations > 2:
                self.current_mutation_rate = min(
                    self.initial_mutation_rate,
                    self.current_mutation_rate * 1.2
                )
    
    def _compute_fitness_sharing(self, model_type: str):
        """Apply fitness sharing to maintain diversity"""
        population = self.population[model_type]
        
        for genome in population:
            niche_count = 0.0
            for other in population:
                if genome.id != other.id:
                    dist = genome_distance(genome, other)
                    niche_count += sharing_function(dist)
            
            niche_count = max(1.0, niche_count)
            genome.adjusted_fitness = genome.fitness / niche_count
    
    def _crowding_replacement(self, parent: ModelGenome, child: ModelGenome,
                               population: List[ModelGenome]) -> ModelGenome:
        """Replace most similar genome if child is better"""
        # Find most similar genome
        min_dist = float('inf')
        most_similar = None
        
        for genome in population:
            dist = genome_distance(child, genome)
            if dist < min_dist:
                min_dist = dist
                most_similar = genome
        
        if most_similar and child.fitness > most_similar.fitness:
            return child
        return most_similar
    
    def _should_restart(self) -> bool:
        """Check if population should be restarted"""
        return self.stagnant_generations >= self.stagnation_limit
    
    def _restart_population(self, model_type: str):
        """Restart population while preserving best"""
        logger.info(f"  ⚡ Restarting population (stagnation detected)")
        
        best = self.best_genomes.get(model_type)
        self.population[model_type] = []
        
        # Keep best
        if best:
            self.population[model_type].append(best)
        
        # Generate new random individuals
        while len(self.population[model_type]) < self.population_size:
            genes = {}
            for gene_name, gene_range in GENE_RANGES[model_type].items():
                genes[gene_name] = self._random_gene_value(gene_range)
            
            genome = ModelGenome(
                model_type=model_type,
                genes=genes,
                id=self.next_id
            )
            self.next_id += 1
            self.population[model_type].append(genome)
        
        self.stagnant_generations = 0
        self.current_mutation_rate = self.initial_mutation_rate
    
    def _update_pareto_front(self, model_type: str):
        """Update Pareto front for accuracy vs complexity"""
        if model_type not in self.pareto_front:
            self.pareto_front[model_type] = []
        
        # Compute complexity for all
        for genome in self.population[model_type]:
            genome.compute_complexity()
        
        # Find non-dominated solutions
        new_front = []
        candidates = self.population[model_type] + self.pareto_front[model_type]
        
        for genome in candidates:
            is_dominated = False
            for other in candidates:
                if genome.id != other.id:
                    # Check if other dominates genome
                    # (higher fitness AND lower complexity)
                    if (other.fitness >= genome.fitness and 
                        other.complexity <= genome.complexity and
                        (other.fitness > genome.fitness or other.complexity < genome.complexity)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                new_front.append(genome)
        
        # Keep only unique
        self.pareto_front[model_type] = new_front[:10]  # Limit size
    
    def initialize_population(self, model_types: List[str] = None):
        """Create random initial population"""
        if model_types is None:
            model_types = list(GENE_RANGES.keys())
        
        for model_type in model_types:
            self.population[model_type] = []
            
            for _ in range(self.population_size):
                genes = {}
                for gene_name, gene_range in GENE_RANGES[model_type].items():
                    genes[gene_name] = self._random_gene_value(gene_range)
                
                genome = ModelGenome(
                    model_type=model_type,
                    genes=genes,
                    id=self.next_id
                )
                self.next_id += 1
                self.population[model_type].append(genome)
        
        logger.info(f"Initialized population: {len(model_types)} types, {self.population_size} each")
    
    def evaluate_fitness(self, model_type: str,
                         fitness_func: Callable[[ModelGenome], float]):
        """Evaluate fitness with optional parallel processing"""
        population = self.population[model_type]
        
        if self.use_parallel and len(population) > 4:
            # Parallel evaluation
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(fitness_func, g): g for g in population}
                for future in as_completed(futures):
                    genome = futures[future]
                    try:
                        genome.fitness = future.result()
                    except Exception as e:
                        logger.warning(f"Parallel eval failed: {e}")
                        genome.fitness = 0.0
        else:
            # Sequential evaluation
            for genome in population:
                genome.fitness = fitness_func(genome)
        
        # Apply fitness sharing
        self._compute_fitness_sharing(model_type)
        
        # Track best
        best = max(population, key=lambda g: g.fitness)
        if model_type not in self.best_genomes or best.fitness > self.best_genomes[model_type].fitness:
            self.best_genomes[model_type] = best
            logger.info(f"  🏆 New best {model_type}: {best.fitness:.4f}")
        
        # Update Pareto front
        self._update_pareto_front(model_type)
    
    def _tournament_select(self, population: List[ModelGenome]) -> ModelGenome:
        """Tournament selection using adjusted fitness"""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda g: g.adjusted_fitness)
    
    def _crossover(self, parent1: ModelGenome, parent2: ModelGenome) -> ModelGenome:
        """Multi-point crossover with averaging option"""
        child_genes = {}
        
        for gene_name in parent1.genes.keys():
            if random.random() < 0.5:
                child_genes[gene_name] = parent1.genes[gene_name]
            else:
                child_genes[gene_name] = parent2.genes[gene_name]
            
            # Occasionally average (blending crossover)
            if random.random() < 0.2:
                gene_range = GENE_RANGES[parent1.model_type].get(gene_name)
                if gene_range and gene_range[2] == 'float':
                    child_genes[gene_name] = (parent1.genes[gene_name] + parent2.genes[gene_name]) / 2
        
        child = ModelGenome(
            model_type=parent1.model_type,
            genes=child_genes,
            parent_ids=[parent1.id, parent2.id],
            id=self.next_id
        )
        self.next_id += 1
        
        return child
    
    def _mutate(self, genome: ModelGenome) -> ModelGenome:
        """Apply adaptive mutation"""
        gene_ranges = GENE_RANGES[genome.model_type]
        
        for gene_name in genome.genes.keys():
            if random.random() < self.current_mutation_rate:
                genome.genes[gene_name] = self._mutate_gene(
                    genome.genes[gene_name],
                    gene_ranges[gene_name]
                )
        
        return genome
    
    def evolve_generation(self, model_type: str):
        """Evolve one generation with enhanced strategies"""
        population = self.population[model_type]
        
        # Sort by adjusted fitness
        population.sort(key=lambda g: g.adjusted_fitness, reverse=True)
        
        new_population = []
        
        # Elitism
        for i in range(min(self.elite_size, len(population))):
            elite = population[i]
            elite.generation += 1
            new_population.append(elite)
        
        # Fill with offspring
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select(population)
            parent2 = self._tournament_select(population)
            
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = ModelGenome(
                    model_type=parent1.model_type,
                    genes=parent1.genes.copy(),
                    id=self.next_id
                )
                self.next_id += 1
            
            child = self._mutate(child)
            new_population.append(child)
        
        self.population[model_type] = new_population
    
    def run_evolution(self, model_type: str,
                      fitness_func: Callable[[ModelGenome], float],
                      verbose: bool = True) -> ModelGenome:
        """Run full evolution with all enhancements"""
        
        if model_type not in self.population:
            self.initialize_population([model_type])
        
        for gen in range(self.generations):
            # Evaluate
            self.evaluate_fitness(model_type, fitness_func)
            
            # Get stats
            fitnesses = [g.fitness for g in self.population[model_type]]
            avg = np.mean(fitnesses)
            best = max(fitnesses)
            diversity = np.std(fitnesses)
            
            # Update adaptive mutation
            self._update_adaptive_mutation(best)
            
            self.history.append({
                'generation': gen,
                'model_type': model_type,
                'best_fitness': best,
                'avg_fitness': avg,
                'diversity': diversity,
                'mutation_rate': self.current_mutation_rate,
            })
            
            if verbose and (gen + 1) % 5 == 0:
                logger.info(f"  Gen {gen+1}: Best={best:.4f}, Avg={avg:.4f}, μ={self.current_mutation_rate:.3f}")
            
            # Check for restart
            if self._should_restart():
                self._restart_population(model_type)
            else:
                self.evolve_generation(model_type)
        
        # Final evaluation
        self.evaluate_fitness(model_type, fitness_func)
        
        return self.best_genomes[model_type]
    
    def run_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       model_types: List[str] = None) -> Dict[str, ModelGenome]:
        """Run evolution for all model types"""
        from .quantum_models import QuantumXGBoost, QuantumLightGBM, QuantumCatBoost
        from .neat_model import NEATFootball
        from .deep_nn import DeepFootballNet
        
        if model_types is None:
            model_types = ['q_xgb', 'q_lgb', 'q_cat']
        
        self.initialize_population(model_types)
        results = {}
        
        for model_type in model_types:
            logger.info(f"\n🧬 Evolving {model_type}...")
            
            def create_fitness_func(mt):
                def fitness(genome: ModelGenome) -> float:
                    try:
                        params = {}
                        for k, v in genome.genes.items():
                            dtype = GENE_RANGES[mt][k][2]
                            if dtype == 'int':
                                params[k] = int(v)
                            elif dtype == 'bool':
                                params[k] = bool(int(v))
                            else:
                                params[k] = v
                        
                        if mt == 'q_xgb':
                            model = QuantumXGBoost(**params)
                        elif mt == 'q_lgb':
                            model = QuantumLightGBM(**params)
                        elif mt == 'q_cat':
                            model = QuantumCatBoost(**params)
                        elif mt == 'neat':
                            model = NEATFootball(**params)
                        elif mt == 'deep_nn':
                            model = DeepFootballNet(**params)
                        else:
                            return 0.0
                        
                        model.fit(X_train, y_train)
                        preds = model.predict(X_val)
                        return (preds == y_val).mean()
                    except Exception as e:
                        logger.debug(f"Fitness eval failed: {e}")
                        return 0.0
                return fitness
            
            best = self.run_evolution(model_type, create_fitness_func(model_type))
            results[model_type] = best
            
            logger.info(f"  ✅ Best {model_type}: {best.fitness:.4f}")
        
        return results
    
    def get_best_params(self) -> Dict[str, Dict]:
        """Get best hyperparameters for each model type"""
        result = {}
        for model_type, genome in self.best_genomes.items():
            params = {}
            for k, v in genome.genes.items():
                dtype = GENE_RANGES[model_type][k][2]
                if dtype == 'int':
                    params[k] = int(v)
                elif dtype == 'bool':
                    params[k] = bool(int(v))
                else:
                    params[k] = v
            result[model_type] = params
        return result
    
    def get_pareto_front(self, model_type: str) -> List[Dict]:
        """Get Pareto-optimal solutions"""
        if model_type not in self.pareto_front:
            return []
        
        return [
            {'fitness': g.fitness, 'complexity': g.complexity, 'genes': g.genes}
            for g in self.pareto_front[model_type]
        ]
    
    def save(self, path: str):
        """Save evolution state"""
        state = {
            'best_genomes': {k: asdict(v) for k, v in self.best_genomes.items()},
            'pareto_front': {k: [asdict(g) for g in v] for k, v in self.pareto_front.items()},
            'history': self.history,
            'generations': self.generations,
            'best_fitness_seen': self.best_fitness_seen,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Saved evolution state to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'SelfEvolutionEngine':
        """Load evolution state"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        engine = cls()
        engine.history = state['history']
        engine.best_fitness_seen = state.get('best_fitness_seen', 0.0)
        engine.best_genomes = {
            k: ModelGenome(**v) for k, v in state['best_genomes'].items()
        }
        engine.pareto_front = {
            k: [ModelGenome(**g) for g in v] 
            for k, v in state.get('pareto_front', {}).items()
        }
        
        return engine


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Testing Enhanced Self-Evolution Engine...")
    
    np.random.seed(42)
    random.seed(42)
    
    X = np.random.randn(500, 20)
    y = np.random.randint(0, 3, 500)
    
    def test_fitness(genome: ModelGenome) -> float:
        # Simulate fitness with some noise
        base = random.random() * 0.5 + 0.3
        # Penalize high complexity
        complexity = genome.compute_complexity()
        return base - complexity * 0.05
    
    engine = SelfEvolutionEngine(
        population_size=15,
        generations=10,
        initial_mutation_rate=0.3
    )
    engine.initialize_population(['q_xgb'])
    best = engine.run_evolution('q_xgb', test_fitness)
    
    print(f"\n✅ Best genome: {best.genes}")
    print(f"   Fitness: {best.fitness:.4f}")
    print(f"   Complexity: {best.complexity:.4f}")
    print(f"   Pareto front size: {len(engine.pareto_front.get('q_xgb', []))}")
