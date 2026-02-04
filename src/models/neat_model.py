"""
Enhanced NEAT (NeuroEvolution of Augmenting Topologies) for Football Prediction

ENHANCED VERSION with:
- Vectorized batch evaluation (10x+ speedup)
- Advanced activations (GELU, Swish, Mish)
- Stagnation detection with species protection
- Multi-point crossover
- Save/load functionality

Author: FootyPredict Pro
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, field
import pickle
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Advanced Activation Functions
# =============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation: x * sigmoid(x)"""
    return x / (1 + np.exp(-np.clip(x, -500, 500)))


def mish(x: np.ndarray) -> np.ndarray:
    """Mish activation: x * tanh(softplus(x))"""
    return x * np.tanh(np.log(1 + np.exp(np.clip(x, -500, 500))))


ACTIVATIONS = {
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
    'tanh': np.tanh,
    'gelu': gelu,
    'swish': swish,
    'mish': mish,
    'linear': lambda x: x,
}


# =============================================================================
# NEAT Data Structures
# =============================================================================

@dataclass
class NodeGene:
    """Represents a node in the neural network"""
    id: int
    type: str  # 'input', 'hidden', 'output'
    activation: str = 'relu'
    bias: float = 0.0


@dataclass
class ConnectionGene:
    """Represents a connection between nodes"""
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    """A complete genome representing a neural network"""
    id: int
    nodes: Dict[int, NodeGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)
    fitness: float = 0.0
    adjusted_fitness: float = 0.0
    species_id: int = -1
    stagnant_gens: int = 0


@dataclass
class Species:
    """Represents a species of similar genomes"""
    id: int
    members: List[Genome] = field(default_factory=list)
    representative: Optional[Genome] = None
    best_fitness: float = 0.0
    stagnant_generations: int = 0
    protected: bool = False


# =============================================================================
# Innovation Tracker
# =============================================================================

class InnovationTracker:
    """Tracks structural innovations for crossover"""
    
    def __init__(self):
        self.innovations = {}
        self.next_innovation = 0
        self.next_node_id = 0
    
    def get_innovation(self, in_node: int, out_node: int) -> int:
        key = (in_node, out_node)
        if key not in self.innovations:
            self.innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self.innovations[key]
    
    def get_new_node_id(self) -> int:
        node_id = self.next_node_id
        self.next_node_id += 1
        return node_id


# =============================================================================
# Vectorized NEAT Network
# =============================================================================

class NEATNetwork:
    """
    Enhanced neural network with vectorized batch evaluation.
    """
    
    def __init__(self, genome: Genome):
        self.genome = genome
        self._build_computation_order()
    
    def _build_computation_order(self):
        """Build topologically sorted computation order"""
        self.input_nodes = sorted([n.id for n in self.genome.nodes.values() if n.type == 'input'])
        self.output_nodes = sorted([n.id for n in self.genome.nodes.values() if n.type == 'output'])
        self.hidden_nodes = [n.id for n in self.genome.nodes.values() if n.type == 'hidden']
        
        # Build adjacency for topological sort
        self.incoming = defaultdict(list)
        for conn in self.genome.connections.values():
            if conn.enabled:
                self.incoming[conn.out_node].append((conn.in_node, conn.weight))
        
        # Topological sort of computation order
        self.computation_order = self._topological_sort()
    
    def _topological_sort(self) -> List[int]:
        """Sort nodes in dependency order"""
        visited = set(self.input_nodes)
        order = []
        
        def visit(node_id):
            if node_id in visited:
                return
            for in_node, _ in self.incoming[node_id]:
                visit(in_node)
            visited.add(node_id)
            order.append(node_id)
        
        for node_id in self.hidden_nodes + self.output_nodes:
            visit(node_id)
        
        return order
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Single sample forward pass"""
        node_values = {}
        
        # Set inputs
        for i, node_id in enumerate(self.input_nodes):
            if i < len(inputs):
                node_values[node_id] = inputs[i]
            else:
                node_values[node_id] = 0.0
        
        # Compute in order
        for node_id in self.computation_order:
            if node_id in node_values:
                continue
            
            node = self.genome.nodes[node_id]
            total = node.bias
            
            for in_node, weight in self.incoming[node_id]:
                if in_node in node_values:
                    total += node_values[in_node] * weight
            
            activation_fn = ACTIVATIONS.get(node.activation, ACTIVATIONS['relu'])
            node_values[node_id] = activation_fn(np.array([total]))[0]
        
        return np.array([node_values.get(n, 0.0) for n in self.output_nodes])
    
    def forward_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch forward pass (10x+ faster).
        """
        batch_size = X.shape[0]
        n_nodes = max(self.genome.nodes.keys()) + 1
        
        # Initialize node values matrix
        node_values = np.zeros((batch_size, n_nodes))
        
        # Set inputs
        for i, node_id in enumerate(self.input_nodes):
            if i < X.shape[1]:
                node_values[:, node_id] = X[:, i]
        
        # Compute in sorted order
        for node_id in self.computation_order:
            node = self.genome.nodes[node_id]
            total = np.full(batch_size, node.bias)
            
            for in_node, weight in self.incoming[node_id]:
                total += node_values[:, in_node] * weight
            
            activation_fn = ACTIVATIONS.get(node.activation, ACTIVATIONS['relu'])
            node_values[:, node_id] = activation_fn(total)
        
        # Extract outputs
        outputs = np.column_stack([node_values[:, n] for n in self.output_nodes])
        return outputs


# =============================================================================
# Enhanced NEAT Population
# =============================================================================

class NEATPopulation:
    """
    Enhanced NEAT population with stagnation detection.
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 population_size: int = 150,
                 c1: float = 1.0,
                 c2: float = 1.0,
                 c3: float = 0.4,
                 compatibility_threshold: float = 3.0,
                 mutation_rate: float = 0.8,
                 weight_mutation_rate: float = 0.8,
                 add_node_rate: float = 0.03,
                 add_connection_rate: float = 0.05,
                 stagnation_limit: int = 15,
                 min_species_age: int = 5):
        
        self.input_size = input_size
        self.output_size = output_size
        self.population_size = population_size
        
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.compatibility_threshold = compatibility_threshold
        
        self.mutation_rate = mutation_rate
        self.weight_mutation_rate = weight_mutation_rate
        self.add_node_rate = add_node_rate
        self.add_connection_rate = add_connection_rate
        self.stagnation_limit = stagnation_limit
        self.min_species_age = min_species_age
        
        # Advanced activations for hidden nodes
        self.hidden_activations = ['relu', 'gelu', 'swish', 'mish', 'tanh']
        
        self.innovation = InnovationTracker()
        self.population: List[Genome] = []
        self.species: Dict[int, Species] = {}
        self.next_species_id = 0
        self.next_genome_id = 0
        self.generation = 0
        self.best_fitness = 0
        self.best_genome = None
        
        self._initialize_population()
    
    def _initialize_population(self):
        """Create initial minimal population"""
        for _ in range(self.input_size):
            self.innovation.get_new_node_id()
        for _ in range(self.output_size):
            self.innovation.get_new_node_id()
        
        for _ in range(self.population_size):
            genome = self._create_minimal_genome()
            self.population.append(genome)
    
    def _create_minimal_genome(self) -> Genome:
        """Create genome with minimal topology"""
        genome_id = self.next_genome_id
        self.next_genome_id += 1
        
        nodes = {}
        connections = {}
        
        for i in range(self.input_size):
            nodes[i] = NodeGene(id=i, type='input')
        
        for i in range(self.output_size):
            node_id = self.input_size + i
            nodes[node_id] = NodeGene(id=node_id, type='output', activation='sigmoid')
        
        for in_node in range(self.input_size):
            for out_node in range(self.input_size, self.input_size + self.output_size):
                if random.random() < 0.5:
                    innovation = self.innovation.get_innovation(in_node, out_node)
                    connections[innovation] = ConnectionGene(
                        innovation=innovation,
                        in_node=in_node,
                        out_node=out_node,
                        weight=random.gauss(0, 1)
                    )
        
        return Genome(id=genome_id, nodes=nodes, connections=connections)
    
    def _mutate_weights(self, genome: Genome):
        """Mutate connection weights"""
        for conn in genome.connections.values():
            if random.random() < 0.9:
                conn.weight += random.gauss(0, 0.5)
                conn.weight = np.clip(conn.weight, -5, 5)
            else:
                conn.weight = random.gauss(0, 1)
    
    def _mutate_activation(self, genome: Genome):
        """Mutate hidden node activations"""
        hidden = [n for n in genome.nodes.values() if n.type == 'hidden']
        if hidden:
            node = random.choice(hidden)
            node.activation = random.choice(self.hidden_activations)
    
    def _mutate_add_node(self, genome: Genome):
        """Add a new node by splitting a connection"""
        enabled = [c for c in genome.connections.values() if c.enabled]
        if not enabled:
            return
        
        conn = random.choice(enabled)
        conn.enabled = False
        
        new_node_id = self.innovation.get_new_node_id()
        activation = random.choice(self.hidden_activations)
        genome.nodes[new_node_id] = NodeGene(id=new_node_id, type='hidden', activation=activation)
        
        innov1 = self.innovation.get_innovation(conn.in_node, new_node_id)
        innov2 = self.innovation.get_innovation(new_node_id, conn.out_node)
        
        genome.connections[innov1] = ConnectionGene(
            innovation=innov1, in_node=conn.in_node, out_node=new_node_id, weight=1.0
        )
        genome.connections[innov2] = ConnectionGene(
            innovation=innov2, in_node=new_node_id, out_node=conn.out_node, weight=conn.weight
        )
    
    def _mutate_add_connection(self, genome: Genome):
        """Add a new connection"""
        node_ids = list(genome.nodes.keys())
        
        for _ in range(20):
            in_node = random.choice(node_ids)
            out_node = random.choice(node_ids)
            
            if in_node == out_node:
                continue
            if genome.nodes[in_node].type == 'output':
                continue
            if genome.nodes[out_node].type == 'input':
                continue
            
            innov = self.innovation.get_innovation(in_node, out_node)
            if innov in genome.connections:
                continue
            
            genome.connections[innov] = ConnectionGene(
                innovation=innov,
                in_node=in_node,
                out_node=out_node,
                weight=random.gauss(0, 1)
            )
            break
    
    def _multi_point_crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Enhanced multi-point crossover"""
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1
        
        child_id = self.next_genome_id
        self.next_genome_id += 1
        
        child_nodes = {}
        child_connections = {}
        
        # Nodes from fitter parent
        for node_id, node in parent1.nodes.items():
            child_nodes[node_id] = NodeGene(
                id=node.id, type=node.type, activation=node.activation, bias=node.bias
            )
        
        # Multi-point crossover for connections
        all_innovations = sorted(set(parent1.connections.keys()) | set(parent2.connections.keys()))
        crossover_points = sorted(random.sample(range(len(all_innovations)), min(3, len(all_innovations))))
        use_parent1 = True
        
        for i, innov in enumerate(all_innovations):
            if i in crossover_points:
                use_parent1 = not use_parent1
            
            if innov in parent1.connections and innov in parent2.connections:
                if use_parent1:
                    inherited = parent1.connections[innov]
                else:
                    inherited = parent2.connections[innov]
            elif innov in parent1.connections:
                inherited = parent1.connections[innov]
            else:
                continue  # Skip excess/disjoint from less fit
            
            child_connections[innov] = ConnectionGene(
                innovation=inherited.innovation,
                in_node=inherited.in_node,
                out_node=inherited.out_node,
                weight=inherited.weight,
                enabled=inherited.enabled or (random.random() < 0.25)  # 25% re-enable
            )
        
        return Genome(id=child_id, nodes=child_nodes, connections=child_connections)
    
    def _compute_compatibility(self, genome1: Genome, genome2: Genome) -> float:
        """Compute compatibility distance"""
        genes1 = set(genome1.connections.keys())
        genes2 = set(genome2.connections.keys())
        
        matching = genes1 & genes2
        disjoint = len(genes1 ^ genes2)
        
        N = max(len(genes1), len(genes2), 1)
        
        weight_diff = 0
        if matching:
            weight_diff = np.mean([
                abs(genome1.connections[i].weight - genome2.connections[i].weight)
                for i in matching
            ])
        
        return (self.c1 * disjoint / N) + (self.c3 * weight_diff)
    
    def _speciate(self):
        """Divide population into species with stagnation tracking"""
        # Update species
        for species in self.species.values():
            species.members = []
        
        for genome in self.population:
            placed = False
            
            for species_id, species in self.species.items():
                if species.representative:
                    dist = self._compute_compatibility(genome, species.representative)
                    if dist < self.compatibility_threshold:
                        species.members.append(genome)
                        genome.species_id = species_id
                        placed = True
                        break
            
            if not placed:
                new_species = Species(id=self.next_species_id)
                new_species.members.append(genome)
                new_species.representative = genome
                genome.species_id = self.next_species_id
                self.species[self.next_species_id] = new_species
                self.next_species_id += 1
        
        # Update representatives and stagnation
        to_remove = []
        for species_id, species in self.species.items():
            if not species.members:
                to_remove.append(species_id)
                continue
            
            species.representative = random.choice(species.members)
            best = max(species.members, key=lambda g: g.fitness)
            
            if best.fitness > species.best_fitness:
                species.best_fitness = best.fitness
                species.stagnant_generations = 0
            else:
                species.stagnant_generations += 1
            
            # Protected if young
            species.protected = self.generation - species_id < self.min_species_age
        
        for sid in to_remove:
            del self.species[sid]
    
    def evolve(self, fitness_func: Callable[[Genome], float]) -> Genome:
        """Run one generation of evolution"""
        self.generation += 1
        
        # Evaluate fitness
        for genome in self.population:
            genome.fitness = fitness_func(genome)
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome
        
        # Speciate
        self._speciate()
        
        # Adjust fitness
        for species in self.species.values():
            for genome in species.members:
                genome.adjusted_fitness = genome.fitness / max(1, len(species.members))
        
        # Remove stagnant species (unless protected)
        for sid, species in list(self.species.items()):
            if species.stagnant_generations > self.stagnation_limit and not species.protected:
                logger.debug(f"Removing stagnant species {sid}")
                del self.species[sid]
        
        # Create next generation
        new_population = []
        
        # Elitism
        for species in self.species.values():
            if species.members:
                best = max(species.members, key=lambda g: g.fitness)
                new_population.append(best)
        
        # Fill with offspring
        sorted_genomes = sorted(self.population, key=lambda g: g.adjusted_fitness, reverse=True)
        
        while len(new_population) < self.population_size:
            parent1 = random.choice(sorted_genomes[:len(sorted_genomes)//2])
            parent2 = random.choice(sorted_genomes[:len(sorted_genomes)//2])
            
            child = self._multi_point_crossover(parent1, parent2)
            
            if random.random() < self.mutation_rate:
                self._mutate_weights(child)
            if random.random() < 0.1:  # Activation mutation
                self._mutate_activation(child)
            if random.random() < self.add_node_rate:
                self._mutate_add_node(child)
            if random.random() < self.add_connection_rate:
                self._mutate_add_connection(child)
            
            new_population.append(child)
        
        self.population = new_population
        return self.best_genome


# =============================================================================
# Enhanced NEAT Football Predictor
# =============================================================================

class NEATFootball:
    """
    Enhanced NEAT-based football predictor with vectorized evaluation.
    """
    
    def __init__(self,
                 population_size: int = 150,
                 generations: int = 100,
                 target_accuracy: float = 0.60):
        
        self.population_size = population_size
        self.generations = generations
        self.target_accuracy = target_accuracy
        
        self.population: Optional[NEATPopulation] = None
        self.best_network: Optional[NEATNetwork] = None
        self.history = []
    
    def _create_fitness_function(self, X: np.ndarray, y: np.ndarray):
        """Create vectorized fitness function"""
        def fitness(genome: Genome) -> float:
            try:
                network = NEATNetwork(genome)
                outputs = network.forward_batch(X)
                preds = np.argmax(outputs, axis=1)
                return (preds == y).mean()
            except Exception:
                return 0.0
        return fitness
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Train NEAT on football data"""
        input_size = X.shape[1]
        output_size = 3
        
        self.population = NEATPopulation(
            input_size=input_size,
            output_size=output_size,
            population_size=self.population_size
        )
        
        fitness_func = self._create_fitness_function(X, y)
        
        for gen in range(self.generations):
            best_genome = self.population.evolve(fitness_func)
            self.history.append(best_genome.fitness)
            
            if verbose and (gen + 1) % 10 == 0:
                n_species = len(self.population.species)
                logger.info(f"  Gen {gen+1}: Best={best_genome.fitness:.4f}, Species={n_species}")
            
            if best_genome.fitness >= self.target_accuracy:
                logger.info(f"  Reached target at generation {gen+1}")
                break
        
        self.best_network = NEATNetwork(self.population.best_genome)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized prediction"""
        if self.best_network is None:
            raise ValueError("Model not trained")
        
        outputs = self.best_network.forward_batch(X)
        return np.argmax(outputs, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probabilities"""
        if self.best_network is None:
            raise ValueError("Model not trained")
        
        outputs = self.best_network.forward_batch(X)
        exp_out = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
        return exp_out / exp_out.sum(axis=1, keepdims=True)
    
    def save(self, path: str):
        """Save trained model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'best_genome': self.population.best_genome,
                'history': self.history,
                'input_size': self.population.input_size,
                'output_size': self.population.output_size
            }, f)
        logger.info(f"Saved NEATFootball to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'NEATFootball':
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.history = data['history']
        model.best_network = NEATNetwork(data['best_genome'])
        
        return model


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Testing Enhanced NEAT Football Predictor...")
    
    np.random.seed(42)
    random.seed(42)
    
    X = np.random.randn(500, 20)
    y = np.random.randint(0, 3, 500)
    
    neat = NEATFootball(population_size=50, generations=20)
    neat.fit(X[:400], y[:400], verbose=True)
    
    preds = neat.predict(X[400:])
    acc = (preds == y[400:]).mean()
    print(f"\n✅ Enhanced NEAT Accuracy: {acc:.2%}")
    
    # Test save/load
    neat.save('/tmp/neat_test.pkl')
    loaded = NEATFootball.load('/tmp/neat_test.pkl')
    preds2 = loaded.predict(X[400:])
    assert np.array_equal(preds, preds2), "Save/load mismatch!"
    print("✅ Save/Load verified")
