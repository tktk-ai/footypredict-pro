# Part 2: Quantum Neural Network Components

# ================================================================================
# SECTION 3: ENHANCED QUANTUM NEURAL NETWORK (CORE)
# ================================================================================

class EnhancedQuantumCircuit:
    """Enhanced Quantum Circuit with Multiple Ansätze"""
    
    def __init__(self, n_qubits: int = 12, n_layers: int = 6, ansatz: str = 'strongly_entangling'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.ansatz = ansatz
        
        try:
            self.dev = qml.device("lightning.qubit", wires=n_qubits)
        except:
            self.dev = qml.device("default.qubit", wires=n_qubits)
        
        self.circuit = qml.QNode(self._build_circuit, self.dev, interface="torch")
    
    def _build_circuit(self, inputs, weights):
        """Build enhanced quantum circuit with data re-uploading"""
        for i in range(self.n_qubits):
            qml.RY(inputs[i % len(inputs)] * np.pi, wires=i)
        
        weight_idx = 0
        for layer in range(self.n_layers):
            for i in range(self.n_qubits):
                qml.RZ(weights[weight_idx], wires=i)
                weight_idx += 1
                qml.RY(weights[weight_idx], wires=i)
                weight_idx += 1
                qml.RZ(weights[weight_idx], wires=i)
                weight_idx += 1
            
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            if layer % 2 == 0 and layer < self.n_layers - 1:
                for i in range(self.n_qubits):
                    qml.RY(inputs[i % len(inputs)] * np.pi * 0.5, wires=i)
            
            if layer % 2 == 1:
                for i in range(0, self.n_qubits - 2, 2):
                    qml.CZ(wires=[i, i + 2])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(min(3, self.n_qubits))]
    
    @property
    def n_params(self):
        return self.n_layers * self.n_qubits * 3


class QuantumProcessingUnit(nn.Module):
    """Core Quantum Processing Unit"""
    
    def __init__(self, input_dim: int, n_qubits: int = 12, n_layers: int = 6, output_dim: int = 64):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits),
            nn.Tanh()
        )
        
        self.qc = EnhancedQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        self.q_weights = nn.Parameter(torch.randn(self.qc.n_params) * 0.1)
        
        self.post_net = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_quantum = self.pre_net(x)
        
        q_outputs = []
        for i in range(batch_size):
            q_out = self.qc.circuit(x_quantum[i] * np.pi, self.q_weights)
            q_outputs.append(torch.stack(q_out))
        
        q_outputs = torch.stack(q_outputs)
        return self.post_net(q_outputs)


class HybridQuantumTransformer(nn.Module):
    """Hybrid Quantum-Classical Transformer"""
    
    def __init__(self, input_dim: int, d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                 n_qubits: int = 12, n_quantum_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.quantum_processor = QuantumProcessingUnit(
            input_dim=d_model, n_qubits=n_qubits, n_layers=n_quantum_layers, output_dim=d_model // 2
        )
        
        self.classical_path = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        x = self.input_embed(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        q_out = self.quantum_processor(x)
        c_out = self.classical_path(x)
        
        combined = torch.cat([q_out, c_out], dim=-1)
        return self.fusion(combined)


# ================================================================================
# SECTION 4: ADVANCED NEURAL ARCHITECTURES
# ================================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim * 2, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return x + self.block(x)


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, n_experts * 2), nn.GELU(),
            nn.Linear(n_experts * 2, n_experts)
        )
        
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, output_dim * 2), nn.GELU(), nn.Linear(output_dim * 2, output_dim))
            for _ in range(n_experts)
        ])
        
    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        batch_size = x.shape[0]
        output = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)
        
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x[mask])
                weight = top_k_weights[mask, (top_k_indices[mask] == i).float().argmax(dim=-1)]
                output[mask] += expert_out * weight.unsqueeze(-1)
        return output


class DeepCrossNetwork(nn.Module):
    def __init__(self, input_dim: int, n_cross_layers: int = 3):
        super().__init__()
        self.n_cross_layers = n_cross_layers
        self.cross_weights = nn.ParameterList([nn.Parameter(torch.randn(input_dim) * 0.01) for _ in range(n_cross_layers)])
        self.cross_biases = nn.ParameterList([nn.Parameter(torch.zeros(input_dim)) for _ in range(n_cross_layers)])
        
    def forward(self, x0):
        x = x0
        for w, b in zip(self.cross_weights, self.cross_biases):
            cross = x0 * (x * w).sum(dim=-1, keepdim=True) + b
            x = cross + x
        return x


class MCDropoutNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], n_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)
        self.dropout = dropout
        
    def forward(self, x, n_samples: int = 1):
        if n_samples == 1: return self.network(x)
        self.train()
        outputs = [self.network(x) for _ in range(n_samples)]
        self.eval()
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(dim=0), outputs.std(dim=0).mean(dim=-1)


class DeepEnsemble(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], n_classes: int, n_networks: int = 5, dropout: float = 0.3):
        super().__init__()
        self.networks = nn.ModuleList([MCDropoutNetwork(input_dim, hidden_dims, n_classes, dropout) for _ in range(n_networks)])
        
    def forward(self, x, mc_samples: int = 1):
        predictions = []
        for net in self.networks:
            pred = net(x, n_samples=mc_samples)
            if isinstance(pred, tuple): pred = pred[0]
            predictions.append(F.softmax(pred, dim=-1))
        ensemble_pred = torch.stack(predictions, dim=0).mean(dim=0)
        uncertainty = torch.stack(predictions, dim=0).std(dim=0).mean(dim=-1)
        return ensemble_pred, uncertainty
