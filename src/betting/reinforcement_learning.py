"""
Reinforcement Learning for Optimal Betting Strategy V3.0
Uses DQN (Deep Q-Network) with experience replay

Features:
- DQN agent for bet sizing decisions
- Betting environment simulation
- Experience replay buffer
- Double DQN with target network
- Epsilon-greedy exploration
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. RL models will not be available.")


class BettingEnvironment:
    """
    RL Environment for sports betting simulation.
    
    State: [bankroll_ratio, recent_win_rate, current_odds, model_confidence, market_features...]
    Action: [bet_size_level (0 = no bet, 1-10 = increasing stake percentages)]
    Reward: Profit/Loss from bet
    """
    
    def __init__(
        self,
        initial_bankroll: float = 1000.0,
        max_bet_fraction: float = 0.10,
        bet_levels: int = 11  # 0 = no bet, 1-10 = 1%-10% of bankroll
    ):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.max_bet_fraction = max_bet_fraction
        self.bet_levels = bet_levels
        
        self.bet_history = []
        self.state_dim = 10  # Adjust based on actual state features
        self.action_dim = bet_levels
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.bankroll = self.initial_bankroll
        self.bet_history = []
        return self._get_state()
    
    def _get_state(self, bet_info: Dict = None) -> np.ndarray:
        """Get current state representation."""
        if bet_info is None:
            return np.zeros(self.state_dim)
        
        state = np.array([
            self.bankroll / self.initial_bankroll,  # Normalized bankroll
            bet_info.get('model_probability', 0.5),
            bet_info.get('odds', 2.0),
            bet_info.get('edge', 0.0),
            bet_info.get('confidence', 0.5),
            self._get_recent_win_rate(),
            self._get_recent_roi(),
            bet_info.get('market_efficiency', 0.5),
            bet_info.get('time_to_event', 1.0),
            min(len(self.bet_history) / 100, 1.0)  # Normalized bet count
        ], dtype=np.float32)
        
        return state
    
    def step(
        self,
        action: int,
        bet_info: Dict,
        outcome: bool
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute betting action.
        
        Args:
            action: Bet size level (0-10)
            bet_info: Information about the bet (odds, probability, etc.)
            outcome: Whether the bet won
            
        Returns:
            next_state, reward, done, info
        """
        # Calculate stake
        if action == 0:
            stake = 0
            profit = 0
        else:
            stake_fraction = action / self.bet_levels * self.max_bet_fraction
            stake = self.bankroll * stake_fraction
            
            if outcome:
                profit = stake * (bet_info.get('odds', 2.0) - 1)
            else:
                profit = -stake
        
        # Update bankroll
        self.bankroll += profit
        
        # Record bet
        self.bet_history.append({
            'stake': stake,
            'odds': bet_info.get('odds', 0),
            'profit': profit,
            'won': outcome,
            'action': action
        })
        
        # Calculate reward (using log returns for stability)
        if profit > 0:
            reward = np.log(1 + profit / max(stake, 1))
        elif profit < 0:
            reward = np.log(1 + profit / self.initial_bankroll) * 2  # Penalize losses more
        else:
            reward = 0
        
        # Check if done (bankrupt)
        done = self.bankroll < self.initial_bankroll * 0.1
        
        next_state = self._get_state(bet_info)
        
        info = {
            'bankroll': self.bankroll,
            'profit': profit,
            'total_roi': (self.bankroll - self.initial_bankroll) / self.initial_bankroll
        }
        
        return next_state, float(reward), done, info
    
    def _get_recent_win_rate(self, n: int = 20) -> float:
        """Get win rate of recent bets."""
        if not self.bet_history:
            return 0.5
        recent = self.bet_history[-n:]
        return sum(1 for b in recent if b['won']) / len(recent)
    
    def _get_recent_roi(self, n: int = 20) -> float:
        """Get ROI of recent bets."""
        if not self.bet_history:
            return 0.0
        recent = self.bet_history[-n:]
        total_staked = sum(b['stake'] for b in recent)
        total_profit = sum(b['profit'] for b in recent)
        if total_staked > 0:
            return total_profit / total_staked
        return 0.0
    
    def get_stats(self) -> Dict:
        """Get betting statistics."""
        if not self.bet_history:
            return {'total_bets': 0}
        
        wins = sum(1 for b in self.bet_history if b['won'])
        total_staked = sum(b['stake'] for b in self.bet_history)
        total_profit = sum(b['profit'] for b in self.bet_history)
        
        return {
            'total_bets': len(self.bet_history),
            'wins': wins,
            'losses': len(self.bet_history) - wins,
            'win_rate': wins / len(self.bet_history),
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': total_profit / total_staked if total_staked > 0 else 0,
            'final_bankroll': self.bankroll
        }


class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


if TORCH_AVAILABLE:
    
    class DQNNetwork(nn.Module):
        """
        Deep Q-Network for betting decisions.
        """
        
        def __init__(
            self,
            state_dim: int,
            action_dim: int,
            hidden_dims: List[int] = [256, 256, 128]
        ):
            super().__init__()
            
            layers = []
            input_dim = state_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, action_dim))
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, state: torch.Tensor) -> torch.Tensor:
            """Forward pass returns Q-values for all actions."""
            return self.network(state)


    class DQNBettingAgent:
        """
        Complete DQN agent for betting strategy learning.
        
        Uses Double DQN with target network for stable learning.
        """
        
        def __init__(
            self,
            state_dim: int = 10,
            action_dim: int = 11,
            learning_rate: float = 1e-4,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.995,
            buffer_size: int = 100000,
            batch_size: int = 64,
            target_update: int = 100
        ):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.batch_size = batch_size
            self.target_update = target_update
            
            # Networks
            self.policy_net = DQNNetwork(state_dim, action_dim).to(self.device)
            self.target_net = DQNNetwork(state_dim, action_dim).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
            
            # Exploration
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            
            # Replay buffer
            self.buffer = ReplayBuffer(buffer_size)
            
            self.steps = 0
            
        def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
            """Select action using epsilon-greedy policy."""
            if not evaluate and random.random() < self.epsilon:
                return random.randrange(self.action_dim)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
        
        def train_step(self) -> Optional[float]:
            """Perform one training step."""
            if len(self.buffer) < self.batch_size:
                return None
            
            # Sample batch
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            
            # Compute current Q values
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
            
            # Compute target Q values (Double DQN)
            with torch.no_grad():
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
                target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))
            
            # Compute loss
            loss = F.smooth_l1_loss(current_q, target_q)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()
            
            # Update target network
            self.steps += 1
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            return loss.item()
        
        def store_transition(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            next_state: np.ndarray,
            done: bool
        ):
            """Store transition in replay buffer."""
            self.buffer.push(state, action, reward, next_state, done)
        
        def get_optimal_bet_size(
            self,
            model_probability: float,
            odds: float,
            confidence: float = 0.5,
            bankroll_ratio: float = 1.0
        ) -> Dict:
            """
            Get optimal bet size recommendation.
            
            Args:
                model_probability: Predicted probability
                odds: Bookmaker odds
                confidence: Model confidence
                bankroll_ratio: Current bankroll / initial
            """
            # Calculate edge
            implied_prob = 1 / odds
            edge = model_probability - implied_prob
            
            # Create state
            state = np.array([
                bankroll_ratio,
                model_probability,
                odds,
                edge,
                confidence,
                0.5,  # Recent win rate (placeholder)
                0.0,  # Recent ROI (placeholder)
                0.5,  # Market efficiency (placeholder)
                1.0,  # Time to event
                0.0   # Bet count
            ], dtype=np.float32)
            
            # Get action
            action = self.select_action(state, evaluate=True)
            
            # Convert to stake percentage
            stake_pct = action / self.action_dim * 10  # 0-10% of bankroll
            
            return {
                'action': action,
                'stake_percentage': round(stake_pct, 1),
                'edge': round(edge, 4),
                'recommendation': 'bet' if action > 0 else 'skip',
                'confidence': confidence,
                'expected_value': round(edge * odds, 4) if edge > 0 else 0
            }
        
        def save(self, path: str):
            """Save model weights."""
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps': self.steps
            }, path)
        
        def load(self, path: str):
            """Load model weights."""
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.steps = checkpoint['steps']


    class RLBettingTrainer:
        """
        Trainer for RL betting agent using historical data.
        """
        
        def __init__(
            self,
            agent: DQNBettingAgent = None,
            env: BettingEnvironment = None
        ):
            self.env = env or BettingEnvironment()
            self.agent = agent or DQNBettingAgent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim
            )
            
        def train(
            self,
            historical_bets: List[Dict],
            n_episodes: int = 100
        ) -> Dict:
            """
            Train agent on historical betting data.
            
            Args:
                historical_bets: List of historical bet opportunities with outcomes
                n_episodes: Number of training episodes
            """
            episode_rewards = []
            episode_rois = []
            
            for episode in range(n_episodes):
                state = self.env.reset()
                total_reward = 0
                
                # Shuffle bets for each episode
                shuffled_bets = random.sample(historical_bets, len(historical_bets))
                
                for bet in shuffled_bets:
                    # Get current state with bet info
                    state = self.env._get_state(bet)
                    
                    # Select action
                    action = self.agent.select_action(state)
                    
                    # Execute action and get outcome
                    next_state, reward, done, info = self.env.step(
                        action,
                        bet,
                        bet.get('outcome', False)
                    )
                    
                    # Store transition
                    self.agent.store_transition(state, action, reward, next_state, done)
                    
                    # Train
                    self.agent.train_step()
                    
                    total_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                episode_rewards.append(total_reward)
                episode_rois.append(info['total_roi'])
                
                if episode % 10 == 0:
                    logger.info(f"Episode {episode}: Reward={total_reward:.2f}, ROI={info['total_roi']:.2%}")
            
            return {
                'episode_rewards': episode_rewards,
                'episode_rois': episode_rois,
                'final_stats': self.env.get_stats()
            }

else:
    # Dummy classes when PyTorch is not available
    class DQNBettingAgent:
        def __init__(self, *args, **kwargs):
            logger.warning("PyTorch not installed. Using rule-based betting instead.")
            self.state_dim = 10
            self.action_dim = 11
            
        def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
            """Rule-based action selection without RL."""
            edge = state[3] if len(state) > 3 else 0
            confidence = state[4] if len(state) > 4 else 0.5
            
            if edge > 0.1 and confidence > 0.6:
                return 5  # 5% stake
            elif edge > 0.05 and confidence > 0.5:
                return 3  # 3% stake
            elif edge > 0.03:
                return 1  # 1% stake
            return 0  # No bet
            
        def get_optimal_bet_size(self, *args, **kwargs) -> Dict:
            return {'action': 0, 'stake_percentage': 0, 'recommendation': 'skip'}
    
    class RLBettingTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for RL training. Install with: pip install torch")
