"""
Reinforcement Learning for Optimal Betting Strategy
Uses Deep Q-Network (DQN) for learning optimal bet sizing

Based on the blueprint's RL betting system.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "rl"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


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
        self.state_dim = 10  # State features
        self.action_dim = bet_levels
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.bankroll = self.initial_bankroll
        self.bet_history = []
        return self._get_state()
    
    def _get_state(self, bet_info: Dict = None) -> np.ndarray:
        """Get current state representation."""
        if bet_info is None:
            return np.zeros(self.state_dim, dtype=np.float32)
        
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
                profit = stake * (bet_info['odds'] - 1)
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
        
        # Calculate reward (log returns for stability)
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
        
        return next_state, reward, done, info
    
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


class DQNNetwork(nn.Module):
    """Deep Q-Network for betting decisions."""
    
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
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNBettingAgent:
    """
    Complete DQN agent for betting strategy learning.
    Uses Double DQN for more stable training.
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
        self.training_losses = []
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def get_stake_recommendation(self, bet_info: Dict) -> Dict:
        """Get betting recommendation from trained agent."""
        state = np.array([
            1.0,  # Normalized bankroll (assume full)
            bet_info.get('probability', 0.5),
            bet_info.get('odds', 2.0),
            bet_info.get('edge', 0.0),
            bet_info.get('confidence', 0.5),
            0.5,  # Recent win rate
            0.0,  # Recent ROI
            bet_info.get('market_efficiency', 0.5),
            1.0,  # Time to event
            0.5   # Bet count ratio
        ], dtype=np.float32)
        
        action = self.select_action(state, evaluate=True)
        
        stake_pct = action / self.action_dim * 10  # 0-10% of bankroll
        
        return {
            'action': action,
            'stake_percentage': stake_pct,
            'recommendation': 'bet' if action > 0 else 'skip',
            'confidence': 'high' if action >= 7 else 'medium' if action >= 4 else 'low'
        }
    
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
        
        self.training_losses.append(loss.item())
        
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
    
    def save(self, path: str = None):
        """Save model weights."""
        if path is None:
            path = MODELS_DIR / "dqn_betting_agent.pt"
        
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        logger.info(f"Saved RL agent to {path}")
    
    def load(self, path: str = None):
        """Load model weights."""
        if path is None:
            path = MODELS_DIR / "dqn_betting_agent.pt"
        
        if not Path(path).exists():
            logger.warning(f"No saved agent at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        logger.info(f"Loaded RL agent from {path}")
        return True


class RLBettingTrainer:
    """Trainer for RL betting agent using historical data."""
    
    def __init__(
        self,
        agent: DQNBettingAgent = None,
        initial_bankroll: float = 1000.0
    ):
        self.agent = agent or DQNBettingAgent()
        self.env = BettingEnvironment(initial_bankroll=initial_bankroll)
        
    def train(
        self,
        historical_bets: List[Dict],
        n_episodes: int = 100
    ) -> Dict:
        """
        Train agent on historical betting data.
        
        Args:
            historical_bets: List of historical bet opportunities with outcomes
                Each dict should have: odds, probability, outcome (True/False)
            n_episodes: Number of training episodes
        """
        episode_rewards = []
        episode_rois = []
        
        logger.info(f"Training RL agent on {len(historical_bets)} historical bets for {n_episodes} episodes")
        
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            
            # Shuffle bets for each episode
            shuffled_bets = random.sample(historical_bets, min(len(historical_bets), 200))
            
            for bet in shuffled_bets:
                # Prepare bet info
                bet_info = {
                    'odds': bet.get('odds', 2.0),
                    'model_probability': bet.get('probability', 0.5),
                    'edge': bet.get('probability', 0.5) - 1/bet.get('odds', 2.0),
                    'confidence': bet.get('confidence', 0.5),
                    'market_efficiency': 0.5,
                    'time_to_event': 1.0
                }
                
                # Get current state with bet info
                state = self.env._get_state(bet_info)
                
                # Select action
                action = self.agent.select_action(state)
                
                # Execute action and get outcome
                next_state, reward, done, info = self.env.step(
                    action,
                    bet_info,
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
                avg_reward = np.mean(episode_rewards[-10:])
                avg_roi = np.mean(episode_rois[-10:])
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.3f}, Avg ROI={avg_roi:.2%}, Epsilon={self.agent.epsilon:.3f}")
        
        # Save trained agent
        self.agent.save()
        
        return {
            'episode_rewards': episode_rewards,
            'episode_rois': episode_rois,
            'final_epsilon': self.agent.epsilon,
            'total_steps': self.agent.steps
        }
    
    def train_on_predictions(
        self,
        predictions_file: str = "data/predictions/prediction_history.json",
        n_episodes: int = 50
    ) -> Dict:
        """Train on historical prediction outcomes."""
        try:
            with open(predictions_file) as f:
                history = json.load(f)
            
            # Convert to training format
            bets = []
            for pred in history:
                if pred.get('actual_result') is not None:
                    bets.append({
                        'odds': pred.get('odds', 2.0),
                        'probability': pred.get('confidence', 0.5),
                        'outcome': pred.get('correct', False),
                        'confidence': pred.get('confidence', 0.5)
                    })
            
            if len(bets) < 50:
                logger.warning(f"Only {len(bets)} historical bets, need more data")
                return {'error': 'Insufficient training data'}
            
            return self.train(bets, n_episodes)
            
        except FileNotFoundError:
            logger.warning(f"No prediction history at {predictions_file}")
            return {'error': 'No prediction history found'}


# Global instances
_agent: Optional[DQNBettingAgent] = None
_trainer: Optional[RLBettingTrainer] = None


def get_rl_agent() -> DQNBettingAgent:
    """Get or create RL betting agent."""
    global _agent
    if _agent is None:
        _agent = DQNBettingAgent()
        _agent.load()  # Try to load saved weights
    return _agent


def get_rl_trainer() -> RLBettingTrainer:
    """Get or create RL trainer."""
    global _trainer
    if _trainer is None:
        _trainer = RLBettingTrainer(get_rl_agent())
    return _trainer


def get_rl_recommendation(bet_info: Dict) -> Dict:
    """Get betting recommendation from RL agent."""
    return get_rl_agent().get_stake_recommendation(bet_info)
