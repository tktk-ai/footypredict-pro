"""
Real Injuries API Client (API-Football)

Fetches real player injury data from API-Football.
Free tier: 100 requests/day

Features:
- Current injuries by team
- Injury severity and return dates
- Cached to minimize API calls
"""

import os
import requests
from datetime import datetime
from typing import Dict, List, Optional
from src.data.api_clients import CacheManager

# API Configuration
API_FOOTBALL_KEY = os.environ.get('API_FOOTBALL_KEY', '')
BASE_URL = 'https://v3.football.api-sports.io'


class RealInjuriesClient:
    """
    Client for API-Football injuries endpoint.
    Replaces simulated injuries in live_data.py
    """
    
    # Team ID mapping (API-Football IDs)
    TEAM_IDS = {
        'Liverpool': 40,
        'Manchester City': 50,
        'Arsenal': 42,
        'Chelsea': 49,
        'Manchester United': 33,
        'Tottenham': 47,
        'Newcastle': 34,
        'Brighton': 51,
        'Aston Villa': 66,
        'West Ham': 48,
        'Bayern': 157,
        'Dortmund': 165,
        'Real Madrid': 541,
        'Barcelona': 529,
        'PSG': 85,
        'Inter': 505,
        'Juventus': 496,
    }
    
    def __init__(self):
        self.api_key = API_FOOTBALL_KEY
        self.cache = CacheManager()
        self.session = requests.Session()
        self.session.headers.update({
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': 'v3.football.api-sports.io'
        })
    
    def has_api_key(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def get_team_injuries(self, team: str) -> List[Dict]:
        """
        Get current injuries for a team.
        
        Returns:
            List of injured players with details
        """
        if not self.api_key:
            return self._get_simulated_injuries(team)
        
        team_id = self._get_team_id(team)
        if not team_id:
            return self._get_simulated_injuries(team)
        
        cache_key = f"injuries_{team_id}"
        cached = self.cache.get(cache_key, max_age_minutes=360)  # 6hr cache
        if cached:
            return cached
        
        url = f"{BASE_URL}/injuries"
        params = {
            'team': team_id,
            'season': 2024
        }
        
        try:
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                injuries = self._parse_injuries(data.get('response', []))
                self.cache.set(cache_key, injuries)
                return injuries
                
        except Exception as e:
            print(f"Injuries API error: {e}")
        
        return self._get_simulated_injuries(team)
    
    def _get_team_id(self, team: str) -> Optional[int]:
        """Get API-Football team ID"""
        if team in self.TEAM_IDS:
            return self.TEAM_IDS[team]
        
        team_lower = team.lower()
        for name, team_id in self.TEAM_IDS.items():
            if name.lower() in team_lower or team_lower in name.lower():
                return team_id
        
        return None
    
    def _parse_injuries(self, data: List) -> List[Dict]:
        """Parse API response to injury list"""
        injuries = []
        
        for item in data:
            player = item.get('player', {})
            injury = item.get('fixture', {}).get('injury', {})
            
            injuries.append({
                'player': player.get('name', 'Unknown'),
                'position': player.get('type', 'Unknown'),
                'injury_type': injury.get('type', 'Unknown'),
                'reason': injury.get('reason', 'Unknown'),
                'severity': self._estimate_severity(injury.get('type', '')),
                'expected_return': 'Unknown',
                'data_source': 'LIVE_API'
            })
        
        return injuries
    
    def _estimate_severity(self, injury_type: str) -> str:
        """Estimate injury severity from type"""
        severe = ['ACL', 'Broken', 'Surgery', 'Ligament']
        moderate = ['Muscle', 'Hamstring', 'Strain', 'Sprain']
        
        injury_lower = injury_type.lower()
        
        for s in severe:
            if s.lower() in injury_lower:
                return 'High'
        
        for m in moderate:
            if m.lower() in injury_lower:
                return 'Medium'
        
        return 'Low'
    
    def _get_simulated_injuries(self, team: str) -> List[Dict]:
        """Fallback simulated injuries when API unavailable"""
        # Minimal simulated data
        simulated = {
            'Liverpool': [
                {'player': 'Unknown Midfielder', 'injury_type': 'Minor Knock', 'severity': 'Low'}
            ],
            'Manchester City': [
                {'player': 'Unknown Defender', 'injury_type': 'Training Issue', 'severity': 'Low'}
            ],
        }
        
        for name, injuries in simulated.items():
            if team.lower() in name.lower():
                return [dict(i, data_source='SIMULATED') for i in injuries]
        
        return []
    
    def get_match_injuries(self, home_team: str, away_team: str) -> Dict:
        """Get injuries for both teams in a match"""
        return {
            'home_team': home_team,
            'home_injuries': self.get_team_injuries(home_team),
            'away_team': away_team,
            'away_injuries': self.get_team_injuries(away_team),
            'data_source': 'LIVE_API' if self.has_api_key() else 'SIMULATED'
        }
    
    def count_key_injuries(self, team: str) -> int:
        """Count high-impact injuries (attackers, key players)"""
        injuries = self.get_team_injuries(team)
        key_positions = ['Forward', 'Attacker', 'Striker', 'Midfielder']
        
        count = 0
        for injury in injuries:
            pos = injury.get('position', '').lower()
            for key_pos in key_positions:
                if key_pos.lower() in pos:
                    count += 1
                    break
        
        return count


# Global instance
injuries_client = RealInjuriesClient()


def get_injuries(team: str) -> List[Dict]:
    """Convenience function for getting injuries"""
    return injuries_client.get_team_injuries(team)


def get_match_injury_impact(home: str, away: str) -> Dict:
    """Get injury impact analysis for a match"""
    home_injuries = injuries_client.count_key_injuries(home)
    away_injuries = injuries_client.count_key_injuries(away)
    
    return {
        'home_key_injuries': home_injuries,
        'away_key_injuries': away_injuries,
        'injury_advantage': 'home' if away_injuries > home_injuries else (
            'away' if home_injuries > away_injuries else 'neutral'
        ),
        'impact_score': abs(home_injuries - away_injuries) * 0.05
    }
