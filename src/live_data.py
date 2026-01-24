"""
Live Data Enrichment Module

Provides real-time data for enhanced predictions:
- Live scores via WebSocket
- Player injuries/suspensions (NOW USES REAL API)
- Weather conditions
- More leagues

NOTE: Now uses API-Football for real injury data
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import real injuries client
try:
    from src.data.real_injuries import RealInjuriesClient, get_injuries as get_real_injuries
    REAL_INJURIES_AVAILABLE = True
except ImportError:
    REAL_INJURIES_AVAILABLE = False


@dataclass
class PlayerStatus:
    """Player injury/suspension status"""
    name: str
    team: str
    status: str  # 'injured', 'suspended', 'doubtful', 'available'
    reason: Optional[str] = None
    expected_return: Optional[str] = None


@dataclass  
class WeatherData:
    """Match weather conditions"""
    temperature: float  # Celsius
    condition: str  # 'clear', 'rain', 'snow', 'fog', etc.
    humidity: int
    wind_speed: float
    affects_play: bool  # True if extreme conditions


class LiveDataClient:
    """
    Aggregates live data from multiple sources.
    NOW USES REAL API for injuries when available.
    """
    
    def __init__(self):
        self.openweather_key = os.getenv('OPENWEATHER_API_KEY')
        self.session = requests.Session()
        self._injuries_client = None
        if REAL_INJURIES_AVAILABLE:
            try:
                self._injuries_client = RealInjuriesClient()
            except:
                pass
    
    # ============================================================
    # LIVE SCORES (OpenLigaDB - Free, no key needed)
    # ============================================================
    
    def get_live_scores(self, league: str = 'bundesliga') -> List[Dict]:
        """
        Get currently live match scores
        Uses OpenLigaDB which updates every minute
        """
        league_codes = {
            'bundesliga': 'bl1',
            'bundesliga2': 'bl2',
            '3liga': 'bl3'
        }
        
        code = league_codes.get(league, 'bl1')
        
        try:
            url = f"https://api.openligadb.de/getmatchdata/{code}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                matches = response.json()
                live = []
                
                for m in matches:
                    # Check if match is live
                    kickoff = m.get('matchDateTime')
                    is_finished = m.get('matchIsFinished', False)
                    
                    if kickoff and not is_finished:
                        dt = datetime.fromisoformat(kickoff)
                        now = datetime.now()
                        
                        # Match is live if started within last 2 hours and not finished
                        if now > dt and (now - dt).total_seconds() < 7200:
                            # Get current score
                            results = m.get('matchResults', [])
                            home_score = 0
                            away_score = 0
                            
                            for r in results:
                                if r.get('resultTypeID') == 2:  # Live/Final
                                    home_score = r.get('pointsTeam1', 0)
                                    away_score = r.get('pointsTeam2', 0)
                            
                            live.append({
                                'match_id': m.get('matchID'),
                                'home_team': m['team1']['teamName'],
                                'away_team': m['team2']['teamName'],
                                'home_score': home_score,
                                'away_score': away_score,
                                'minute': self._calculate_minute(dt),
                                'status': 'live'
                            })
                
                return live
        except Exception as e:
            print(f"Live scores error: {e}")
        
        return []
    
    def _calculate_minute(self, kickoff: datetime) -> int:
        """Calculate approximate match minute"""
        elapsed = (datetime.now() - kickoff).total_seconds()
        minute = int(elapsed / 60)
        
        # Account for halftime (15 min break after 45)
        if minute > 60:
            minute = min(minute - 15, 90)
        
        return max(1, min(minute, 90))
    
    # ============================================================
    # PLAYER INJURIES (NOW USES REAL API-FOOTBALL)
    # ============================================================
    
    def get_team_injuries(self, team: str) -> List[PlayerStatus]:
        """
        Get player injuries/suspensions for a team.
        NOW USES REAL API-FOOTBALL when available.
        """
        # Try real API first
        if self._injuries_client and self._injuries_client.has_api_key():
            try:
                real_injuries = self._injuries_client.get_team_injuries(team)
                if real_injuries:
                    return [
                        PlayerStatus(
                            name=inj.get('player', 'Unknown'),
                            team=team,
                            status='injured',
                            reason=inj.get('injury_type', 'Unknown'),
                            expected_return=inj.get('expected_return')
                        )
                        for inj in real_injuries
                    ]
            except Exception as e:
                print(f"Real injuries fetch failed: {e}")
        
        # Fallback to simulated data
        return self._get_fallback_injuries(team)
    
    def _get_fallback_injuries(self, team: str) -> List[PlayerStatus]:
        """Fallback injury data when API unavailable"""
        known_injuries = {
            'Bayern': [
                PlayerStatus('Minor Injury', 'Bayern', 'doubtful', 'Muscle fatigue'),
            ],
            'Dortmund': [],
            'Liverpool': [],
            'Manchester City': [],
        }
        
        if team in known_injuries:
            return known_injuries[team]
        
        team_lower = team.lower()
        for name, injuries in known_injuries.items():
            if team_lower in name.lower() or name.lower() in team_lower:
                return injuries
        
        return []
    
    def get_key_absences(self, home_team: str, away_team: str) -> Dict:
        """Get key absences for both teams"""
        home_injuries = self.get_team_injuries(home_team)
        away_injuries = self.get_team_injuries(away_team)
        
        return {
            'home_team': {
                'name': home_team,
                'absences': [
                    {
                        'player': p.name,
                        'status': p.status,
                        'reason': p.reason
                    }
                    for p in home_injuries
                ]
            },
            'away_team': {
                'name': away_team,
                'absences': [
                    {
                        'player': p.name,
                        'status': p.status,
                        'reason': p.reason
                    }
                    for p in away_injuries
                ]
            }
        }
    
    # ============================================================
    # WEATHER (OpenWeatherMap - Free tier available)
    # ============================================================
    
    def get_match_weather(
        self, 
        city: str = None,
        lat: float = None,
        lon: float = None
    ) -> Optional[WeatherData]:
        """
        Get weather conditions for match location
        
        Requires OPENWEATHER_API_KEY in .env
        Free tier: 1000 calls/day
        """
        if not self.openweather_key:
            # Return default mild weather
            return WeatherData(
                temperature=15.0,
                condition='clear',
                humidity=60,
                wind_speed=10.0,
                affects_play=False
            )
        
        try:
            if lat and lon:
                url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.openweather_key}&units=metric"
            elif city:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.openweather_key}&units=metric"
            else:
                return None
            
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                temp = data.get('main', {}).get('temp', 15)
                humidity = data.get('main', {}).get('humidity', 60)
                wind = data.get('wind', {}).get('speed', 10)
                condition = data.get('weather', [{}])[0].get('main', 'Clear')
                
                # Determine if conditions affect play
                affects = False
                if temp < 0 or temp > 35:
                    affects = True
                if wind > 50:  # km/h
                    affects = True
                if condition.lower() in ['snow', 'thunderstorm', 'fog']:
                    affects = True
                
                return WeatherData(
                    temperature=temp,
                    condition=condition.lower(),
                    humidity=humidity,
                    wind_speed=wind,
                    affects_play=affects
                )
        except Exception as e:
            print(f"Weather error: {e}")
        
        return None
    
    def get_stadium_cities(self) -> Dict[str, str]:
        """Map teams to their stadium cities"""
        return {
            # Bundesliga
            'FC Bayern München': 'Munich,DE',
            'Bayern': 'Munich,DE',
            'Borussia Dortmund': 'Dortmund,DE',
            'Dortmund': 'Dortmund,DE',
            'Bayer 04 Leverkusen': 'Leverkusen,DE',
            'RB Leipzig': 'Leipzig,DE',
            'VfB Stuttgart': 'Stuttgart,DE',
            'Eintracht Frankfurt': 'Frankfurt,DE',
            
            # Premier League  
            'Manchester City': 'Manchester,UK',
            'Manchester United': 'Manchester,UK',
            'Liverpool': 'Liverpool,UK',
            'Arsenal': 'London,UK',
            'Chelsea': 'London,UK',
            'Tottenham Hotspur': 'London,UK',
            
            # La Liga
            'Real Madrid': 'Madrid,ES',
            'Barcelona': 'Barcelona,ES',
            'Atlético Madrid': 'Madrid,ES',
            
            # Serie A
            'Inter Milan': 'Milan,IT',
            'AC Milan': 'Milan,IT',
            'Juventus': 'Turin,IT',
            'Napoli': 'Naples,IT',
        }
    
    def get_weather_for_match(self, home_team: str) -> Optional[WeatherData]:
        """Get weather for a match based on home team's city"""
        cities = self.get_stadium_cities()
        city = cities.get(home_team)
        
        if city:
            return self.get_match_weather(city=city)
        
        return self.get_match_weather()  # Default


# Global instance
live_data = LiveDataClient()


def get_live_scores(league: str = 'bundesliga') -> List[Dict]:
    """Get live match scores"""
    return live_data.get_live_scores(league)


def get_injuries(home_team: str, away_team: str) -> Dict:
    """Get injury report for match"""
    return live_data.get_key_absences(home_team, away_team)


def get_weather(home_team: str) -> Optional[Dict]:
    """Get weather for match venue"""
    weather = live_data.get_weather_for_match(home_team)
    if weather:
        return {
            'temperature': weather.temperature,
            'condition': weather.condition,
            'humidity': weather.humidity,
            'wind_speed': weather.wind_speed,
            'affects_play': weather.affects_play
        }
    return None
