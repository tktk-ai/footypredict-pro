/**
 * FootyPredict Pro - Cloudflare Worker Prediction API
 * 
 * Lightweight serverless prediction API with real fixtures
 * Endpoints:
 *   GET /fixtures - Get real fixtures from all leagues
 *   GET /fixtures/:league - League-specific fixtures
 *   POST /predict - Get match predictions
 *   GET /health - Health check
 *   GET /models/info - Model metadata
 */

// ============= Configuration =============
const CONFIG = {
  version: "2.1.0",
  markets: ["result", "over25", "btts", "combo"],
  leagues: {
    // Europe - Major
    premier_league: { country: 'England', file: 'E0', name: '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League' },
    championship: { country: 'England', file: 'E1', name: '🏴󠁧󠁢󠁥󠁮󠁧󠁿 Championship' },
    league_one: { country: 'England', file: 'E2', name: '🏴󠁧󠁢󠁥󠁮󠁧󠁿 League One' },
    league_two: { country: 'England', file: 'E3', name: '🏴󠁧󠁢󠁥󠁮󠁧󠁿 League Two' },
    scottish_premiership: { country: 'Scotland', file: 'SC0', name: '🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish Premiership' },
    bundesliga: { country: 'Germany', file: 'D1', name: '🇩🇪 Bundesliga' },
    bundesliga_2: { country: 'Germany', file: 'D2', name: '🇩🇪 2. Bundesliga' },
    la_liga: { country: 'Spain', file: 'SP1', name: '🇪🇸 La Liga' },
    la_liga_2: { country: 'Spain', file: 'SP2', name: '🇪🇸 La Liga 2' },
    serie_a: { country: 'Italy', file: 'I1', name: '🇮🇹 Serie A' },
    serie_b: { country: 'Italy', file: 'I2', name: '🇮🇹 Serie B' },
    ligue_1: { country: 'France', file: 'F1', name: '🇫🇷 Ligue 1' },
    ligue_2: { country: 'France', file: 'F2', name: '🇫🇷 Ligue 2' },
    eredivisie: { country: 'Netherlands', file: 'N1', name: '🇳🇱 Eredivisie' },
    belgian_pro_league: { country: 'Belgium', file: 'B1', name: '🇧🇪 Jupiler Pro League' },
    primeira_liga: { country: 'Portugal', file: 'P1', name: '🇵🇹 Primeira Liga' },
    super_lig: { country: 'Turkey', file: 'T1', name: '🇹🇷 Süper Lig' },
    super_league_greece: { country: 'Greece', file: 'G1', name: '🇬🇷 Super League Greece' },
    swiss_super: { country: 'Switzerland', file: null, name: '🇨🇭 Swiss Super League' },
    austrian_bundesliga: { country: 'Austria', file: null, name: '🇦🇹 Austrian Bundesliga' },
    russian_premier: { country: 'Russia', file: null, name: '🇷🇺 Russian Premier League' },
    ukrainian_premier: { country: 'Ukraine', file: null, name: '🇺🇦 Ukrainian Premier League' },
    
    // South America
    brasileirao: { country: 'Brazil', file: null, name: '🇧🇷 Brasileirão Serie A' },
    argentina_primera: { country: 'Argentina', file: null, name: '🇦🇷 Liga Profesional' },
    colombian_primera: { country: 'Colombia', file: null, name: '🇨🇴 Categoría Primera A' },
    chilean_primera: { country: 'Chile', file: null, name: '🇨🇱 Primera División' },
    
    // North/Central America  
    mls: { country: 'USA', file: null, name: '🇺🇸 MLS' },
    liga_mx: { country: 'Mexico', file: null, name: '🇲🇽 Liga MX' },
    
    // Asia & Middle East
    j_league: { country: 'Japan', file: null, name: '🇯🇵 J1 League' },
    k_league: { country: 'South Korea', file: null, name: '🇰🇷 K League 1' },
    chinese_super: { country: 'China', file: null, name: '🇨🇳 Chinese Super League' },
    saudi_pro: { country: 'Saudi Arabia', file: null, name: '🇸🇦 Saudi Pro League' },
    indian_super: { country: 'India', file: null, name: '🇮🇳 Indian Super League' },
    a_league: { country: 'Australia', file: null, name: '🇦🇺 A-League' },
    
    // Africa
    egyptian_premier: { country: 'Egypt', file: null, name: '🇪🇬 Egyptian Premier League' },
    south_african_psl: { country: 'South Africa', file: null, name: '🇿🇦 PSL' },
    moroccan_botola: { country: 'Morocco', file: null, name: '🇲🇦 Botola Pro' },
  },
  lastUpdated: new Date().toISOString()
};

// ============= CORS Headers =============
const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
  "Content-Type": "application/json"
};

// ============= HuggingFace ML API =============
const HF_API_URL = "https://nananie143-footypredict-pro.hf.space";

// Cache for ML predictions to avoid hitting rate limits
const mlPredictionCache = new Map();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

async function fetchMLPrediction(homeTeam, awayTeam, league = '') {
  const cacheKey = `${homeTeam}_${awayTeam}_${league}`.toLowerCase();
  
  // Check cache
  const cached = mlPredictionCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return { ...cached.data, source: 'ml_cached' };
  }
  
  try {
    const url = `${HF_API_URL}/api/predict?home=${encodeURIComponent(homeTeam)}&away=${encodeURIComponent(awayTeam)}&league=${encodeURIComponent(league)}`;
    
    const response = await fetch(url, {
      method: 'GET',
      headers: { 'User-Agent': 'FootyPredict-Worker/2.1' },
      signal: AbortSignal.timeout(5000) // 5 second timeout
    });
    
    if (!response.ok) {
      console.log(`ML API returned ${response.status} for ${homeTeam} vs ${awayTeam}`);
      return null;
    }
    
    const data = await response.json();
    
    if (data.success && data.prediction) {
      const result = {
        home_prob: data.prediction.home_win_prob || 0.35,
        draw_prob: data.prediction.draw_prob || 0.28,
        away_prob: data.prediction.away_win_prob || 0.27,
        over25_prob: data.goals?.over_under?.['over_2.5'] || 0.52,
        btts_prob: data.goals?.btts?.yes || 0.48,
        confidence: data.prediction.confidence || 0.70,
        recommendation: data.prediction.predicted_outcome || 
          (data.prediction.home_win_prob > data.prediction.away_win_prob && 
           data.prediction.home_win_prob > data.prediction.draw_prob ? 'Home Win' :
           data.prediction.away_win_prob > data.prediction.home_win_prob ? 'Away Win' : 'Draw'),
        source: 'ml_model'
      };
      
      // Cache the result
      mlPredictionCache.set(cacheKey, { data: result, timestamp: Date.now() });
      
      return result;
    }
    
    return null;
  } catch (error) {
    console.log(`ML API error for ${homeTeam} vs ${awayTeam}:`, error.message);
    return null;
  }
}

// ============= Fixtures Fetcher =============

// ============= The Odds API Integration (Primary for fixtures + odds) =============
const THE_ODDS_API_BASE = 'https://api.the-odds-api.com/v4';

// League mappings for The Odds API
const THE_ODDS_API_LEAGUES = {
  'premier_league': 'soccer_epl',
  'championship': 'soccer_efl_champ',
  'la_liga': 'soccer_spain_la_liga',
  'bundesliga': 'soccer_germany_bundesliga',
  'serie_a': 'soccer_italy_serie_a',
  'ligue_1': 'soccer_france_ligue_one',
  'eredivisie': 'soccer_netherlands_eredivisie',
  'primeira_liga': 'soccer_portugal_primeira_liga',
  'mls': 'soccer_usa_mls',
  'liga_mx': 'soccer_mexico_ligamx',
  'a_league': 'soccer_australia_aleague',
  'scottish_premiership': 'soccer_spl'
};

// Cache for The Odds API
const theOddsApiCache = new Map();
const THE_ODDS_API_CACHE_TTL = 10 * 60 * 1000; // 10 minutes (to conserve requests)

// Fetch fixtures with odds from The Odds API
async function fetchTheOddsApiFixtures(apiKey, leagues = ['soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga', 'soccer_italy_serie_a', 'soccer_france_ligue_one']) {
  if (!apiKey) return [];
  
  const allFixtures = [];
  
  for (const league of leagues) {
    const cacheKey = `the_odds_api_${league}`;
    const cached = theOddsApiCache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < THE_ODDS_API_CACHE_TTL) {
      allFixtures.push(...cached.data);
      continue;
    }
    
    try {
      const url = `${THE_ODDS_API_BASE}/sports/${league}/odds/?apiKey=${apiKey}&regions=uk,eu&markets=h2h,totals&oddsFormat=decimal`;
      const response = await fetch(url, {
        signal: AbortSignal.timeout(10000)
      });
      
      if (!response.ok) continue;
      
      const matches = await response.json();
      
      const fixtures = matches.map(match => {
        // Parse odds from first bookmaker
        const bm = match.bookmakers?.[0];
        const h2hMarket = bm?.markets?.find(m => m.key === 'h2h');
        const totalsMarket = bm?.markets?.find(m => m.key === 'totals');
        
        let odds = { home: 0, draw: 0, away: 0, over25: 0, under25: 0, source: 'the-odds-api' };
        if (h2hMarket) {
          const outcomes = {};
          for (const o of h2hMarket.outcomes) {
            outcomes[o.name] = o.price;
          }
          odds.home = outcomes[match.home_team] || 0;
          odds.away = outcomes[match.away_team] || 0;
          odds.draw = outcomes['Draw'] || 0;
          odds.bookmaker = bm.title;
        }
        if (totalsMarket) {
          const over = totalsMarket.outcomes.find(o => o.name === 'Over');
          const under = totalsMarket.outcomes.find(o => o.name === 'Under');
          odds.over25 = over?.price || 0;
          odds.under25 = under?.price || 0;
        }
        
        // Parse date/time
        const startDate = new Date(match.commence_time);
        
        // Map league key to friendly name
        const leagueNames = {
          'soccer_epl': 'Premier League',
          'soccer_spain_la_liga': 'La Liga',
          'soccer_germany_bundesliga': 'Bundesliga',
          'soccer_italy_serie_a': 'Serie A',
          'soccer_france_ligue_one': 'Ligue 1',
          'soccer_netherlands_eredivisie': 'Eredivisie',
          'soccer_portugal_primeira_liga': 'Primeira Liga',
          'soccer_efl_champ': 'Championship',
          'soccer_usa_mls': 'MLS',
          'soccer_mexico_ligamx': 'Liga MX'
        };
        
        const countryMap = {
          'soccer_epl': 'England',
          'soccer_spain_la_liga': 'Spain',
          'soccer_germany_bundesliga': 'Germany',
          'soccer_italy_serie_a': 'Italy',
          'soccer_france_ligue_one': 'France',
          'soccer_netherlands_eredivisie': 'Netherlands',
          'soccer_portugal_primeira_liga': 'Portugal',
          'soccer_efl_champ': 'England',
          'soccer_usa_mls': 'USA',
          'soccer_mexico_ligamx': 'Mexico'
        };
        
        return {
          date: startDate.toISOString().split('T')[0],
          time: startDate.toISOString().split('T')[1].substring(0, 5),
          home_team: match.home_team,
          away_team: match.away_team,
          status: 'scheduled',
          league_name: leagueNames[match.sport_key] || match.sport_title,
          country: countryMap[match.sport_key] || 'Unknown',
          source: 'the-odds-api',
          odds: odds
        };
      });
      
      theOddsApiCache.set(cacheKey, { data: fixtures, timestamp: Date.now() });
      allFixtures.push(...fixtures);
    } catch (error) {
      console.log(`The Odds API error for ${league}:`, error.message);
    }
  }
  
  return allFixtures;
}

// ============= Game Forecast API Integration (RapidAPI) =============
const GAME_FORECAST_BASE = 'https://game-forecast-api.p.rapidapi.com';


// Cache for Game Forecast data
const gameForecastCache = new Map();
const GAME_FORECAST_CACHE_TTL = 15 * 60 * 1000; // 15 minutes

// Fetch upcoming events with AI predictions from Game Forecast API
async function fetchGameForecastPredictions(apiKey, pageSize = 50) {
  if (!apiKey) return [];
  
  const cacheKey = 'game_forecast_events';
  const cached = gameForecastCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < GAME_FORECAST_CACHE_TTL) {
    return cached.data;
  }
  
  try {
    const url = `${GAME_FORECAST_BASE}/events?status_code=NOT_STARTED&page_size=${pageSize}`;
    const response = await fetch(url, {
      headers: {
        'x-rapidapi-host': 'game-forecast-api.p.rapidapi.com',
        'x-rapidapi-key': apiKey
      },
      signal: AbortSignal.timeout(10000)
    });
    
    if (!response.ok) return [];
    
    const data = await response.json();
    const events = data.data || [];
    
    // Transform to our fixture format
    const fixtures = events.map(event => {
      const prediction = event.predictions?.[0];
      const matchResult = prediction?.match_result || {};
      const odds = event.odds?.find(o => o.key === 'match_winner')?.values || {};
      const over25Odds = event.odds?.find(o => o.key === 'over_under_25')?.values || {};
      
      return {
        date: event.start_at?.split('T')[0],
        time: event.start_at?.split('T')[1]?.substring(0, 5),
        home_team: event.team_home?.name,
        away_team: event.team_away?.name,
        status: event.status_code === 'NOT_STARTED' ? 'scheduled' : event.status_code?.toLowerCase(),
        league_name: event.league?.name,
        league_id: event.league?.id,
        country: event.league?.country_code,
        round: event.round,
        referee: event.referee,
        source: 'game-forecast-api',
        prediction: {
          result: {
            home: (matchResult.home || 33) / 100,
            draw: (matchResult.draw || 33) / 100,
            away: (matchResult.away || 33) / 100,
            recommendation: matchResult.home > matchResult.draw && matchResult.home > matchResult.away ? 'Home Win' :
                           matchResult.away > matchResult.home && matchResult.away > matchResult.draw ? 'Away Win' : 'Draw'
          }
        },
        confidence: Math.max(matchResult.home || 0, matchResult.draw || 0, matchResult.away || 0) / 100,
        prediction_source: 'game-forecast-ai',
        odds: {
          home: odds.Home || 0,
          draw: odds.Draw || 0,
          away: odds.Away || 0,
          over25: over25Odds.Over || 0,
          under25: over25Odds.Under || 0,
          source: 'game-forecast-api'
        }
      };
    }).filter(f => f.home_team && f.away_team);
    
    gameForecastCache.set(cacheKey, { data: fixtures, timestamp: Date.now() });
    return fixtures;
  } catch (error) {
    console.log('Game Forecast API error:', error.message);
    return [];
  }
}
const SPORTRADAR_BASE = 'https://api.sportradar.com/soccer/trial/v4/en';

// Cache for Sportradar data to respect rate limits
const sportradarCache = new Map();
const SPORTRADAR_CACHE_TTL = 30 * 60 * 1000; // 30 minutes

// Map common team names to Sportradar IDs (expanded as needed)
const teamIdMap = {
  'manchester city': 'sr:competitor:17',
  'man city': 'sr:competitor:17',
  'arsenal': 'sr:competitor:42',
  'arsenal fc': 'sr:competitor:42',
  'liverpool': 'sr:competitor:44',
  'liverpool fc': 'sr:competitor:44',
  'chelsea': 'sr:competitor:38',
  'chelsea fc': 'sr:competitor:38',
  'manchester united': 'sr:competitor:35',
  'man united': 'sr:competitor:35',
  'tottenham': 'sr:competitor:33',
  'tottenham hotspur': 'sr:competitor:33',
  'newcastle': 'sr:competitor:39',
  'newcastle united': 'sr:competitor:39',
  'aston villa': 'sr:competitor:40',
  'brighton': 'sr:competitor:30',
  'west ham': 'sr:competitor:37',
  'crystal palace': 'sr:competitor:31',
  'fulham': 'sr:competitor:43',
  'brentford': 'sr:competitor:7629',
  'nottingham forest': 'sr:competitor:36',
  'bournemouth': 'sr:competitor:60',
  'everton': 'sr:competitor:48',
  'wolverhampton': 'sr:competitor:3',
  'wolves': 'sr:competitor:3',
  'leicester': 'sr:competitor:46',
  'ipswich': 'sr:competitor:29',
  'southampton': 'sr:competitor:45',
  // La Liga
  'real madrid': 'sr:competitor:2829',
  'barcelona': 'sr:competitor:2817',
  'fc barcelona': 'sr:competitor:2817',
  'atletico madrid': 'sr:competitor:2836',
  // Bundesliga
  'bayern munich': 'sr:competitor:2672',
  'bayern': 'sr:competitor:2672',
  'borussia dortmund': 'sr:competitor:2673',
  'dortmund': 'sr:competitor:2673',
  'rb leipzig': 'sr:competitor:36382',
  'bayer leverkusen': 'sr:competitor:2681',
  // Serie A
  'juventus': 'sr:competitor:2953',
  'inter milan': 'sr:competitor:2957',
  'inter': 'sr:competitor:2957',
  'ac milan': 'sr:competitor:2948',
  'milan': 'sr:competitor:2948',
  'napoli': 'sr:competitor:2963',
  'roma': 'sr:competitor:2954',
  'as roma': 'sr:competitor:2954',
  // Ligue 1
  'psg': 'sr:competitor:2847',
  'paris saint-germain': 'sr:competitor:2847',
  'marseille': 'sr:competitor:2859',
  'monaco': 'sr:competitor:2857',
};

// Season IDs for major leagues (2024-25 season)
const seasonMap = {
  'premier_league': 'sr:season:118689',
  'la_liga': 'sr:season:118691',
  'bundesliga': 'sr:season:118687',
  'serie_a': 'sr:season:118693',
  'ligue_1': 'sr:season:118695',
  'eredivisie': 'sr:season:118699',
  'primeira_liga': 'sr:season:118701',
};

// Get Sportradar team ID from name
function getSportradarTeamId(teamName) {
  const normalized = teamName.toLowerCase().trim();
  return teamIdMap[normalized] || null;
}

// Fetch team profile from Sportradar
async function fetchSportradarTeamProfile(teamId, apiKey) {
  if (!apiKey || !teamId) return null;
  
  const cacheKey = `team_${teamId}`;
  const cached = sportradarCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < SPORTRADAR_CACHE_TTL) {
    return cached.data;
  }
  
  try {
    const url = `${SPORTRADAR_BASE}/competitors/${teamId}/profile.json?api_key=${apiKey}`;
    const response = await fetch(url, {
      signal: AbortSignal.timeout(5000)
    });
    
    if (!response.ok) return null;
    
    const data = await response.json();
    const result = {
      id: teamId,
      name: data.competitor?.name,
      country: data.competitor?.country,
      manager: data.manager?.name || 'Unknown',
      venue: data.venue?.name || 'Unknown',
      squadSize: data.players?.length || 0
    };
    
    sportradarCache.set(cacheKey, { data: result, timestamp: Date.now() });
    return result;
  } catch (error) {
    console.log(`Sportradar team profile error for ${teamId}:`, error.message);
    return null;
  }
}

// Fetch league standings from Sportradar
async function fetchSportradarStandings(seasonId, apiKey) {
  if (!apiKey || !seasonId) return null;
  
  const cacheKey = `standings_${seasonId}`;
  const cached = sportradarCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < SPORTRADAR_CACHE_TTL) {
    return cached.data;
  }
  
  try {
    const url = `${SPORTRADAR_BASE}/seasons/${seasonId}/standings.json?api_key=${apiKey}`;
    const response = await fetch(url, {
      signal: AbortSignal.timeout(5000)
    });
    
    if (!response.ok) return null;
    
    const data = await response.json();
    const standings = data.standings?.[0]?.groups?.[0]?.standings || [];
    
    const result = standings.map(s => ({
      teamId: s.competitor?.id,
      teamName: s.competitor?.name,
      position: s.rank,
      points: s.points,
      played: s.played,
      won: s.win,
      drawn: s.draw,
      lost: s.loss,
      goalsFor: s.goals_for,
      goalsAgainst: s.goals_against,
      form: s.current_outcome || null
    }));
    
    sportradarCache.set(cacheKey, { data: result, timestamp: Date.now() });
    return result;
  } catch (error) {
    console.log(`Sportradar standings error for ${seasonId}:`, error.message);
    return null;
  }
}

// Get team position from standings
function getTeamPosition(standings, teamName) {
  if (!standings || !Array.isArray(standings)) return null;
  const normalized = teamName.toLowerCase().trim();
  const team = standings.find(s => 
    s.teamName?.toLowerCase().includes(normalized) ||
    normalized.includes(s.teamName?.toLowerCase())
  );
  return team || null;
}

// Enrich fixture with Sportradar data
async function enrichWithSportradar(fixture, apiKey) {
  if (!apiKey) return fixture;
  
  const enriched = { ...fixture, sportradar: {} };
  
  // Get team IDs
  const homeId = getSportradarTeamId(fixture.home_team);
  const awayId = getSportradarTeamId(fixture.away_team);
  
  // Fetch team profiles (in parallel)
  const [homeProfile, awayProfile] = await Promise.all([
    homeId ? fetchSportradarTeamProfile(homeId, apiKey) : null,
    awayId ? fetchSportradarTeamProfile(awayId, apiKey) : null
  ]);
  
  if (homeProfile) {
    enriched.sportradar.home_manager = homeProfile.manager;
    enriched.sportradar.home_venue = homeProfile.venue;
  }
  
  if (awayProfile) {
    enriched.sportradar.away_manager = awayProfile.manager;
  }
  
  // Try to get standings for league position
  let leagueKey = null;
  const leagueName = (fixture.league_name || '').toLowerCase();
  
  if (leagueName.includes('premier') || leagueName.includes('england')) {
    leagueKey = 'premier_league';
  } else if (leagueName.includes('la liga') || leagueName.includes('spain')) {
    leagueKey = 'la_liga';
  } else if (leagueName.includes('bundesliga') || leagueName.includes('german')) {
    leagueKey = 'bundesliga';
  } else if (leagueName.includes('serie a') || leagueName.includes('italy')) {
    leagueKey = 'serie_a';
  } else if (leagueName.includes('ligue 1') || leagueName.includes('france')) {
    leagueKey = 'ligue_1';
  }
  
  if (leagueKey && seasonMap[leagueKey]) {
    const standings = await fetchSportradarStandings(seasonMap[leagueKey], apiKey);
    
    const homeStats = getTeamPosition(standings, fixture.home_team);
    const awayStats = getTeamPosition(standings, fixture.away_team);
    
    if (homeStats) {
      enriched.sportradar.home_position = homeStats.position;
      enriched.sportradar.home_points = homeStats.points;
      enriched.sportradar.home_form = homeStats.form;
    }
    
    if (awayStats) {
      enriched.sportradar.away_position = awayStats.position;
      enriched.sportradar.away_points = awayStats.points;
      enriched.sportradar.away_form = awayStats.form;
    }
  }
  
  // Only return enriched data if we found something
  if (Object.keys(enriched.sportradar).length > 0) {
    return enriched;
  }
  
  return fixture;
}

// ============= Odds-API.io v3 Integration (250+ Bookmakers) =============
const ODDS_API_IO_BASE = 'https://api.odds-api.io/v3';

// Cache for odds data
const oddsCache = new Map();
const ODDS_CACHE_TTL = 10 * 60 * 1000; // 10 minutes

// Map league names to Odds-API.io league slugs
const oddsApiLeagueSlugs = {
  'premier_league': 'england-premier-league',
  'la_liga': 'spain-la-liga',
  'bundesliga': 'germany-bundesliga',
  'serie_a': 'italy-serie-a',
  'ligue_1': 'france-ligue-1',
  'champions_league': 'uefa-champions-league',
  'europa_league': 'uefa-europa-league',
  'eredivisie': 'netherlands-eredivisie'
};

// Fetch events for a league
async function fetchOddsApiEvents(leagueSlug, apiKey) {
  if (!apiKey || !leagueSlug) return [];
  
  const cacheKey = `events_${leagueSlug}`;
  const cached = oddsCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < ODDS_CACHE_TTL) {
    return cached.data;
  }
  
  try {
    const url = `${ODDS_API_IO_BASE}/events?sport=football&league=${leagueSlug}&apiKey=${apiKey}`;
    const response = await fetch(url, {
      signal: AbortSignal.timeout(8000)
    });
    
    if (!response.ok) return [];
    
    const events = await response.json();
    oddsCache.set(cacheKey, { data: events, timestamp: Date.now() });
    return events;
  } catch (error) {
    console.log(`Odds-API.io events error for ${leagueSlug}:`, error.message);
    return [];
  }
}

// Fetch odds for a specific event
async function fetchOddsForEvent(eventId, apiKey) {
  if (!apiKey || !eventId) return null;
  
  const cacheKey = `odds_${eventId}`;
  const cached = oddsCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < ODDS_CACHE_TTL) {
    return cached.data;
  }
  
  try {
    // Use Bet365 as primary bookmaker (most reliable)
    const url = `${ODDS_API_IO_BASE}/odds?eventId=${eventId}&bookmakers=Bet365&apiKey=${apiKey}`;
    const response = await fetch(url, {
      signal: AbortSignal.timeout(8000)
    });
    
    if (!response.ok) return null;
    
    const oddsData = await response.json();
    oddsCache.set(cacheKey, { data: oddsData, timestamp: Date.now() });
    return oddsData;
  } catch (error) {
    console.log(`Odds-API.io odds error for event ${eventId}:`, error.message);
    return null;
  }
}

// Parse odds data from Odds-API.io response
function parseOddsData(oddsData) {
  if (!oddsData || !oddsData.bookmakers) return null;
  
  const result = {
    home_odds: 0,
    draw_odds: 0,
    away_odds: 0,
    over25: 0,
    under25: 0,
    btts_yes: 0,
    btts_no: 0,
    double_chance: {},
    bookmaker: null,
    source: 'odds-api.io'
  };
  
  for (const [bookmaker, markets] of Object.entries(oddsData.bookmakers)) {
    result.bookmaker = bookmaker;
    
    for (const market of markets) {
      if (market.name === 'ML' && market.odds && market.odds[0]) {
        result.home_odds = parseFloat(market.odds[0].home) || 0;
        result.draw_odds = parseFloat(market.odds[0].draw) || 0;
        result.away_odds = parseFloat(market.odds[0].away) || 0;
      }
      
      if ((market.name === 'Goals Over/Under' || market.name === 'Totals') && market.odds) {
        const over25Market = market.odds.find(o => o.hdp === 2.5 || o.hdp === '2.5');
        if (over25Market) {
          result.over25 = parseFloat(over25Market.over) || 0;
          result.under25 = parseFloat(over25Market.under) || 0;
        }
      }
      
      if (market.name === 'Both Teams To Score' && market.odds && market.odds[0]) {
        result.btts_yes = parseFloat(market.odds[0].yes) || 0;
        result.btts_no = parseFloat(market.odds[0].no) || 0;
      }
      
      if (market.name === 'Double Chance' && market.odds) {
        for (const dc of market.odds) {
          if (dc.label === 'Home or Draw' || dc.label?.includes('or Draw')) {
            result.double_chance['1X'] = parseFloat(dc.under || dc.odds) || 0;
          } else if (dc.label === 'Draw or Away' || dc.label?.includes('Draw or')) {
            result.double_chance['X2'] = parseFloat(dc.under || dc.odds) || 0;
          } else if (dc.label === 'Home or Away' || dc.label?.includes('or ')) {
            result.double_chance['12'] = parseFloat(dc.under || dc.odds) || 0;
          }
        }
      }
    }
    break; // Use first bookmaker
  }
  
  return result;
}

// Match fixture to event from Odds-API.io
function matchFixtureToEvent(fixture, events) {
  if (!events || events.length === 0) return null;
  
  const homeTeam = (fixture.home_team || '').toLowerCase();
  const awayTeam = (fixture.away_team || '').toLowerCase();
  
  for (const event of events) {
    const eventHome = (event.home || '').toLowerCase();
    const eventAway = (event.away || '').toLowerCase();
    
    // Exact match
    if (eventHome === homeTeam && eventAway === awayTeam) return event;
    
    // Fuzzy match (first word)
    const homeWord = homeTeam.split(' ')[0];
    const awayWord = awayTeam.split(' ')[0];
    
    if (eventHome.includes(homeWord) && eventAway.includes(awayWord)) return event;
    if (homeTeam.includes(eventHome.split(' ')[0]) && awayTeam.includes(eventAway.split(' ')[0])) return event;
  }
  
  return null;
}

// Enrich fixture with odds from Odds-API.io
async function enrichWithOdds(fixture, apiKey) {
  if (!apiKey) return fixture;
  
  // Determine league slug from league name AND country
  let leagueSlug = null;
  const leagueName = (fixture.league_name || '').toLowerCase();
  const country = (fixture.country || '').toLowerCase();
  
  // English Premier League - use country to be precise
  if ((leagueName === 'premier league' || leagueName.includes('premier league')) && 
      (country === 'england' || country === 'uk' || country === 'united kingdom' || country === '')) {
    // If country is empty, check for known EPL teams
    const homeTeam = (fixture.home_team || '').toLowerCase();
    const awayTeam = (fixture.away_team || '').toLowerCase();
    const eplTeams = ['arsenal', 'chelsea', 'liverpool', 'manchester', 'tottenham', 'everton', 'leeds', 'newcastle', 'brighton', 'wolves', 'aston villa', 'brentford', 'fulham', 'crystal palace', 'nottingham forest', 'bournemouth', 'west ham', 'leicester', 'ipswich'];
    
    if (country === 'england' || eplTeams.some(t => homeTeam.includes(t) || awayTeam.includes(t))) {
      leagueSlug = 'england-premier-league';
    }
  } else if ((leagueName.includes('la liga') || leagueName.includes('laliga')) && 
             (country === 'spain' || country === '')) {
    leagueSlug = 'spain-la-liga';
  } else if (leagueName.includes('bundesliga') && 
             (country === 'germany' || country === '') && !leagueName.includes('austria')) {
    leagueSlug = 'germany-bundesliga';
  } else if (leagueName.includes('serie a') && 
             (country === 'italy' || country === '') && !leagueName.includes('brazil')) {
    leagueSlug = 'italy-serie-a';
  } else if (leagueName.includes('ligue 1') && (country === 'france' || country === '')) {
    leagueSlug = 'france-ligue-1';
  } else if (leagueName.includes('champions league')) {
    leagueSlug = 'uefa-champions-league';
  }
  
  if (!leagueSlug) return fixture;
  
  // Get events for this league
  const events = await fetchOddsApiEvents(leagueSlug, apiKey);
  
  // Find matching event
  const matchingEvent = matchFixtureToEvent(fixture, events);
  if (!matchingEvent) return fixture;
  
  // Fetch odds for this event
  const oddsData = await fetchOddsForEvent(matchingEvent.id, apiKey);
  if (!oddsData) return fixture;
  
  // Parse odds
  const parsedOdds = parseOddsData(oddsData);
  if (!parsedOdds) return fixture;
  
  return {
    ...fixture,
    odds: {
      home: parsedOdds.home_odds,
      draw: parsedOdds.draw_odds,
      away: parsedOdds.away_odds,
      over25: parsedOdds.over25,
      under25: parsedOdds.under25,
      btts_yes: parsedOdds.btts_yes,
      btts_no: parsedOdds.btts_no,
      double_chance: parsedOdds.double_chance,
      bookmaker: parsedOdds.bookmaker,
      source: 'odds-api.io'
    }
  };
}
async function fetchOpenLigaDBFixtures(league = 'bl1', season = '2024') {
  const url = `https://api.openligadb.de/getmatchdata/${league}/${season}`;
  
  try {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'FootyPredict/2.0' }
    });
    
    if (!response.ok) return [];
    
    const matches = await response.json();
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    return matches.map(m => {
      const matchDate = new Date(m.matchDateTime || m.matchDateTimeUTC);
      return {
        date: matchDate.toISOString().split('T')[0],
        time: matchDate.toTimeString().slice(0, 5),
        home_team: m.team1?.teamName || 'Unknown',
        away_team: m.team2?.teamName || 'Unknown',
        home_score: m.matchResults?.[0]?.pointsTeam1 ?? null,
        away_score: m.matchResults?.[0]?.pointsTeam2 ?? null,
        status: m.matchIsFinished ? 'finished' : 'scheduled',
        home_odds: null,
        draw_odds: null,
        away_odds: null,
        source: 'openligadb'
      };
    });
  } catch (error) {
    console.error('OpenLigaDB error:', error);
    return [];
  }
}

// TheSportsDB - Fetch live/today's matches for 24/7 coverage
async function fetchTodaysMatches() {
  // TheSportsDB free API for today's live soccer events
  const url = 'https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d=' + 
              new Date().toISOString().split('T')[0] + '&s=Soccer';
  
  try {
    const response = await fetch(url);
    if (!response.ok) return [];
    
    const data = await response.json();
    if (!data.events) return [];
    
    return data.events.map(e => ({
      date: e.dateEvent,
      time: e.strTime || '15:00',
      home_team: e.strHomeTeam,
      away_team: e.strAwayTeam,
      home_score: e.intHomeScore ? parseInt(e.intHomeScore) : null,
      away_score: e.intAwayScore ? parseInt(e.intAwayScore) : null,
      status: e.strStatus === 'Match Finished' ? 'finished' : 
              e.strStatus === 'Not Started' ? 'scheduled' : 'live',
      league_name: e.strLeague,
      country: e.strCountry || '',
      source: 'thesportsdb_today'
    }));
  } catch (error) {
    console.error('Today matches error:', error);
    return [];
  }
}

// Fetch yesterday's matches for recent results
async function fetchYesterdaysMatches() {
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const url = 'https://www.thesportsdb.com/api/v1/json/3/eventsday.php?d=' + 
              yesterday.toISOString().split('T')[0] + '&s=Soccer';
  
  try {
    const response = await fetch(url);
    if (!response.ok) return [];
    
    const data = await response.json();
    if (!data.events) return [];
    
    return data.events.map(e => ({
      date: e.dateEvent,
      time: e.strTime || '15:00',
      home_team: e.strHomeTeam,
      away_team: e.strAwayTeam,
      home_score: e.intHomeScore ? parseInt(e.intHomeScore) : null,
      away_score: e.intAwayScore ? parseInt(e.intAwayScore) : null,
      status: 'finished',
      league_name: e.strLeague,
      country: e.strCountry || '',
      source: 'thesportsdb_yesterday'
    }));
  } catch (error) {
    console.error('Yesterday matches error:', error);
    return [];
  }
}

// TheSportsDB - Free API with upcoming fixtures for major leagues
async function fetchTheSportsDBFixtures(leagueId = '4328') {
  const url = `https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php?id=${leagueId}`;
  
  try {
    const response = await fetch(url);
    if (!response.ok) return [];
    
    const data = await response.json();
    if (!data.events) return [];
    
    return data.events.map(e => {
      // Properly detect if match is finished (handle null, undefined, empty string)
      const hasScore = e.intHomeScore !== null && e.intHomeScore !== undefined && e.intHomeScore !== '';
      
      return {
        date: e.dateEvent,
        time: e.strTime || '15:00',
        home_team: e.strHomeTeam,
        away_team: e.strAwayTeam,
        home_score: hasScore ? parseInt(e.intHomeScore) : null,
        away_score: e.intAwayScore && e.intAwayScore !== '' ? parseInt(e.intAwayScore) : null,
        status: hasScore ? 'finished' : 'scheduled',
        home_odds: null,
        draw_odds: null,
        away_odds: null,
        league_from_api: e.strLeague,  // Use actual league from API
        country_from_api: e.strCountry || '',
        source: 'thesportsdb'
      };
    });
  } catch (error) {
    console.error('TheSportsDB error:', error);
    return [];
  }
}

// API-Football - Premium data source with 1200+ fixtures per day
// Docs: https://www.api-football.com/documentation-v3
const API_FOOTBALL_KEY = '29856afe069eaca8d8ecb3049132d8f8';
const API_FOOTBALL_BASE = 'https://v3.football.api-sports.io';

// Major league IDs for API-Football
const API_FOOTBALL_LEAGUES = {
  // England
  premier_league: 39,
  championship: 40,
  league_one: 41,
  league_two: 42,
  // Germany
  bundesliga: 78,
  bundesliga_2: 79,
  // Spain
  la_liga: 140,
  la_liga_2: 141,
  // Italy
  serie_a: 135,
  serie_b: 136,
  // France
  ligue_1: 61,
  ligue_2: 62,
  // Netherlands
  eredivisie: 88,
  // Portugal
  primeira_liga: 94,
  // Belgium
  belgian_pro_league: 144,
  // Turkey
  super_lig: 203,
  // Scotland
  scottish_premiership: 179,
  // Greece
  super_league_greece: 197,
  // Brazil
  brasileirao: 71,
  // Argentina
  argentina_primera: 128,
  // USA
  mls: 253,
  // Mexico
  liga_mx: 262,
  // European competitions
  champions_league: 2,
  europa_league: 3,
  conference_league: 848
};

// Fetch fixtures from API-Football for a specific date range
async function fetchAPIFootballFixtures(days = 7) {
  const fixtures = [];
  const today = new Date();
  
  try {
    // Fetch fixtures for each day in the range
    for (let i = 0; i <= Math.min(days, 7); i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      const dateStr = date.toISOString().split('T')[0];
      
      const response = await fetch(`${API_FOOTBALL_BASE}/fixtures?date=${dateStr}&timezone=UTC`, {
        headers: {
          'x-apisports-key': API_FOOTBALL_KEY
        }
      });
      
      if (!response.ok) {
        console.error(`API-Football error for ${dateStr}: ${response.status}`);
        continue;
      }
      
      const data = await response.json();
      
      if (data.errors && Object.keys(data.errors).length > 0) {
        console.error('API-Football errors:', data.errors);
        continue;
      }
      
      const dayFixtures = (data.response || []).map(f => ({
        date: f.fixture.date.split('T')[0],
        time: f.fixture.date.split('T')[1]?.substring(0, 5) || '15:00',
        home_team: f.teams.home.name,
        away_team: f.teams.away.name,
        home_score: f.goals.home,
        away_score: f.goals.away,
        status: mapAPIFootballStatus(f.fixture.status.short),
        league: f.league.name,
        league_name: f.league.name,
        league_id: f.league.id,
        country: f.league.country,
        home_logo: f.teams.home.logo,
        away_logo: f.teams.away.logo,
        league_logo: f.league.logo,
        venue: f.fixture.venue?.name || null,
        fixture_id: f.fixture.id,
        source: 'api-football'
      }));
      
      fixtures.push(...dayFixtures);
    }
    
    return fixtures;
  } catch (error) {
    console.error('API-Football fetch error:', error);
    return [];
  }
}

// Map API-Football status codes to our status
function mapAPIFootballStatus(status) {
  const statusMap = {
    'NS': 'scheduled',    // Not Started
    'TBD': 'scheduled',   // Time To Be Defined
    '1H': 'live',         // First Half
    'HT': 'live',         // Halftime
    '2H': 'live',         // Second Half
    'ET': 'live',         // Extra Time
    'P': 'live',          // Penalty In Progress
    'FT': 'finished',     // Match Finished
    'AET': 'finished',    // Match Finished After Extra Time
    'PEN': 'finished',    // Match Finished After Penalty
    'BT': 'live',         // Break Time
    'SUSP': 'suspended',  // Match Suspended
    'INT': 'suspended',   // Match Interrupted
    'PST': 'postponed',   // Match Postponed
    'CANC': 'cancelled',  // Match Cancelled
    'ABD': 'cancelled',   // Match Abandoned
    'AWD': 'finished',    // Technical Loss
    'WO': 'finished'      // WalkOver
  };
  return statusMap[status] || 'scheduled';
}

// League IDs for TheSportsDB - verified unique IDs for major leagues
const SPORTSDB_LEAGUES = {
  // Europe - Major (verified unique IDs)
  premier_league: '4328',
  la_liga: '4335',
  bundesliga: '4331',
  serie_a: '4332',
  ligue_1: '4334',
  eredivisie: '4337',
  primeira_liga: '4344',
  scottish_premiership: '4330',
  belgian_pro_league: '4355',
  super_lig_turkey: '4339',
  
  // Europe - Secondary (verified)
  championship: '4329',
  serie_b_italy: '4401',
  segunda_division: '4378',
  bundesliga_2: '4403',
  russian_premier: '4470',  // Fixed: was duplicate of primeira_liga
  ukrainian_premier: '4454',
  swiss_super: '4392',
  austrian_bundesliga: '4408',  // Fixed: was duplicate
  greek_super: '4396',
  
  // South America (verified)
  brasileirao: '4351',
  argentina_primera: '4406',
  colombian_primera: '4441',
  chilean_primera: '4439',
  
  // North/Central America (verified)
  mls: '4346',
  liga_mx: '4350',
  
  // Asia (set unique or null for unavailable)
  j_league: '4388',  // Fixed: was duplicate of liga_mx
  k_league: '4395',
  chinese_super: null,  // Fixed: was duplicate - no valid ID known
  saudi_pro: '4536',  // Fixed: was duplicate  
  indian_super: '4423',  // Fixed: was duplicate
  a_league: '4356',
  
  // Africa (set null for unavailable to prevent duplicates)
  egyptian_premier: '4436',  // Fixed
  south_african_psl: '4480',  // Fixed
  moroccan_botola: null  // No valid ID - disable to prevent duplicates
};

async function fetchFootballDataCSV(leagueFile, season = null) {
  // Current season code (e.g., "2425" for 2024/25)
  if (!season) {
    const now = new Date();
    const year = now.getMonth() >= 7 ? now.getFullYear() : now.getFullYear() - 1;
    season = `${String(year).slice(-2)}${String(year + 1).slice(-2)}`;
  }
  
  const url = `https://www.football-data.co.uk/mmz4281/${season}/${leagueFile}.csv`;
  
  try {
    const response = await fetch(url, {
      headers: { 'User-Agent': 'FootyPredict/2.0' }
    });
    
    if (!response.ok) return [];
    
    const csvText = await response.text();
    return parseCSV(csvText);
  } catch (error) {
    console.error(`Error fetching ${leagueFile}:`, error);
    return [];
  }
}

function parseCSV(csvText) {
  const lines = csvText.split('\n');
  if (lines.length < 2) return [];
  
  const headers = lines[0].split(',').map(h => h.trim());
  const matches = [];
  
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(',');
    if (values.length < 5) continue;
    
    const row = {};
    headers.forEach((h, idx) => { row[h] = values[idx]?.trim() || ''; });
    
    if (!row.Date || !row.HomeTeam) continue;
    
    // Parse date (DD/MM/YYYY format)
    const dateParts = row.Date.split('/');
    if (dateParts.length !== 3) continue;
    
    const matchDate = `${dateParts[2]}-${dateParts[1].padStart(2,'0')}-${dateParts[0].padStart(2,'0')}`;
    
    matches.push({
      date: matchDate,
      time: row.Time || '15:00',
      home_team: row.HomeTeam,
      away_team: row.AwayTeam,
      home_score: row.FTHG ? parseInt(row.FTHG) : null,
      away_score: row.FTAG ? parseInt(row.FTAG) : null,
      status: row.FTHG ? 'finished' : 'scheduled',
      home_odds: parseFloat(row.B365H) || parseFloat(row.AvgH) || null,
      draw_odds: parseFloat(row.B365D) || parseFloat(row.AvgD) || null,
      away_odds: parseFloat(row.B365A) || parseFloat(row.AvgA) || null,
    });
  }
  
  return matches;
}

// ============= Prediction Engine =============
// Generate dynamic predictions based on team characteristics
function hashTeam(name) {
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = ((hash << 5) - hash) + name.charCodeAt(i);
    hash = hash & hash;
  }
  return Math.abs(hash);
}

function getTeamStrength(teamName) {
  // Generate consistent strength factor (0.3 - 0.9) based on team name
  const hash = hashTeam(teamName.toLowerCase());
  return 0.3 + (hash % 60) / 100;
}

function predictMatch(homeTeam, awayTeam, options = {}) {
  const timestamp = new Date().toISOString();
  const matchId = `${homeTeam.toLowerCase().replace(/\s+/g, '_')}_vs_${awayTeam.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
  
  // Use odds if available to improve predictions
  let homeProb, drawProb, awayProb, over25Prob, bttsProb, confidence, predictionSource;
  
  // Priority 1: ML model predictions (from HuggingFace)
  if (options.mlPrediction) {
    const ml = options.mlPrediction;
    homeProb = ml.home_prob;
    drawProb = ml.draw_prob;
    awayProb = ml.away_prob;
    over25Prob = ml.over25_prob || 0.52;
    bttsProb = ml.btts_prob || 0.48;
    confidence = ml.confidence || 0.70;
    predictionSource = ml.source || 'ml_model';
  }
  // Priority 2: Odds-based predictions
  else if (options.home_odds && options.draw_odds && options.away_odds) {
    const totalInv = 1/options.home_odds + 1/options.draw_odds + 1/options.away_odds;
    homeProb = (1/options.home_odds) / totalInv;
    drawProb = (1/options.draw_odds) / totalInv;
    awayProb = (1/options.away_odds) / totalInv;
    
    const homeAttack = getTeamStrength(homeTeam + '_attack');
    const awayAttack = getTeamStrength(awayTeam + '_attack');
    const combinedAttack = (homeAttack + awayAttack) / 2;
    over25Prob = 0.40 + combinedAttack * 0.25;
    bttsProb = 0.35 + combinedAttack * 0.28;
    
    const maxProb = Math.max(homeProb, drawProb, awayProb);
    confidence = 0.70 + maxProb * 0.15;
    predictionSource = 'odds_based';
  }
  // Priority 3: Local team-strength algorithm (fallback)
  else {
    const homeStrength = getTeamStrength(homeTeam);
    const awayStrength = getTeamStrength(awayTeam);
    const homeAdvantage = 0.12;
    
    const homeScore = homeStrength + homeAdvantage;
    const awayScore = awayStrength;
    const strengthDiff = homeScore - awayScore;
    
    if (strengthDiff > 0.15) {
      homeProb = 0.48 + (strengthDiff * 0.3);
      drawProb = 0.24 - (strengthDiff * 0.1);
      awayProb = 0.28 - (strengthDiff * 0.2);
    } else if (strengthDiff < -0.05) {
      awayProb = 0.38 + (Math.abs(strengthDiff) * 0.25);
      drawProb = 0.28 - (Math.abs(strengthDiff) * 0.08);
      homeProb = 0.34 - (Math.abs(strengthDiff) * 0.17);
    } else {
      homeProb = 0.38 + (strengthDiff * 0.2);
      drawProb = 0.30;
      awayProb = 0.32 - (strengthDiff * 0.2);
    }
    
    homeProb = Math.max(0.15, Math.min(0.75, homeProb));
    drawProb = Math.max(0.15, Math.min(0.35, drawProb));
    awayProb = Math.max(0.12, Math.min(0.65, awayProb));
    
    const total = homeProb + drawProb + awayProb;
    homeProb /= total;
    drawProb /= total;
    awayProb /= total;
    
    const homeAttack = getTeamStrength(homeTeam + '_attack');
    const awayAttack = getTeamStrength(awayTeam + '_attack');
    const combinedAttack = (homeAttack + awayAttack) / 2;
    over25Prob = 0.40 + combinedAttack * 0.25;
    bttsProb = 0.35 + combinedAttack * 0.28;
    
    const maxProb = Math.max(homeProb, drawProb, awayProb);
    confidence = 0.55 + maxProb * 0.25;
    predictionSource = 'local_algorithm';
  }
  
  // Determine best combo predictions
  const combos = generateComboPredictions(homeProb, drawProb, awayProb, over25Prob, bttsProb);
  
  return {
    match_id: matchId,
    home_team: homeTeam,
    away_team: awayTeam,
    league: options.league || "Unknown",
    league_name: options.league_name || options.league || "Unknown",
    date: options.date || null,
    time: options.time || null,
    predictions: {
      result: {
        home: parseFloat(homeProb.toFixed(3)),
        draw: parseFloat(drawProb.toFixed(3)),
        away: parseFloat(awayProb.toFixed(3)),
        recommendation: homeProb > drawProb && homeProb > awayProb ? "Home Win" :
                       awayProb > homeProb && awayProb > drawProb ? "Away Win" : "Draw"
      },
      over_25: {
        yes: parseFloat(over25Prob.toFixed(3)),
        no: parseFloat((1 - over25Prob).toFixed(3)),
        recommendation: over25Prob > 0.5 ? "Over 2.5" : "Under 2.5"
      },
      btts: {
        yes: parseFloat(bttsProb.toFixed(3)),
        no: parseFloat((1 - bttsProb).toFixed(3)),
        recommendation: bttsProb > 0.5 ? "BTTS Yes" : "BTTS No"
      },
      combos: combos
    },
    odds: {
      home: options.home_odds || null,
      draw: options.draw_odds || null,
      away: options.away_odds || null
    },
    confidence: parseFloat(confidence.toFixed(2)),
    model_version: CONFIG.version,
    timestamp: timestamp
  };
}

function generateComboPredictions(homeProb, drawProb, awayProb, over25Prob, bttsProb) {
  const combos = [];
  
  // Result + Over 2.5
  if (homeProb > 0.4 && over25Prob > 0.5) {
    combos.push({
      type: '1X2 & Over/Under',
      pick: 'Home + O2.5',
      probability: (homeProb * over25Prob).toFixed(3),
      odds: (1 / (homeProb * over25Prob)).toFixed(2)
    });
  }
  
  // Result + BTTS
  if (homeProb > 0.35 && bttsProb > 0.5) {
    combos.push({
      type: '1X2 & GG/NG',
      pick: 'Home + GG',
      probability: (homeProb * bttsProb).toFixed(3),
      odds: (1 / (homeProb * bttsProb)).toFixed(2)
    });
  }
  
  // Over + BTTS
  if (over25Prob > 0.52 && bttsProb > 0.5) {
    combos.push({
      type: 'Over/Under & GG/NG',
      pick: 'O2.5 + GG',
      probability: (over25Prob * bttsProb).toFixed(3),
      odds: (1 / (over25Prob * bttsProb)).toFixed(2)
    });
  }
  
  // Double Chance + Over
  const dcProb = homeProb + drawProb;
  if (dcProb > 0.6 && over25Prob > 0.5) {
    combos.push({
      type: 'DC & Over/Under',
      pick: '1X + O1.5',
      probability: (dcProb * 0.75).toFixed(3),
      odds: (1 / (dcProb * 0.75)).toFixed(2)
    });
  }
  
  return combos;
}

// ============= Fixtures Handler =============
async function handleFixtures(request, specificLeague = null, env = {}) {
  const url = new URL(request.url);
  const days = parseInt(url.searchParams.get('days') || '14');
  const includePredictions = url.searchParams.get('predictions') !== 'false';
  const includeRecent = url.searchParams.get('recent') === 'true';
  const includeToday = url.searchParams.get('today') === 'true'; // Disabled by default - eventsday.php returns stale data
  const minConfidence = parseFloat(url.searchParams.get('minConfidence') || '0'); // Filter by min confidence (0-1)
  const highConfidenceOnly = url.searchParams.get('highConfidence') === 'true'; // Shortcut for >0.65
  const effectiveMinConfidence = highConfidenceOnly ? 0.65 : minConfidence;
  
  const leagues = specificLeague 
    ? { [specificLeague]: CONFIG.leagues[specificLeague] }
    : CONFIG.leagues;
  
  if (specificLeague && !CONFIG.leagues[specificLeague]) {
    return new Response(JSON.stringify({
      error: "League not found",
      available_leagues: Object.keys(CONFIG.leagues),
      timestamp: new Date().toISOString()
    }), { status: 404, headers: corsHeaders });
  }
  
  const allFixtures = [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const cutoff = new Date(today);
  cutoff.setDate(cutoff.getDate() + days);
  
  // PRIMARY SOURCE: The Odds API (fixtures + live odds for major leagues)
  const theOddsApiKey = env.THE_ODDS_API_KEY;
  if (theOddsApiKey) {
    try {
      // Fetch from major leagues
      const majorLeagues = [
        'soccer_epl', 'soccer_spain_la_liga', 'soccer_germany_bundesliga',
        'soccer_italy_serie_a', 'soccer_france_ligue_one', 'soccer_efl_champ'
      ];
      
      const oddsFixtures = await fetchTheOddsApiFixtures(theOddsApiKey, majorLeagues);
      
      for (const fixture of oddsFixtures) {
        if (fixture.date) {
          const todayStr = today.toISOString().split('T')[0];
          const cutoffStr = cutoff.toISOString().split('T')[0];
          if (fixture.date >= todayStr && fixture.date <= cutoffStr) {
            // Add local predictions since The Odds API doesn't provide them
            if (includePredictions) {
              const prediction = predictMatch(fixture.home_team, fixture.away_team, {
                league: fixture.league_name,
                date: fixture.date,
                odds: fixture.odds  // Use odds for better predictions
              });
              fixture.prediction = prediction.predictions;
              fixture.confidence = prediction.confidence;
              fixture.prediction_source = 'local';
            }
            allFixtures.push(fixture);
          }
        }
      }
    } catch (error) {
      console.log('The Odds API error:', error.message);
    }
  }
  
  // SECONDARY SOURCE: API-Football (1200+ fixtures per day) - NOTE: May require subscription
  try {
    const apiFootballFixtures = await fetchAPIFootballFixtures(Math.min(days, 7));
    
    // Filter to major leagues if specific league requested, otherwise keep all
    for (const match of apiFootballFixtures) {
      // Filter by specific league if requested
      if (specificLeague) {
        const leagueId = API_FOOTBALL_LEAGUES[specificLeague];
        if (leagueId && match.league_id !== leagueId) continue;
      }
      
      // Only include scheduled or live matches
      if (match.status !== 'scheduled' && match.status !== 'live') continue;
      
      // Add predictions
      if (includePredictions) {
        // Try ML API for first 20 matches (to respect rate limits)
        let mlPrediction = null;
        if (allFixtures.length < 20) {
          mlPrediction = await fetchMLPrediction(match.home_team, match.away_team, match.league_name || '');
        }
        
        const prediction = predictMatch(match.home_team, match.away_team, {
          league: match.league,
          league_name: match.league_name,
          date: match.date,
          time: match.time,
          mlPrediction: mlPrediction // Will use ML if available
        });
        match.prediction = prediction.predictions;
        match.confidence = prediction.confidence;
        match.prediction_source = prediction.prediction_source || (mlPrediction ? 'ml_model' : 'local');
      }
      
      allFixtures.push(match);
    }
    
    console.log(`API-Football returned ${apiFootballFixtures.length} total, ${allFixtures.length} after filtering`);
  } catch (error) {
    console.error('API-Football failed, falling back to TheSportsDB:', error);
    
    // Fallback to TheSportsDB if API-Football fails
    const fetchPromises = Object.entries(leagues).map(async ([leagueId, leagueInfo]) => {
      const sportsDbId = SPORTSDB_LEAGUES[leagueId];
      const fixtures = [];
      
      if (sportsDbId) {
        try {
          const matches = await fetchTheSportsDBFixtures(sportsDbId);
          
          for (const match of matches) {
            if (!match.date) continue;
            
            const todayStr = today.toISOString().split('T')[0];
            const cutoffStr = cutoff.toISOString().split('T')[0];
            const matchDateStr = match.date;
            
            if (match.status === 'scheduled' && matchDateStr >= todayStr && matchDateStr <= cutoffStr) {
              const actualLeagueName = match.league_from_api || leagueInfo.name;
              const actualCountry = match.country_from_api || leagueInfo.country;
              
              const fixture = {
                ...match,
                league: leagueId,
                league_name: actualLeagueName,
                country: actualCountry
              };
              
              if (includePredictions) {
                const prediction = predictMatch(match.home_team, match.away_team, {
                  league: leagueId,
                  league_name: actualLeagueName,
                  date: match.date,
                  time: match.time
                });
                fixture.prediction = prediction.predictions;
                fixture.confidence = prediction.confidence;
              }
              
              fixtures.push(fixture);
            }
          }
        } catch (error) {
          console.error(`TheSportsDB error for ${leagueId}:`, error);
        }
      }
      
      return fixtures;
    });
    
    const results = await Promise.all(fetchPromises);
    results.forEach(fixtures => allFixtures.push(...fixtures));
  }
  
  // Add recent results from Football-Data.co.uk if requested
  if (includeRecent) {
    for (const [leagueId, leagueInfo] of Object.entries(leagues)) {
      if (!leagueInfo.file) continue;
      try {
        const historicalMatches = await fetchFootballDataCSV(leagueInfo.file);
        const finished = historicalMatches
          .filter(m => m.status === 'finished')
          .slice(-10);
        
        for (const match of finished) {
          allFixtures.push({
            ...match,
            league: leagueId,
            league_name: leagueInfo.name,
            country: leagueInfo.country
          });
        }
      } catch (error) {
        console.error(`Historical data error for ${leagueId}:`, error);
      }
    }
  }
  
  // Game Forecast enrichment already done in primary source above
  
  // Deduplicate fixtures (TheSportsDB API sometimes returns same match for multiple league IDs)
  const seenMatches = new Map();
  const deduplicatedFixtures = [];
  for (const fixture of allFixtures) {
    const key = `${fixture.date}_${fixture.home_team}_${fixture.away_team}`.toLowerCase();
    if (!seenMatches.has(key)) {
      seenMatches.set(key, true);
      // Use the league from the API response itself, not our mapping
      if (fixture.source === 'thesportsdb' && fixture.league_name) {
        // Keep original league_name from API
      }
      deduplicatedFixtures.push(fixture);
    }
  }
  
  // Apply confidence filter if specified (for high win probability predictions)
  let filteredFixtures = deduplicatedFixtures;
  if (effectiveMinConfidence > 0) {
    filteredFixtures = deduplicatedFixtures.filter(f => {
      // Only filter scheduled matches with predictions
      if (f.status === 'finished') return true; // Keep finished matches
      if (!f.confidence) return false; // Skip if no confidence
      return f.confidence >= effectiveMinConfidence;
    });
  }
  
  // Enrich major league fixtures with Sportradar data (manager, standings, form)
  const sportradarApiKey = env.SPORTRADAR_API_KEY;
  if (sportradarApiKey) {
    const majorLeagues = ['premier', 'la liga', 'bundesliga', 'serie a', 'ligue 1'];
    const enrichPromises = filteredFixtures.slice(0, 30).map(async (fixture, idx) => {
      const leagueLower = (fixture.league_name || '').toLowerCase();
      if (majorLeagues.some(l => leagueLower.includes(l))) {
        try {
          return await enrichWithSportradar(fixture, sportradarApiKey);
        } catch (e) {
          return fixture;
        }
      }
      return fixture;
    });
    
    try {
      const enrichedMajor = await Promise.all(enrichPromises);
      // Replace first 30 with enriched versions
      for (let i = 0; i < enrichedMajor.length; i++) {
        filteredFixtures[i] = enrichedMajor[i];
      }
    } catch (e) {
      console.log('Sportradar enrichment error:', e.message);
    }
  }
  
  // Enrich fixtures with live odds from Odds-API.io
  const oddsApiKey = env.ODDS_API_KEY;
  if (oddsApiKey) {
    // Find ALL major league fixtures (not just first 30) for odds enrichment
    const majorLeagueFixtures = [];
    const majorLeagueIndices = [];
    
    for (let i = 0; i < filteredFixtures.length && majorLeagueFixtures.length < 50; i++) {
      const fixture = filteredFixtures[i];
      const leagueLower = (fixture.league_name || '').toLowerCase();
      const country = (fixture.country || '').toLowerCase();
      
      // Check if it's a major league (EPL, La Liga, Bundesliga, Serie A, Ligue 1, UCL)
      const isEPL = leagueLower === 'premier league' && country === 'england';
      const isLaLiga = (leagueLower.includes('la liga') || leagueLower === 'laliga') && country === 'spain';
      const isBundesliga = leagueLower === 'bundesliga' && country === 'germany';
      const isSerieA = leagueLower === 'serie a' && country === 'italy';
      const isLigue1 = leagueLower === 'ligue 1' && country === 'france';
      const isUCL = leagueLower.includes('champions league');
      
      if (isEPL || isLaLiga || isBundesliga || isSerieA || isLigue1 || isUCL) {
        majorLeagueFixtures.push(fixture);
        majorLeagueIndices.push(i);
      }
    }
    
    // Enrich major league fixtures with odds
    if (majorLeagueFixtures.length > 0) {
      const oddsPromises = majorLeagueFixtures.map(async (fixture) => {
        try {
          return await enrichWithOdds(fixture, oddsApiKey);
        } catch (e) {
          return fixture;
        }
      });
      
      try {
        const enrichedWithOdds = await Promise.all(oddsPromises);
        // Replace fixtures at their original positions
        for (let j = 0; j < enrichedWithOdds.length; j++) {
          filteredFixtures[majorLeagueIndices[j]] = enrichedWithOdds[j];
        }
      } catch (e) {
        console.log('Odds enrichment error:', e.message);
      }
    }
  }
  
  // Sort: live first, then scheduled by date, then finished
  filteredFixtures.sort((a, b) => {
    // Live matches at top
    if (a.status === 'live' && b.status !== 'live') return -1;
    if (a.status !== 'live' && b.status === 'live') return 1;
    // Scheduled next
    if (a.status === 'scheduled' && b.status === 'finished') return -1;
    if (a.status === 'finished' && b.status === 'scheduled') return 1;
    // Then by date/time
    const dateComp = a.date.localeCompare(b.date);
    if (dateComp !== 0) return dateComp;
    return (a.time || '00:00').localeCompare(b.time || '00:00');
  });
  
  return new Response(JSON.stringify({
    fixtures: filteredFixtures,
    count: filteredFixtures.length,
    live: filteredFixtures.filter(f => f.status === 'live').length,
    scheduled: filteredFixtures.filter(f => f.status === 'scheduled').length,
    finished: filteredFixtures.filter(f => f.status === 'finished').length,
    high_confidence: filteredFixtures.filter(f => f.confidence >= 0.65).length,
    leagues: Object.keys(leagues).length,
    date_range: { from: today.toISOString(), to: cutoff.toISOString() },
    filters: { minConfidence: effectiveMinConfidence, highConfidenceOnly },
    data_source: "TheSportsDB (Live + Upcoming)",
    timestamp: new Date().toISOString()
  }), { status: 200, headers: corsHeaders });
}

// ============= Other Handlers =============
async function handlePredict(request) {
  try {
    const body = await request.json();
    
    if (!body.home_team || !body.away_team) {
      return new Response(JSON.stringify({
        error: "Missing required fields",
        detail: "home_team and away_team are required",
        timestamp: new Date().toISOString()
      }), { status: 400, headers: corsHeaders });
    }
    
    const prediction = predictMatch(
      body.home_team,
      body.away_team,
      { 
        league: body.league, 
        match_date: body.match_date,
        home_odds: body.home_odds,
        draw_odds: body.draw_odds,
        away_odds: body.away_odds
      }
    );
    
    return new Response(JSON.stringify(prediction), {
      status: 200,
      headers: corsHeaders
    });
    
  } catch (error) {
    return new Response(JSON.stringify({
      error: "Invalid request",
      detail: error.message,
      timestamp: new Date().toISOString()
    }), { status: 400, headers: corsHeaders });
  }
}

function handleHealth() {
  return new Response(JSON.stringify({
    status: "healthy",
    version: CONFIG.version,
    markets: CONFIG.markets,
    leagues: Object.keys(CONFIG.leagues).length,
    timestamp: new Date().toISOString()
  }), { status: 200, headers: corsHeaders });
}

function handleLeagues() {
  return new Response(JSON.stringify({
    leagues: CONFIG.leagues,
    count: Object.keys(CONFIG.leagues).length,
    timestamp: new Date().toISOString()
  }), { status: 200, headers: corsHeaders });
}

function handleModelsInfo() {
  return new Response(JSON.stringify({
    version: CONFIG.version,
    markets: CONFIG.markets,
    leagues: Object.keys(CONFIG.leagues).length,
    data_source: "API-Football",
    prediction_features: ["odds-based", "home-advantage", "combo-generation", "acca-builder"],
    timestamp: new Date().toISOString()
  }), { status: 200, headers: corsHeaders });
}

// ============= ACCA Generation Handler =============
async function handleAccas(request) {
  const url = new URL(request.url);
  const folds = parseInt(url.searchParams.get('folds') || '3');
  const minConfidence = parseFloat(url.searchParams.get('confidence') || '65') / 100;
  const maxAccas = parseInt(url.searchParams.get('limit') || '5');
  const market = url.searchParams.get('market') || 'result'; // result, over25, btts
  
  try {
    // Fetch fixtures with predictions
    const fixtures = await fetchAPIFootballFixtures(3); // Next 3 days for ACCAs
    
    // Add predictions and filter by confidence
    const highConfMatches = [];
    for (const match of fixtures) {
      if (match.status !== 'scheduled') continue;
      
      const prediction = predictMatch(match.home_team, match.away_team, {
        league: match.league,
        league_name: match.league_name,
        date: match.date,
        time: match.time
      });
      
      if (prediction.confidence >= minConfidence) {
        highConfMatches.push({
          ...match,
          prediction: prediction.predictions,
          confidence: prediction.confidence
        });
      }
    }
    
    // Sort by confidence descending
    highConfMatches.sort((a, b) => b.confidence - a.confidence);
    
    // Generate ACCAs
    const accas = [];
    const usedMatches = new Set();
    
    // Generate multiple ACCAs with different combinations
    for (let accaNum = 0; accaNum < maxAccas && highConfMatches.length >= folds; accaNum++) {
      const selections = [];
      let combinedOdds = 1;
      let avgConfidence = 0;
      
      // Select matches for this ACCA
      for (const match of highConfMatches) {
        const matchKey = `${match.date}_${match.home_team}_${match.away_team}`;
        if (usedMatches.has(matchKey)) continue;
        
        let pick, odds, prob;
        
        if (market === 'result') {
          const result = match.prediction.result;
          if (result.home > result.away && result.home > result.draw) {
            pick = 'Home Win';
            prob = result.home;
          } else if (result.away > result.home && result.away > result.draw) {
            pick = 'Away Win';
            prob = result.away;
          } else {
            pick = 'Draw';
            prob = result.draw;
          }
          odds = Math.max(1.1, 1 / prob);
        } else if (market === 'over25') {
          const over = match.prediction.over_25;
          pick = over.yes > 0.5 ? 'Over 2.5 Goals' : 'Under 2.5 Goals';
          prob = over.yes > 0.5 ? over.yes : over.no;
          odds = Math.max(1.1, 1 / prob);
        } else if (market === 'btts') {
          const btts = match.prediction.btts;
          pick = btts.yes > 0.5 ? 'BTTS Yes' : 'BTTS No';
          prob = btts.yes > 0.5 ? btts.yes : btts.no;
          odds = Math.max(1.1, 1 / prob);
        }
        
        selections.push({
          match: `${match.home_team} vs ${match.away_team}`,
          home_team: match.home_team,
          away_team: match.away_team,
          league: match.league_name,
          date: match.date,
          time: match.time,
          pick,
          odds: parseFloat(odds.toFixed(2)),
          probability: parseFloat(prob.toFixed(3)),
          confidence: match.confidence,
          home_logo: match.home_logo,
          away_logo: match.away_logo
        });
        
        combinedOdds *= odds;
        avgConfidence += match.confidence;
        usedMatches.add(matchKey);
        
        if (selections.length >= folds) break;
      }
      
      if (selections.length === folds) {
        avgConfidence /= folds;
        
        accas.push({
          id: accaNum + 1,
          type: `${folds}-fold`,
          market,
          selections,
          combined_odds: parseFloat(combinedOdds.toFixed(2)),
          confidence: parseFloat((avgConfidence * 100).toFixed(1)),
          potential_return_10: parseFloat((10 * combinedOdds).toFixed(2)),
          potential_return_50: parseFloat((50 * combinedOdds).toFixed(2)),
          risk_level: avgConfidence >= 0.7 ? 'Low' : avgConfidence >= 0.6 ? 'Medium' : 'High'
        });
      }
    }
    
    return new Response(JSON.stringify({
      accas,
      count: accas.length,
      folds,
      market,
      min_confidence: minConfidence,
      high_conf_matches: highConfMatches.length,
      total_matches_scanned: fixtures.length,
      timestamp: new Date().toISOString()
    }), { status: 200, headers: corsHeaders });
    
  } catch (error) {
    console.error('ACCA generation error:', error);
    return new Response(JSON.stringify({
      error: "ACCA generation failed",
      detail: error.message,
      timestamp: new Date().toISOString()
    }), { status: 500, headers: corsHeaders });
  }
}

function handleNotFound() {
  return new Response(JSON.stringify({
    error: "Not Found",
    available_endpoints: [
      "GET /fixtures - Get upcoming fixtures with predictions",
      "GET /fixtures?days=14 - Fixtures for next 14 days",
      "GET /fixtures/:league - League-specific fixtures",
      "GET /leagues - List all leagues",
      "POST /predict - Get match prediction",
      "GET /health - Health check",
      "GET /models/info - Model information"
    ],
    timestamp: new Date().toISOString()
  }), { status: 404, headers: corsHeaders });
}

// ============= Retraining Handler =============
async function handleRetrain(request, env) {
  const url = new URL(request.url);
  const secret = url.searchParams.get('secret');
  
  // Verify secret (set via Cloudflare Worker secret)
  const expectedSecret = env.RETRAIN_SECRET || 'footypredict-retrain-2024';
  if (secret !== expectedSecret) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Invalid secret',
      timestamp: new Date().toISOString()
    }), { status: 401, headers: corsHeaders });
  }
  
  const startTime = Date.now();
  const results = {
    triggered: false,
    kaggle_notebook: null,
    huggingface_sync: null,
    cache_cleared: false,
    errors: []
  };
  
  try {
    // Step 1: Clear ML prediction cache to force fresh predictions
    mlPredictionCache.clear();
    results.cache_cleared = true;
    
    // Step 2: Trigger Kaggle notebook via API (if credentials available)
    const kaggleUsername = env.KAGGLE_USERNAME;
    const kaggleKey = env.KAGGLE_KEY;
    
    if (kaggleUsername && kaggleKey) {
      try {
        const kaggleAuth = btoa(`${kaggleUsername}:${kaggleKey}`);
        const kagglePush = await fetch(
          `https://www.kaggle.com/api/v1/kernels/push`,
          {
            method: 'POST',
            headers: {
              'Authorization': `Basic ${kaggleAuth}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              id: `${kaggleUsername}/footypredict-training`,
              kernel_type: 'notebook',
              is_private: true,
              enable_gpu: true,
              run_as: 'draft'
            }),
            signal: AbortSignal.timeout(10000)
          }
        );
        
        if (kagglePush.ok) {
          results.kaggle_notebook = {
            status: 'triggered',
            kernel: `${kaggleUsername}/footypredict-training`,
            timestamp: new Date().toISOString()
          };
          results.triggered = true;
        } else {
          const errorText = await kagglePush.text();
          results.kaggle_notebook = {
            status: 'failed',
            error: errorText.substring(0, 200)
          };
          results.errors.push(`Kaggle API error: ${errorText.substring(0, 100)}`);
        }
      } catch (kaggleError) {
        results.kaggle_notebook = {
          status: 'error',
          error: kaggleError.message
        };
        results.errors.push(`Kaggle error: ${kaggleError.message}`);
      }
    } else {
      results.kaggle_notebook = {
        status: 'skipped',
        reason: 'Kaggle credentials not configured'
      };
    }
    
    // Step 3: Notify HuggingFace Space to refresh models (optional)
    try {
      const hfResponse = await fetch(`${HF_API_URL}/api/health`, {
        method: 'GET',
        headers: { 'User-Agent': 'FootyPredict-Retrain/1.0' },
        signal: AbortSignal.timeout(5000)
      });
      
      if (hfResponse.ok) {
        results.huggingface_sync = {
          status: 'healthy',
          message: 'HuggingFace Space is running and will pick up new models'
        };
      }
    } catch (hfError) {
      results.huggingface_sync = {
        status: 'unreachable',
        error: hfError.message
      };
    }
    
    const duration = Date.now() - startTime;
    
    return new Response(JSON.stringify({
      success: true,
      message: 'Retraining pipeline initiated',
      results,
      duration_ms: duration,
      next_steps: [
        'Kaggle notebook will retrain models (if triggered)',
        'Models will be uploaded to HuggingFace',
        'Worker will automatically use new predictions'
      ],
      timestamp: new Date().toISOString()
    }), { status: 200, headers: corsHeaders });
    
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message,
      results,
      timestamp: new Date().toISOString()
    }), { status: 500, headers: corsHeaders });
  }
}

// ============= SportyBet Booking Code Integration =============
const SPORTYBET_API_BASE = 'https://www.sportybet.com/api';

// Cache for SportyBet match mappings
const sportyBetMatchCache = new Map();
const SPORTYBET_CACHE_TTL = 30 * 60 * 1000; // 30 minutes

// Market ID mappings for SportyBet
const SPORTYBET_MARKETS = {
  '1x2': { marketId: '1', outcomes: { home: '1', draw: '2', away: '3' } },
  'over_under_1.5': { marketId: '18', specifier: 'total=1.5', outcomes: { over: '12', under: '13' } },
  'over_under_2.5': { marketId: '18', specifier: 'total=2.5', outcomes: { over: '12', under: '13' } },
  'over_under_3.5': { marketId: '18', specifier: 'total=3.5', outcomes: { over: '12', under: '13' } },
  'btts': { marketId: '29', outcomes: { yes: '74', no: '76' } },
  'double_chance': { marketId: '10', outcomes: { '1x': '9', '12': '10', 'x2': '11' } },
  'draw_no_bet': { marketId: '11', outcomes: { home: '1714', away: '1715' } },
};

// Fetch SportyBet match list for a given date range
async function fetchSportyBetMatches(country = 'gh') {
  const cacheKey = `sportybet_matches_${country}`;
  const cached = sportyBetMatchCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < SPORTYBET_CACHE_TTL) {
    return cached.data;
  }
  
  const matches = [];
  const headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json'
  };
  
  try {
    // 1. Fetch popular matches
    const popularUrl = `${SPORTYBET_API_BASE}/${country}/factsCenter/popularAndSportList?sportId=sr:sport:1`;
    const popularResponse = await fetch(popularUrl, {
      headers,
      signal: AbortSignal.timeout(10000)
    });
    
    if (popularResponse.ok) {
      const data = await popularResponse.json();
      if (data.data?.eventList) {
        for (const event of data.data.eventList) {
          if (event.eventId && event.homeTeamName) {
            matches.push({
              eventId: event.eventId,
              homeTeam: event.homeTeamName,
              awayTeam: event.awayTeamName,
              league: event.leagueName,
              startTime: event.estimateStartTime
            });
          }
        }
      }
    }
    
    // 2. Fetch scheduled matches for today and next 2 days
    const now = new Date();
    for (let dayOffset = 0; dayOffset < 3; dayOffset++) {
      const date = new Date(now);
      date.setDate(date.getDate() + dayOffset);
      const dateStr = date.toISOString().split('T')[0].replace(/-/g, '');
      
      try {
        const scheduleUrl = `${SPORTYBET_API_BASE}/${country}/factsCenter/pcEvents?sportId=sr:sport:1&marketId=1&date=${dateStr}&pageSize=100`;
        const scheduleResponse = await fetch(scheduleUrl, {
          headers,
          signal: AbortSignal.timeout(8000)
        });
        
        if (scheduleResponse.ok) {
          const scheduleData = await scheduleResponse.json();
          if (scheduleData.data?.eventList) {
            for (const event of scheduleData.data.eventList) {
              if (event.eventId && event.homeTeamName) {
                // Avoid duplicates
                if (!matches.some(m => m.eventId === event.eventId)) {
                  matches.push({
                    eventId: event.eventId,
                    homeTeam: event.homeTeamName,
                    awayTeam: event.awayTeamName,
                    league: event.leagueName,
                    startTime: event.estimateStartTime
                  });
                }
              }
            }
          }
        }
      } catch (err) {
        console.log(`SportyBet schedule fetch error for ${dateStr}:`, err.message);
      }
    }
    
    // 3. Also try fetching from league-specific endpoints for major leagues
    const majorLeagueSportIds = [
      'sr:tournament:8', // Premier League
      'sr:tournament:35', // Bundesliga
      'sr:tournament:87', // La Liga
      'sr:tournament:23', // Serie A
      'sr:tournament:34', // Ligue 1
    ];
    
    for (const tournamentId of majorLeagueSportIds) {
      try {
        const leagueUrl = `${SPORTYBET_API_BASE}/${country}/factsCenter/pcEvents?sportId=sr:sport:1&tournamentId=${tournamentId}&marketId=1&pageSize=50`;
        const leagueResponse = await fetch(leagueUrl, {
          headers,
          signal: AbortSignal.timeout(5000)
        });
        
        if (leagueResponse.ok) {
          const leagueData = await leagueResponse.json();
          if (leagueData.data?.eventList) {
            for (const event of leagueData.data.eventList) {
              if (event.eventId && event.homeTeamName) {
                if (!matches.some(m => m.eventId === event.eventId)) {
                  matches.push({
                    eventId: event.eventId,
                    homeTeam: event.homeTeamName,
                    awayTeam: event.awayTeamName,
                    league: event.leagueName,
                    startTime: event.estimateStartTime
                  });
                }
              }
            }
          }
        }
      } catch (err) {
        // Silently continue on league fetch errors
      }
    }
    
    sportyBetMatchCache.set(cacheKey, { data: matches, timestamp: Date.now() });
    console.log(`SportyBet: fetched ${matches.length} matches for ${country}`);
    return matches;
  } catch (error) {
    console.log('SportyBet match fetch error:', error.message);
    return [];
  }
}

// Team name aliases for matching (covering different naming conventions)
const TEAM_ALIASES = {
  // Premier League
  'manchester united': ['man united', 'man utd', 'united', 'mufc'],
  'manchester city': ['man city', 'city', 'mcfc'],
  'chelsea': ['chelsea fc'],
  'liverpool': ['liverpool fc', 'lfc'],
  'arsenal': ['arsenal fc', 'the gunners'],
  'tottenham': ['tottenham hotspur', 'spurs', 'thfc'],
  'west ham': ['west ham united', 'hammers'],
  'newcastle': ['newcastle united', 'newcastle utd', 'nufc'],
  'aston villa': ['villa'],
  'brighton': ['brighton & hove albion', 'brighton hove albion'],
  'everton': ['everton fc'],
  'wolverhampton': ['wolves', 'wolverhampton wanderers'],
  'crystal palace': ['palace'],
  'nottingham forest': ["nott'm forest", 'forest'],
  
  // Bundesliga
  '1. fc köln': ['fc cologne', 'cologne', 'köln', 'koln', 'fc koln'],
  'vfl wolfsburg': ['wolfsburg'],
  'rb leipzig': ['leipzig', 'rbl'],
  'fsv mainz 05': ['mainz', 'mainz 05'],
  'bayern munich': ['bayern münchen', 'bayern', 'fcb'],
  'borussia dortmund': ['dortmund', 'bvb'],
  'bayer leverkusen': ['leverkusen', 'bayer 04'],
  'eintracht frankfurt': ['frankfurt', 'sge'],
  'borussia mönchengladbach': ['gladbach', "m'gladbach", 'mönchengladbach', 'monchengladbach'],
  'sc freiburg': ['freiburg'],
  'vfb stuttgart': ['stuttgart'],
  'union berlin': ['fc union berlin', 'berlin'],
  
  // La Liga
  'athletic bilbao': ['athletic club', 'bilbao', 'athletic'],
  'real sociedad': ['sociedad', 'la real'],
  'real madrid': ['madrid', 'real'],
  'barcelona': ['fc barcelona', 'barca'],
  'atletico madrid': ['atlético madrid', 'atletico', 'atleti'],
  'sevilla': ['sevilla fc'],
  'real betis': ['betis'],
  'villarreal': ['villarreal cf', 'yellow submarine'],
  'valencia': ['valencia cf'],
  'celta vigo': ['celta', 'rc celta'],
  
  // Serie A
  'inter milan': ['inter', 'internazionale', 'inter milano'],
  'ac milan': ['milan', 'rossoneri'],
  'juventus': ['juve'],
  'napoli': ['ssc napoli'],
  'roma': ['as roma'],
  'lazio': ['ss lazio'],
  'atalanta': ['atalanta bc', 'atalanta bergamo'],
  'fiorentina': ['acf fiorentina'],
  
  // Ligue 1
  'paris saint-germain': ['psg', 'paris', 'paris sg'],
  'olympique marseille': ['marseille', 'om'],
  'olympique lyon': ['lyon', 'ol'],
  'as monaco': ['monaco'],
  'lille': ['losc', 'losc lille'],
};

// Find SportyBet event ID for a match
function findSportyBetEventId(matches, homeTeam, awayTeam) {
  // Enhanced normalization
  const normalizeTeam = (name) => {
    return name.toLowerCase()
      .normalize('NFD').replace(/[\u0300-\u036f]/g, '') // Remove accents
      .replace(/\s+fc$/i, '')
      .replace(/^fc\s+/i, '')
      .replace(/[^a-z0-9\s]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
  };
  
  // Get all possible names for a team
  const getTeamVariants = (name) => {
    const normalized = normalizeTeam(name);
    const variants = [normalized];
    
    // Check aliases
    for (const [key, aliases] of Object.entries(TEAM_ALIASES)) {
      const normalizedKey = normalizeTeam(key);
      if (normalized.includes(normalizedKey) || normalizedKey.includes(normalized)) {
        variants.push(...aliases.map(a => normalizeTeam(a)));
      }
      for (const alias of aliases) {
        const normalizedAlias = normalizeTeam(alias);
        if (normalized.includes(normalizedAlias) || normalizedAlias.includes(normalized)) {
          variants.push(normalizedKey);
          variants.push(...aliases.map(a => normalizeTeam(a)));
        }
      }
    }
    
    return [...new Set(variants)]; // Remove duplicates
  };
  
  const homeVariants = getTeamVariants(homeTeam);
  const awayVariants = getTeamVariants(awayTeam);
  
  for (const match of matches) {
    const matchHome = normalizeTeam(match.homeTeam || '');
    const matchAway = normalizeTeam(match.awayTeam || '');
    
    // Check if any variant matches
    const homeMatch = homeVariants.some(v => 
      matchHome.includes(v) || v.includes(matchHome) ||
      matchHome.split(' ').some(word => v.includes(word) && word.length > 3)
    );
    const awayMatch = awayVariants.some(v => 
      matchAway.includes(v) || v.includes(matchAway) ||
      matchAway.split(' ').some(word => v.includes(word) && word.length > 3)
    );
    
    if (homeMatch && awayMatch) {
      return match.eventId;
    }
  }
  
  return null;
}

// Generate real SportyBet booking code
async function generateSportyBetBookingCode(selections, country = 'gh') {
  try {
    // First, fetch current matches from SportyBet
    const sportyMatches = await fetchSportyBetMatches(country);
    
    // Build selections array with SportyBet event IDs
    const sportySelections = [];
    const matchedSelections = [];
    const unmatchedSelections = [];
    
    for (const sel of selections) {
      const eventId = findSportyBetEventId(sportyMatches, sel.homeTeam, sel.awayTeam);
      
      if (eventId) {
        // Get market config
        const marketConfig = SPORTYBET_MARKETS[sel.market] || SPORTYBET_MARKETS['1x2'];
        const outcomeId = marketConfig.outcomes[sel.selection.toLowerCase()] || '1';
        
        sportySelections.push({
          eventId: eventId,
          marketId: marketConfig.marketId,
          specifier: marketConfig.specifier || null,
          outcomeId: outcomeId
        });
        
        matchedSelections.push({
          ...sel,
          eventId: eventId,
          matched: true
        });
      } else {
        unmatchedSelections.push({
          ...sel,
          matched: false,
          reason: 'Match not found on SportyBet'
        });
      }
    }
    
    // If no matches found, return error with selections for manual entry
    if (sportySelections.length === 0) {
      return {
        success: false,
        error: 'No matches found on SportyBet',
        selections: selections,
        hint: 'These matches may not be available on SportyBet or use different team names'
      };
    }
    
    // Call SportyBet share API to generate booking code
    const shareUrl = `${SPORTYBET_API_BASE}/${country}/orders/share`;
    const shareResponse = await fetch(shareUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Origin': 'https://www.sportybet.com',
        'Referer': 'https://www.sportybet.com/'
      },
      body: JSON.stringify({ selections: sportySelections }),
      signal: AbortSignal.timeout(15000)
    });
    
    if (!shareResponse.ok) {
      const errorText = await shareResponse.text();
      return {
        success: false,
        error: `SportyBet API error: ${shareResponse.status}`,
        details: errorText,
        matchedSelections: matchedSelections,
        unmatchedSelections: unmatchedSelections
      };
    }
    
    const shareData = await shareResponse.json();
    
    // Extract booking code from response
    const bookingCode = shareData.data?.shareCode || shareData.shareCode || null;
    
    if (bookingCode) {
      return {
        success: true,
        bookingCode: bookingCode,
        country: country,
        totalSelections: sportySelections.length,
        matchedSelections: matchedSelections,
        unmatchedSelections: unmatchedSelections,
        loadUrl: `https://www.sportybet.com/${country}/sport/football?shareCode=${bookingCode}`
      };
    } else {
      return {
        success: false,
        error: 'No booking code in response',
        rawResponse: shareData,
        matchedSelections: matchedSelections
      };
    }
    
  } catch (error) {
    return {
      success: false,
      error: error.message,
      selections: selections
    };
  }
}

// Handle SportyBet booking code request
async function handleSportyBetBookingCode(request) {
  try {
    const body = await request.json();
    const { selections, country = 'gh' } = body;
    
    if (!selections || !Array.isArray(selections) || selections.length === 0) {
      return new Response(JSON.stringify({
        success: false,
        error: 'selections array is required'
      }), { status: 400, headers: corsHeaders });
    }
    
    // Validate selections format
    for (const sel of selections) {
      if (!sel.homeTeam || !sel.awayTeam) {
        return new Response(JSON.stringify({
          success: false,
          error: 'Each selection must have homeTeam and awayTeam'
        }), { status: 400, headers: corsHeaders });
      }
    }
    
    const result = await generateSportyBetBookingCode(selections, country);
    
    return new Response(JSON.stringify(result), { 
      status: result.success ? 200 : 400, 
      headers: corsHeaders 
    });
    
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: 'Invalid request body',
      details: error.message
    }), { status: 400, headers: corsHeaders });
  }
}

// ============= Main Router =============
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;
    
    // Handle CORS preflight
    if (method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }
    
    // Fixtures endpoints
    if (method === "GET" && path === "/fixtures") {
      return handleFixtures(request, null, env);
    }
    
    if (method === "GET" && path.startsWith("/fixtures/")) {
      const league = path.split("/")[2];
      return handleFixtures(request, league, env);
    }
    
    if (method === "GET" && path === "/leagues") {
      return handleLeagues();
    }
    
    // Prediction endpoint
    if (method === "POST" && path === "/predict") {
      return handlePredict(request);
    }
    
    // ACCA generation endpoint
    if (method === "GET" && path === "/accas") {
      return handleAccas(request);
    }
    
    // Retraining endpoint (called by cron-job.org)
    if (method === "GET" && path === "/retrain") {
      return handleRetrain(request, env);
    }
    
    // SportyBet booking code generation
    if (method === "POST" && path === "/sportybet/booking-code") {
      return handleSportyBetBookingCode(request);
    }
    
    // Info endpoints
    if (method === "GET" && path === "/health") {
      return handleHealth();
    }
    
    if (method === "GET" && path === "/models/info") {
      return handleModelsInfo();
    }
    
    if (method === "GET" && path === "/") {
      return new Response(JSON.stringify({
        name: "FootyPredict Pro API",
        version: CONFIG.version,
        description: "Football prediction API with real fixtures from 22 leagues",
        endpoints: {
          fixtures: "GET /fixtures",
          leagues: "GET /leagues",
          predict: "POST /predict",
          sportybet: "POST /sportybet/booking-code",
          debug: "GET /debug/game-forecast (test Game Forecast API)"
        },
        timestamp: new Date().toISOString()
      }), { status: 200, headers: corsHeaders });
    }
    
    // Debug endpoint for Game Forecast API
    if (method === "GET" && path === "/debug/game-forecast") {
      const apiKey = env.RAPIDAPI_KEY;
      
      try {
        const url = `${GAME_FORECAST_BASE}/events?status_code=NOT_STARTED&page_size=10`;
        const response = await fetch(url, {
          headers: {
            'x-rapidapi-host': 'game-forecast-api.p.rapidapi.com',
            'x-rapidapi-key': apiKey
          }
        });
        
        const data = await response.json();
        const events = data.data || [];
        
        return new Response(JSON.stringify({
          rapidapi_key_exists: !!apiKey,
          rapidapi_key_length: apiKey ? apiKey.length : 0,
          api_response_ok: response.ok,
          api_status: response.status,
          events_count: events.length,
          sample_event: events.length > 0 ? {
            home: events[0].team_home?.name,
            away: events[0].team_away?.name,
            league: events[0].league?.name,
            has_predictions: !!events[0].predictions?.length,
            has_odds: !!events[0].odds?.length
          } : null
        }, null, 2), { status: 200, headers: corsHeaders });
      } catch (error) {
        return new Response(JSON.stringify({
          rapidapi_key_exists: !!apiKey,
          rapidapi_key_length: apiKey ? apiKey.length : 0,
          error: error.message
        }, null, 2), { status: 200, headers: corsHeaders });
      }
    }
    
    return handleNotFound();
  }
};

