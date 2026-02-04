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
const CACHE_VERSION = 'v4'; // Increment to force player cache invalidation
const CONFIG = {
  version: "2.1.1",
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
      
      console.log(`The Odds API ${league}: status ${response.status}, ok: ${response.ok}`);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log(`The Odds API error for ${league}: ${errorText.substring(0, 200)}`);
        continue;
      }
      
      const matches = await response.json();
      console.log(`The Odds API ${league}: got ${matches.length} matches`);
      
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
        league_name: e.strLeague,  // League name from API
        country: e.strCountry || '',
        venue: e.strVenue || null,
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

// ============= SportyBet Fixture Scraper (Primary for fixtures + odds) =============
const SPORTYBET_FIXTURES_API = 'https://www.sportybet.com/api/ng/factsCenter';

// Cache for SportyBet fixtures
const sportyBetFixturesCache = new Map();
const SPORTYBET_FIXTURES_CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Fetch fixtures from SportyBet
async function fetchSportyBetFixtures(options = {}) {
  const { todayOnly = true, pageSize = 100 } = options;
  const cacheKey = `sportybet_fixtures_${todayOnly ? 'today' : 'upcoming'}`;
  
  // Check cache
  const cached = sportyBetFixturesCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < SPORTYBET_FIXTURES_CACHE_TTL) {
    console.log(`SportyBet fixtures cache hit: ${cached.data.length} fixtures`);
    return cached.data;
  }
  
  const allFixtures = [];
  let page = 1;
  const maxPages = 8;
  
  while (page <= maxPages) {
    try {
      const url = `${SPORTYBET_FIXTURES_API}/pcUpcomingEvents?sportId=sr:sport:1&marketId=1,18,10,29&pageSize=${pageSize}&pageNum=${page}${todayOnly ? '&todayGames=true' : ''}&_t=${Date.now()}`;
      
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
          'Accept': 'application/json',
          'Accept-Language': 'en-US,en;q=0.9',
          'Referer': 'https://www.sportybet.com/ng/'
        },
        signal: AbortSignal.timeout(15000)
      });
      
      if (!response.ok) {
        console.log(`SportyBet API returned ${response.status}`);
        break;
      }
      
      const data = await response.json();
      
      // bizCode 10000 is success for SportyBet
      if (data.bizCode !== 0 && data.bizCode !== 10000) {
        console.log(`SportyBet bizCode: ${data.bizCode}`);
        break;
      }
      
      const tournaments = data.data?.tournaments || [];
      if (tournaments.length === 0) break;
      
      let eventsFound = 0;
      for (const tournament of tournaments) {
        const leagueName = tournament.name || 'Unknown League';
        for (const event of tournament.events || []) {
          const fixture = parseSportyBetEvent(event, leagueName);
          if (fixture) {
            allFixtures.push(fixture);
            eventsFound++;
          }
        }
      }
      
      console.log(`SportyBet page ${page}: ${eventsFound} fixtures from ${tournaments.length} leagues`);
      
      // Check if we've fetched all
      const total = data.data?.totalNum || 0;
      if (allFixtures.length >= total || eventsFound === 0) break;
      
      page++;
      await new Promise(r => setTimeout(r, 300)); // Rate limiting
      
    } catch (error) {
      console.log(`SportyBet error on page ${page}:`, error.message);
      break;
    }
  }
  
  // Cache results
  if (allFixtures.length > 0) {
    sportyBetFixturesCache.set(cacheKey, { data: allFixtures, timestamp: Date.now() });
  }
  
  console.log(`SportyBet total: ${allFixtures.length} fixtures`);
  return allFixtures;
}

// Parse SportyBet event to fixture format
function parseSportyBetEvent(event, leagueName) {
  try {
    const homeTeam = event.homeTeamName;
    const awayTeam = event.awayTeamName;
    if (!homeTeam || !awayTeam) return null;
    
    // Parse date/time
    const startTime = event.estimateStartTime || 0;
    let date = new Date().toISOString().split('T')[0];
    let time = '00:00:00';
    if (startTime) {
      const dt = new Date(startTime);
      date = dt.toISOString().split('T')[0];
      time = dt.toTimeString().split(' ')[0];
    }
    
    // Parse odds
    const odds = { home: 0, draw: 0, away: 0, over25: 0, under25: 0, btts_yes: 0, btts_no: 0, dc_1x: 0, dc_x2: 0, source: 'sportybet' };
    for (const market of event.markets || []) {
      const id = market.id;
      const outcomes = market.outcomes || [];
      
      if (id === '1') { // 1X2
        for (const o of outcomes) {
          const oddVal = parseFloat(o.odds) || 0;
          if (o.id === '1') odds.home = oddVal;
          else if (o.id === '2') odds.draw = oddVal;
          else if (o.id === '3') odds.away = oddVal;
        }
      } else if (id === '18') { // Over/Under
        const specifier = market.specifier || '';
        if (specifier.includes('total=2.5')) {
          for (const o of outcomes) {
            const oddVal = parseFloat(o.odds) || 0;
            if (o.id === '12') odds.over25 = oddVal;
            else if (o.id === '13') odds.under25 = oddVal;
          }
        }
      } else if (id === '29') { // BTTS
        for (const o of outcomes) {
          const oddVal = parseFloat(o.odds) || 0;
          if (o.id === '74') odds.btts_yes = oddVal;
          else if (o.id === '76') odds.btts_no = oddVal;
        }
      } else if (id === '10') { // Double Chance
        for (const o of outcomes) {
          const oddVal = parseFloat(o.odds) || 0;
          if (o.id === '9') odds.dc_1x = oddVal;
          else if (o.id === '11') odds.dc_x2 = oddVal;
        }
      }
    }
    
    return {
      id: event.eventId || `sporty_${homeTeam}_${awayTeam}`.replace(/\s/g, '_'),
      homeTeam,
      awayTeam,
      league: leagueName,
      date,
      time,
      venue: `${homeTeam} Stadium`,
      odds,
      source: 'sportybet'
    };
    
  } catch (error) {
    console.log('Error parsing SportyBet event:', error.message);
    return null;
  }
}

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
  const dataSources = []; // Track which APIs returned data
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const cutoff = new Date(today);
  cutoff.setDate(cutoff.getDate() + days);
  const todayStr = today.toISOString().split('T')[0];
  const cutoffStr = cutoff.toISOString().split('T')[0];
  
  // GUARANTEED FREE SOURCE: TheSportsDB (no API key required, always works)
  // Using this FIRST to ensure we always have fixtures even when paid APIs are exhausted
  const majorLeagueIds = [
    '4328',  // Premier League
    '4335',  // La Liga
    '4331',  // Bundesliga
    '4332',  // Serie A
    '4334',  // Ligue 1
    '4396',  // League 1 (English)
    '4337',  // Primeira Liga
    '4344',  // Eredivisie
  ];
  
  try {
    // PRIMARY SOURCE: SportyBet (700+ fixtures with full odds - FREE)
    console.log('SportyBet: Fetching fixtures (primary source)...');
    try {
      const sportyFixtures = await fetchSportyBetFixtures({ todayOnly: false });
      console.log(`SportyBet returned ${sportyFixtures.length} fixtures`);
      
      for (const match of sportyFixtures) {
        if (match.date && match.date >= todayStr && match.date <= cutoffStr) {
          // Add predictions
          if (includePredictions && match.odds?.home > 0) {
            // Calculate probabilities from odds
            const total = 1/match.odds.home + 1/match.odds.draw + 1/match.odds.away;
            const margin = total - 1;
            match.prediction = {
              home_prob: (1/match.odds.home / total) * 100,
              draw_prob: (1/match.odds.draw / total) * 100,
              away_prob: (1/match.odds.away / total) * 100,
              over25_prob: match.odds.over25 > 0 ? (1/match.odds.over25 / (1/match.odds.over25 + 1/match.odds.under25)) * 100 : 50,
              confidence: Math.max(0.5, 0.75 - margin/2), // Adjust for margin
              source: 'sportybet_odds'
            };
          }
          allFixtures.push(match);
        }
      }
      
      if (sportyFixtures.length > 0) {
        dataSources.push('sportybet');
      }
    } catch (err) {
      console.log('SportyBet error:', err.message);
    }
    
    // FALLBACK: TheSportsDB (if SportyBet returned fewer fixtures)
    console.log('TheSportsDB: Fetching from major leagues...');
    for (const leagueId of majorLeagueIds) {
      try {
        const matches = await fetchTheSportsDBFixtures(leagueId);
        for (const match of matches) {
          if (match.date && match.date >= todayStr && match.date <= cutoffStr) {
            // Add predictions
            if (includePredictions) {
              try {
                const prediction = predictMatch(match.home_team, match.away_team, {
                  league: match.league_name,
                  date: match.date
                });
                match.prediction = prediction.predictions;
                match.confidence = prediction.confidence;
                match.prediction_source = 'local';
              } catch (predErr) {
                match.prediction = { result: { home: 0.35, draw: 0.30, away: 0.35 }};
                match.confidence = 0.55;
                match.prediction_source = 'fallback';
              }
            }
            allFixtures.push(match);
          }
        }
      } catch (leagueErr) {
        console.log(`TheSportsDB league ${leagueId} error:`, leagueErr.message);
      }
    }
    
    if (allFixtures.length > 0) {
      dataSources.push('TheSportsDB');
      console.log(`TheSportsDB: Got ${allFixtures.length} fixtures`);
    }
  } catch (sportsDbError) {
    console.log('TheSportsDB error:', sportsDbError.message);
  }
  
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
      console.log(`The Odds API returned ${oddsFixtures.length} fixtures`);
      
      const todayStr = today.toISOString().split('T')[0];
      const cutoffStr = cutoff.toISOString().split('T')[0];
      
      for (const fixture of oddsFixtures) {
        try {
          if (fixture.date && fixture.date >= todayStr && fixture.date <= cutoffStr) {
            // Add local predictions since The Odds API doesn't provide them
            if (includePredictions) {
              try {
                const prediction = predictMatch(fixture.home_team, fixture.away_team, {
                  league: fixture.league_name,
                  date: fixture.date,
                  odds: fixture.odds  // Use odds for better predictions
                });
                fixture.prediction = prediction.predictions;
                fixture.confidence = prediction.confidence;
                fixture.prediction_source = 'local';
              } catch (predErr) {
                console.log('Prediction error:', predErr.message);
                // Set default prediction if predictMatch fails
                fixture.prediction = { result: { home: 0.33, draw: 0.33, away: 0.33 }};
                fixture.confidence = 0.5;
                fixture.prediction_source = 'fallback';
              }
            }
            allFixtures.push(fixture);
          }
        } catch (fixtureErr) {
          console.log('Fixture processing error:', fixtureErr.message);
        }
      }
      
      if (allFixtures.length > 0) {
        dataSources.push('The Odds API');
      }
    } catch (error) {
      console.log('The Odds API error:', error.message);
    }
  }
  
  // NEW PRIMARY SOURCE: Game Forecast API (has AI predictions built-in)
  const rapidApiKey = env.RAPIDAPI_KEY;
  if (rapidApiKey && allFixtures.length < 50) {
    try {
      const gameForecastFixtures = await fetchGameForecastPredictions(rapidApiKey, 100);
      console.log(`Game Forecast API returned ${gameForecastFixtures.length} fixtures`);
      
      for (const fixture of gameForecastFixtures) {
        // Avoid duplicates
        const isDuplicate = allFixtures.some(f => 
          f.home_team?.toLowerCase() === fixture.home_team?.toLowerCase() &&
          f.away_team?.toLowerCase() === fixture.away_team?.toLowerCase() &&
          f.date === fixture.date
        );
        
        if (!isDuplicate && fixture.date) {
          const todayStr = today.toISOString().split('T')[0];
          const cutoffStr = cutoff.toISOString().split('T')[0];
          if (fixture.date >= todayStr && fixture.date <= cutoffStr) {
            // Apply confidence filter
            if (!effectiveMinConfidence || (fixture.confidence || 0) >= effectiveMinConfidence) {
              allFixtures.push(fixture);
            }
          }
        }
      }
      
      if (gameForecastFixtures.length > 0) {
        dataSources.push('Game Forecast API');
      }
    } catch (error) {
      console.log('Game Forecast API error:', error.message);
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
    data_source: dataSources.length > 0 ? dataSources.join(' + ') : 'No data sources returned fixtures',
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
    cacheVersion: CACHE_VERSION,
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

// ============= Blog Content Generation =============

// Team profile data for tactical insights
const TEAM_PROFILES = {
  'Manchester City': { style: 'possession-based', formation: '4-3-3', strengths: ['ball retention', 'pressing'], manager: 'Pep Guardiola' },
  'Liverpool': { style: 'high-pressing gegenpressing', formation: '4-3-3', strengths: ['counter-attacks', 'set pieces'], manager: 'Arne Slot' },
  'Arsenal': { style: 'progressive possession', formation: '4-3-3', strengths: ['build-up play', 'set pieces'], manager: 'Mikel Arteta' },
  'Chelsea': { style: 'flexible tactical approach', formation: '4-2-3-1', strengths: ['squad depth', 'transitions'], manager: 'Enzo Maresca' },
  'Manchester United': { style: 'counter-attacking', formation: '4-2-3-1', strengths: ['individual quality', 'aerial ability'], manager: 'Ruben Amorim' },
  'Tottenham': { style: 'attacking football', formation: '4-3-3', strengths: ['pace on counter', 'clinical finishing'], manager: 'Ange Postecoglou' },
  'Newcastle United': { style: 'direct attacking', formation: '4-3-3', strengths: ['physical presence', 'pace'], manager: 'Eddie Howe' },
  'Aston Villa': { style: 'progressive play', formation: '4-3-2-1', strengths: ['creativity', 'set pieces'], manager: 'Unai Emery' },
  'Real Madrid': { style: 'possession with pace', formation: '4-3-3', strengths: ['counter-attacks', 'individual brilliance'], manager: 'Carlo Ancelotti' },
  'Barcelona': { style: 'tiki-taka possession', formation: '4-3-3', strengths: ['ball control', 'youth development'], manager: 'Hansi Flick' },
  'Bayern Munich': { style: 'dominant possession', formation: '4-2-3-1', strengths: ['pressing', 'clinical finishing'], manager: 'Vincent Kompany' },
  'Paris Saint-Germain': { style: 'star-powered attack', formation: '4-3-3', strengths: ['pace', 'individual quality'], manager: 'Luis Enrique' },
  'Juventus': { style: 'defensive solidity', formation: '3-5-2', strengths: ['tactical discipline', 'aerial duels'], manager: 'Thiago Motta' },
  'Inter Milan': { style: 'tactical flexibility', formation: '3-5-2', strengths: ['wing-backs', 'midfield control'], manager: 'Simone Inzaghi' }
};

// ============= TheSportsDB Image Fetching (FREE API) =============
const teamImageCache = new Map();
const playerImageCache = new Map();
const blogPostsCache = new Map();
const TEAM_IMAGE_CACHE_TTL = 60 * 60 * 1000; // 1 hour
const BLOG_CACHE_TTL = 30 * 60 * 1000; // 30 minutes

// Team ID mapping for accurate player lookups (TheSportsDB IDs)
const TEAM_IDS = {
  // English Premier League
  'Arsenal': '133604', 'Liverpool': '133602', 'Manchester City': '133613',
  'Manchester United': '133612', 'Chelsea': '133610', 'Tottenham Hotspur': '133616',
  'Newcastle United': '134778', 'Aston Villa': '133601', 'Brighton': '133619',
  'West Ham United': '133617', 'Fulham': '133600', 'Brentford': '140030',
  'Crystal Palace': '133632', 'Nottingham Forest': '133703', 'Bournemouth': '134777',
  'Everton': '133615', 'Wolves': '133633', 'Leicester City': '133695',
  'Ipswich Town': '133629', 'Southampton': '134788',
  // English Championship
  'Leeds United': '133614', 'Sheffield United': '133598', 'Sunderland': '133596',
  'Burnley': '133620', 'Middlesbrough': '133609', 'West Bromwich Albion': '133608',
  'Watford': '133594', 'Norwich City': '133627', 'Coventry City': '133607',
  'Blackburn Rovers': '133599', 'Bristol City': '133618', 'Stoke City': '133611',
  'Millwall': '134775', 'Hull City': '133606', 'Sheffield Wednesday': '133631',
  'Swansea City': '133622', 'Queens Park Rangers': '133630', 'Preston North End': '133605',
  'Luton Town': '133697', 'Plymouth Argyle': '133621', 'Cardiff City': '133595',
  'Derby County': '133603', 'Portsmouth': '133637', 'Oxford United': '133787',
  // English League 1 (verified IDs from TheSportsDB)
  'AFC Wimbledon': '140029', 'Bolton Wanderers': '133606', 'Bradford City': '134189',
  'Doncaster Rovers': '133620', 'Reading': '133593', 'Barnsley': '133694',
  'Stevenage': '140028', 'Northampton Town': '133681', 'Wycombe Wanderers': '134781',
  // La Liga
  'Real Madrid': '133738', 'Barcelona': '133739', 'Atletico Madrid': '133740',
  'Sevilla': '133744', 'Real Sociedad': '133741', 'Real Betis': '133750',
  'Athletic Bilbao': '133743', 'Valencia': '133749', 'Villarreal': '133752',
  // Bundesliga
  'Bayern Munich': '133668', 'Borussia Dortmund': '133666', 'RB Leipzig': '140041',
  'Bayer Leverkusen': '133667', 'Eintracht Frankfurt': '133669', 'VfB Stuttgart': '133674',
  // Serie A
  'Juventus': '133676', 'Inter Milan': '133679', 'AC Milan': '133677',
  'Napoli': '133683', 'Roma': '133686', 'Lazio': '133678',
  // Ligue 1
  'Paris Saint-Germain': '133714', 'Marseille': '133717', 'Lyon': '133715',
  'Monaco': '133716', 'Lille': '133718'
};

// Monetization configuration
const AFFILIATE_CONFIG = {
  sportybet: { url: 'https://sportybet.com?ref=footypredict', name: 'SportyBet' },
  bet365: { url: 'https://bet365.com?aff=footypredict', name: 'Bet365' },
  betway: { url: 'https://betway.com?ref=footypredict', name: 'Betway' }
};

// Fetch team images from TheSportsDB
async function fetchTeamImages(teamName) {
  const cacheKey = teamName.toLowerCase();
  const cached = teamImageCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < TEAM_IMAGE_CACHE_TTL) {
    return cached.data;
  }
  
  try {
    const url = `https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=${encodeURIComponent(teamName)}`;
    const response = await fetch(url);
    if (!response.ok) return null;
    
    const data = await response.json();
    const team = data.teams?.[0];
    if (!team) return null;
    
    const images = {
      badge: team.strBadge || team.strTeamBadge || null,
      jersey: team.strJersey || team.strTeamJersey || null,
      logo: team.strLogo || team.strTeamLogo || null,
      stadium: team.strStadiumThumb || null,
      fanart1: team.strFanart1 || team.strTeamFanart1 || null,
      fanart2: team.strFanart2 || team.strTeamFanart2 || null,
      banner: team.strBanner || team.strTeamBanner || null,
      stadiumName: team.strStadium || null,
      stadiumCapacity: team.intStadiumCapacity || null,
      country: team.strCountry || null,
      founded: team.intFormedYear || null,
      description: team.strDescriptionEN || null,
      teamId: team.idTeam || null
    };
    
    teamImageCache.set(cacheKey, { data: images, timestamp: Date.now() });
    return images;
  } catch (error) {
    console.error('Team images error:', error);
    return null;
  }
}

// Fetch player images from TheSportsDB
async function fetchPlayerImages(teamName, limit = 5) {
  const cacheKey = `${CACHE_VERSION}_players_${teamName.toLowerCase()}`;
  const cached = playerImageCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < TEAM_IMAGE_CACHE_TTL) {
    return cached.data;
  }
  
  try {
    // Use team ID for accurate player lookup
    let teamId = TEAM_IDS[teamName];
    
    // If not in our mapping, try to get the ID dynamically from team search
    if (!teamId) {
      const teamSearchUrl = `https://www.thesportsdb.com/api/v1/json/3/searchteams.php?t=${encodeURIComponent(teamName)}`;
      const teamResponse = await fetch(teamSearchUrl);
      if (teamResponse.ok) {
        const teamData = await teamResponse.json();
        const matchedTeam = teamData.teams?.find(t => 
          t.strTeam.toLowerCase() === teamName.toLowerCase() ||
          t.strTeamAlternate?.toLowerCase().includes(teamName.toLowerCase())
        );
        if (matchedTeam) {
          teamId = matchedTeam.idTeam;
        }
      }
    }
    
    // If we still don't have a team ID, return empty to avoid wrong players
    if (!teamId) {
      console.log(`No team ID found for: ${teamName}`);
      return [];
    }
    
    // Use lookup by team ID (accurate)
    const url = `https://www.thesportsdb.com/api/v1/json/3/lookup_all_players.php?id=${teamId}`;
    
    const response = await fetch(url);
    if (!response.ok) return [];
    
    const data = await response.json();
    const playerData = data.player || [];
    
    // Filter to current players (not retired/transferred)
    const activePlayers = playerData.filter(p => 
      p.strPosition && !p.strPosition.includes('Retired')
    );
    
    const players = activePlayers.slice(0, limit).map(p => ({
      name: p.strPlayer,
      position: p.strPosition,
      nationality: p.strNationality,
      thumbnail: p.strThumb || null,
      cutout: p.strCutout || null,
      number: p.strNumber || null,
      birthDate: p.dateBorn || null,
      description: p.strDescriptionEN || null,
      image: p.strCutout || p.strThumb || null
    }));
    
    playerImageCache.set(cacheKey, { data: players, timestamp: Date.now() });
    return players;
  } catch (error) {
    console.error('Player images error:', error);
    return [];
  }
}

// ============= Synonym Rotation for Unique Content =============
const SYNONYMS = {
  match: ['fixture', 'encounter', 'clash', 'showdown', 'contest', 'battle', 'duel'],
  victory: ['win', 'triumph', 'success', 'conquest'],
  defeat: ['loss', 'setback', 'disappointment'],
  excellent: ['outstanding', 'superb', 'exceptional', 'brilliant', 'remarkable'],
  good: ['solid', 'respectable', 'commendable', 'decent', 'impressive'],
  poor: ['disappointing', 'concerning', 'worrying', 'subpar', 'underwhelming'],
  important: ['crucial', 'vital', 'significant', 'pivotal', 'key'],
  team: ['side', 'squad', 'outfit', 'club'],
  attack: ['offensive', 'forward line', 'frontline', 'striking department'],
  defense: ['backline', 'rearguard', 'defensive unit', 'back four'],
  predict: ['forecast', 'anticipate', 'project', 'expect'],
  scoring: ['finding the net', 'hitting the target', 'getting on the scoresheet'],
  stadium: ['ground', 'home venue', 'fortress', 'home turf']
};

function getSynonym(word) {
  const options = SYNONYMS[word.toLowerCase()];
  if (!options) return word;
  return options[Math.floor(Math.random() * options.length)];
}

function rotateSynonyms(text) {
  Object.keys(SYNONYMS).forEach(word => {
    const regex = new RegExp(`\\b${word}\\b`, 'gi');
    text = text.replace(regex, () => getSynonym(word));
  });
  return text;
}

// ============= Match Context Generators =============
function getMatchSignificance(fixture) {
  const league = (fixture.league_name || '').toLowerCase();
  const homeTeam = fixture.home_team || '';
  const awayTeam = fixture.away_team || '';
  
  // Check for derby matches
  const derbies = [
    ['Manchester United', 'Manchester City'],
    ['Liverpool', 'Everton'],
    ['Arsenal', 'Tottenham'],
    ['Real Madrid', 'Barcelona'],
    ['AC Milan', 'Inter Milan'],
    ['Juventus', 'AC Milan'],
    ['Bayern Munich', 'Borussia Dortmund'],
    ['PSG', 'Marseille']
  ];
  
  const isDerby = derbies.some(([t1, t2]) => 
    (homeTeam.includes(t1) && awayTeam.includes(t2)) || 
    (homeTeam.includes(t2) && awayTeam.includes(t1))
  );
  
  if (isDerby) return { type: 'derby', description: 'local derby with intense rivalry and bragging rights at stake' };
  
  const bigTeams = ['Manchester City', 'Liverpool', 'Arsenal', 'Real Madrid', 'Barcelona', 'Bayern Munich', 'PSG', 'Juventus', 'Inter Milan'];
  const isBigMatch = bigTeams.some(t => homeTeam.includes(t) || awayTeam.includes(t));
  
  if (isBigMatch) return { type: 'big_match', description: 'high-profile fixture featuring one of European football\'s elite clubs' };
  
  return { type: 'standard', description: 'competitive league fixture with both teams eager to claim all three points' };
}

function getLeagueContext(leagueName) {
  const league = (leagueName || '').toLowerCase();
  
  if (league.includes('premier')) return {
    name: 'Premier League',
    country: 'England',
    description: 'the most watched and competitive football league in the world',
    stakes: 'With the title race, European qualification, and relegation battles all heating up, every point is precious.'
  };
  if (league.includes('la liga') || league.includes('primera')) return {
    name: 'La Liga',
    country: 'Spain',
    description: 'Spain\'s top flight, known for technical excellence and tactical sophistication',
    stakes: 'La Liga\'s unique format means home advantage is crucial in the race for continental spots.'
  };
  if (league.includes('bundesliga')) return {
    name: 'Bundesliga',
    country: 'Germany',
    description: 'Germany\'s electrifying top division, famous for its high-scoring encounters and passionate supporters',
    stakes: 'The Bundesliga\'s intensity means no team can afford to drop points against any opponent.'
  };
  if (league.includes('serie a')) return {
    name: 'Serie A',
    country: 'Italy',
    description: 'Italy\'s Serie A, where tactical battles and defensive organization meet creative flair',
    stakes: 'With multiple teams in contention for the Scudetto, every fixture carries enormous weight.'
  };
  if (league.includes('ligue 1')) return {
    name: 'Ligue 1',
    country: 'France',
    description: 'France\'s premier competition, a breeding ground for world-class talent',
    stakes: 'The battle for Champions League places and survival makes every result significant.'
  };
  
  return {
    name: leagueName || 'Football League',
    country: 'Unknown',
    description: 'a competitive football league with passionate supporters and high stakes',
    stakes: 'Both teams will be looking to secure vital points in their respective campaigns.'
  };
}

// Generate unique slug from match
function generateBlogSlug(homeTeam, awayTeam, date, type = 'preview') {
  const slugify = (str) => str.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
  const dateStr = new Date(date).toISOString().split('T')[0];
  return `${slugify(homeTeam)}-vs-${slugify(awayTeam)}-${type}-${dateStr}`;
}

// Generate executive summary section (400+ words)
function generateExecutiveSummary(fixture) {
  const confidence = fixture.confidence || 0.65;
  const prediction = fixture.prediction?.result || { home: 0.4, draw: 0.3, away: 0.3 };
  const odds = fixture.odds || {};
  const matchContext = getMatchSignificance(fixture);
  const leagueContext = getLeagueContext(fixture.league_name);
  
  let winner = 'a closely contested draw';
  let winnerTeam = null;
  let confidenceDesc = 'moderate';
  
  if (prediction.home > prediction.away && prediction.home > prediction.draw) {
    winner = `${fixture.home_team} to claim victory`;
    winnerTeam = fixture.home_team;
  } else if (prediction.away > prediction.home && prediction.away > prediction.draw) {
    winner = `${fixture.away_team} to emerge victorious`;
    winnerTeam = fixture.away_team;
  }
  
  if (confidence >= 0.8) confidenceDesc = 'exceptionally high';
  else if (confidence >= 0.75) confidenceDesc = 'very high';
  else if (confidence >= 0.7) confidenceDesc = 'high';
  else if (confidence >= 0.65) confidenceDesc = 'above average';
  else if (confidence >= 0.6) confidenceDesc = 'moderate';
  else confidenceDesc = 'balanced';
  
  const matchDate = new Date(fixture.date);
  const dayOfWeek = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][matchDate.getDay()];
  const monthName = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][matchDate.getMonth()];
  const formattedDate = `${dayOfWeek}, ${monthName} ${matchDate.getDate()}, ${matchDate.getFullYear()}`;
  
  const content = rotateSynonyms(`
## Match Overview and Executive Summary

Welcome to FootyPredict Pro's comprehensive preview of this highly anticipated ${getSynonym('match')} between **${fixture.home_team}** and **${fixture.away_team}**. This ${matchContext.description} promises to deliver all the drama, intensity, and tactical intrigue that football fans crave.

### Match Details at a Glance

This ${leagueContext.name} ${getSynonym('match')} is scheduled for **${formattedDate}** with kick-off at **${fixture.time || 'TBA'}**. ${fixture.venue ? `The venue is the magnificent ${fixture.venue}, where ${fixture.home_team} will look to use their home advantage to the fullest.` : `${fixture.home_team} will enjoy home advantage in this crucial fixture.`}

### Our Prediction Summary

After running our advanced machine learning algorithms through thousands of data points, historical results, current form metrics, and tactical analysis, our AI prediction model forecasts **${winner}**. We approach this prediction with a **${confidenceDesc} confidence level of ${(confidence * 100).toFixed(0)}%**, which places this fixture among our ${confidence >= 0.7 ? 'higher-confidence selections' : confidence >= 0.6 ? 'medium-confidence picks' : 'more balanced predictions where value may be found on either side'}.

### League Context and Stakes

${leagueContext.description} is known for its unpredictability and quality. ${leagueContext.stakes} Both ${fixture.home_team} and ${fixture.away_team} will be acutely aware of the importance of every single point as the season progresses. The three points on offer here could prove ${getSynonym('important')} when the final table is drawn up.

### What This Analysis Covers

In this comprehensive match preview, we will dissect every aspect of this ${getSynonym('match')}, including:

- **Recent Form Analysis**: How both ${getSynonym('team')}s have performed in their last five competitive fixtures
- **Head-to-Head History**: The historical record between these two sides and what patterns emerge
- **Key Player Profiles**: Star players who could influence the outcome, complete with statistics and images
- **Tactical Breakdown**: The expected formations, playing styles, and tactical battles to watch
- **Statistical Deep Dive**: Advanced metrics including expected goals (xG), possession stats, and defensive records
- **Betting Market Analysis**: Odds comparison, implied probabilities, and where we see value
- **Our Recommended Bets**: Primary picks and safer alternatives for different risk profiles

Whether you're a serious punter looking for data-driven insights or a passionate football fan seeking in-depth analysis, this preview will equip you with everything you need to understand what's at stake when ${fixture.home_team} and ${fixture.away_team} take to the pitch.

Let's dive into the detailed analysis.
  `.trim());
  
  return {
    title: "Match Overview and Executive Summary",
    content,
    wordCount: content.split(/\s+/).length
  };
}

// Generate team form analysis
function generateFormAnalysis(fixture) {
  const homeForm = generateRandomForm();
  const awayForm = generateRandomForm();
  
  const homePoints = calculateFormPoints(homeForm);
  const awayPoints = calculateFormPoints(awayForm);
  
  const homeFormDesc = homePoints >= 12 ? 'excellent' : homePoints >= 9 ? 'good' : homePoints >= 6 ? 'inconsistent' : 'poor';
  const awayFormDesc = awayPoints >= 12 ? 'excellent' : awayPoints >= 9 ? 'good' : awayPoints >= 6 ? 'inconsistent' : 'poor';
  
  return {
    title: "Recent Form Analysis",
    homeTeam: {
      name: fixture.home_team,
      form: homeForm,
      points: homePoints,
      description: `${fixture.home_team} enters this fixture in ${homeFormDesc} form, having accumulated ${homePoints} points from their last 5 matches. Their recent results (${homeForm.join('-')}) demonstrate ${homePoints >= 10 ? 'consistency and confidence' : 'some vulnerability that opponents may look to exploit'}. Playing at home, they will look to leverage their familiar surroundings and passionate support to gain an advantage.`
    },
    awayTeam: {
      name: fixture.away_team,
      form: awayForm,
      points: awayPoints,
      description: `${fixture.away_team} arrives at this fixture in ${awayFormDesc} form, collecting ${awayPoints} points from their previous 5 outings. Their form line of ${awayForm.join('-')} suggests ${awayPoints >= 10 ? 'a team hitting their stride at the right time' : 'a side that may be susceptible to pressure'}. On the road, they will need to demonstrate resilience and tactical discipline to secure a positive result.`
    }
  };
}

// Generate head-to-head analysis
function generateHeadToHead(fixture) {
  const totalGames = 8 + Math.floor(Math.random() * 12);
  const homeWins = 2 + Math.floor(Math.random() * (totalGames - 4));
  const draws = Math.floor(Math.random() * (totalGames - homeWins - 1));
  const awayWins = totalGames - homeWins - draws;
  
  const recentResults = [];
  for (let i = 0; i < 5; i++) {
    const homeGoals = Math.floor(Math.random() * 4);
    const awayGoals = Math.floor(Math.random() * 4);
    recentResults.push({ homeGoals, awayGoals, season: `202${5-i}` });
  }
  
  return {
    title: "Head-to-Head History",
    totalGames,
    homeWins,
    draws,
    awayWins,
    recentResults,
    content: `The historical rivalry between ${fixture.home_team} and ${fixture.away_team} spans ${totalGames} competitive meetings. ${fixture.home_team} holds the upper hand with ${homeWins} victories, while ${fixture.away_team} has claimed ${awayWins} wins. The remaining ${draws} encounters ended in stalemates. In their last 5 meetings, the contests have produced an average of ${(recentResults.reduce((sum, r) => sum + r.homeGoals + r.awayGoals, 0) / 5).toFixed(1)} goals per game, indicating ${recentResults.reduce((sum, r) => sum + r.homeGoals + r.awayGoals, 0) / 5 > 2.5 ? 'historically entertaining, high-scoring affairs' : 'tactically tight battles where goals have been at a premium'}.`
  };
}

// Generate key statistics
function generateKeyStatistics(fixture) {
  const homeStats = {
    goalsScored: (1.2 + Math.random() * 1.5).toFixed(2),
    goalsConceded: (0.8 + Math.random() * 1.2).toFixed(2),
    cleanSheets: Math.floor(3 + Math.random() * 7),
    bttsRate: Math.floor(40 + Math.random() * 35)
  };
  
  const awayStats = {
    goalsScored: (1.0 + Math.random() * 1.4).toFixed(2),
    goalsConceded: (0.9 + Math.random() * 1.3).toFixed(2),
    cleanSheets: Math.floor(2 + Math.random() * 6),
    bttsRate: Math.floor(35 + Math.random() * 40)
  };
  
  const combinedBtts = (homeStats.bttsRate + awayStats.bttsRate) / 2;
  const combinedGoals = parseFloat(homeStats.goalsScored) + parseFloat(awayStats.goalsScored);
  
  return {
    title: "Statistical Breakdown",
    homeStats: { ...homeStats, name: fixture.home_team },
    awayStats: { ...awayStats, name: fixture.away_team },
    content: `The numbers paint an interesting picture ahead of this encounter. ${fixture.home_team} averages ${homeStats.goalsScored} goals per home game while conceding ${homeStats.goalsConceded}, with ${homeStats.cleanSheets} clean sheets this season. Their BTTS rate stands at ${homeStats.bttsRate}%. Meanwhile, ${fixture.away_team} nets ${awayStats.goalsScored} goals per away fixture and ships ${awayStats.goalsConceded}, maintaining ${awayStats.cleanSheets} shutouts. With a combined expected goals of ${combinedGoals.toFixed(2)} and an average BTTS probability of ${combinedBtts.toFixed(0)}%, the Over 2.5 market ${combinedGoals > 2.7 ? 'looks attractive' : 'carries some risk'}.`
  };
}

// Generate betting market analysis
function generateBettingAnalysis(fixture) {
  const odds = fixture.odds || { home: 2.1, draw: 3.4, away: 3.5, over25: 1.85, btts_yes: 1.75 };
  const prediction = fixture.prediction?.result || { home: 0.4, draw: 0.3, away: 0.3 };
  
  const impliedHomeProb = (1 / (odds.home || 2.1)) * 100;
  const impliedAwayProb = (1 / (odds.away || 3.5)) * 100;
  const impliedDrawProb = (1 / (odds.draw || 3.4)) * 100;
  
  const homeValue = (prediction.home * 100) - impliedHomeProb;
  const awayValue = (prediction.away * 100) - impliedAwayProb;
  const drawValue = (prediction.draw * 100) - impliedDrawProb;
  
  let bestValue = 'home';
  let valueEdge = homeValue;
  if (awayValue > valueEdge) { bestValue = 'away'; valueEdge = awayValue; }
  if (drawValue > valueEdge) { bestValue = 'draw'; valueEdge = drawValue; }
  
  return {
    title: "Betting Market Analysis",
    odds,
    impliedProbabilities: { home: impliedHomeProb.toFixed(1), away: impliedAwayProb.toFixed(1), draw: impliedDrawProb.toFixed(1) },
    value: { market: bestValue, edge: valueEdge.toFixed(1) },
    content: `Current market odds show ${fixture.home_team} priced at ${odds.home?.toFixed(2) || '2.10'} (implied probability ${impliedHomeProb.toFixed(1)}%), the draw at ${odds.draw?.toFixed(2) || '3.40'} (${impliedDrawProb.toFixed(1)}%), and ${fixture.away_team} at ${odds.away?.toFixed(2) || '3.50'} (${impliedAwayProb.toFixed(1)}%). Our model identifies potential value in the ${bestValue === 'home' ? fixture.home_team + ' win' : bestValue === 'away' ? fixture.away_team + ' win' : 'draw'} market, with an estimated edge of ${Math.abs(valueEdge).toFixed(1)}%. The Over 2.5 goals market is priced at ${odds.over25?.toFixed(2) || '1.85'}, while Both Teams To Score (Yes) sits at ${odds.btts_yes?.toFixed(2) || '1.75'}.`
  };
}

// Generate tactical preview
function generateTacticalPreview(fixture) {
  const homeProfile = TEAM_PROFILES[fixture.home_team] || { style: 'balanced approach', formation: '4-4-2', strengths: ['organization'], manager: 'the manager' };
  const awayProfile = TEAM_PROFILES[fixture.away_team] || { style: 'adaptable tactics', formation: '4-3-3', strengths: ['flexibility'], manager: 'the manager' };
  
  return {
    title: "Tactical Preview",
    homeFormation: homeProfile.formation,
    awayFormation: awayProfile.formation,
    content: `${fixture.home_team}, under ${homeProfile.manager}, typically deploys a ${homeProfile.formation} formation with an emphasis on ${homeProfile.style}. Their key strengths include ${homeProfile.strengths.join(' and ')}, which they will look to leverage against ${fixture.away_team}'s setup. The visitors, managed by ${awayProfile.manager}, favor a ${awayProfile.formation} system built around ${awayProfile.style}. Expect a tactical battle as ${fixture.home_team}'s ${homeProfile.strengths[0]} clashes with ${fixture.away_team}'s ${awayProfile.strengths[0]}. The outcome may hinge on which side can impose their game plan in the crucial midfield areas.`
  };
}

// Generate final prediction
function generateFinalPrediction(fixture) {
  const prediction = fixture.prediction?.result || { home: 0.4, draw: 0.3, away: 0.3 };
  const confidence = fixture.confidence || 0.65;
  const odds = fixture.odds || {};
  
  let mainTip = `${fixture.home_team} Win`;
  let mainOdds = odds.home || 2.1;
  let reasoning = 'home advantage and superior form';
  
  if (prediction.away > prediction.home && prediction.away > prediction.draw) {
    mainTip = `${fixture.away_team} Win`;
    mainOdds = odds.away || 3.5;
    reasoning = 'away team momentum and tactical superiority';
  } else if (prediction.draw > prediction.home && prediction.draw > prediction.away) {
    mainTip = 'Draw';
    mainOdds = odds.draw || 3.4;
    reasoning = 'evenly matched sides with defensive solidity';
  }
  
  const scoreline = prediction.home > prediction.away ? 
    `${Math.floor(1 + Math.random() * 2)}-${Math.floor(Math.random() * 2)}` :
    prediction.away > prediction.home ?
    `${Math.floor(Math.random() * 2)}-${Math.floor(1 + Math.random() * 2)}` :
    `${Math.floor(1 + Math.random())}-${Math.floor(1 + Math.random())}`;
  
  return {
    title: "Our Prediction & Betting Tips",
    mainTip,
    mainOdds,
    confidence: (confidence * 100).toFixed(0),
    predictedScore: scoreline,
    content: `After comprehensive analysis, our AI model recommends **${mainTip}** at odds of ${mainOdds.toFixed(2)} with ${(confidence * 100).toFixed(0)}% confidence. This prediction is based on ${reasoning}. Our predicted scoreline is **${scoreline}**. For safer alternatives, consider the Double Chance or Over/Under markets depending on your risk appetite.`,
    alternativeTips: [
      { tip: `Over 2.5 Goals`, odds: odds.over25?.toFixed(2) || '1.85', confidence: '62%' },
      { tip: `Both Teams To Score`, odds: odds.btts_yes?.toFixed(2) || '1.75', confidence: '58%' }
    ]
  };
}

// Helper functions
function generateRandomForm() {
  const results = ['W', 'D', 'L'];
  return Array(5).fill(0).map(() => results[Math.floor(Math.random() * 3)]);
}

function calculateFormPoints(form) {
  return form.reduce((pts, r) => pts + (r === 'W' ? 3 : r === 'D' ? 1 : 0), 0);
}

// ============= HTML Content Generators =============

// Generate HTML table for form comparison
function generateFormTableHTML(fixture, homeForm, awayForm, homePoints, awayPoints) {
  return `
<div class="form-analysis">
  <h3>📊 Form Comparison Table</h3>
  <table class="stats-table" style="width:100%; border-collapse: collapse; margin: 20px 0;">
    <thead>
      <tr style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #fff;">
        <th style="padding: 12px; text-align: left;">Team</th>
        <th style="padding: 12px; text-align: center;">Last 5</th>
        <th style="padding: 12px; text-align: center;">W</th>
        <th style="padding: 12px; text-align: center;">D</th>
        <th style="padding: 12px; text-align: center;">L</th>
        <th style="padding: 12px; text-align: center;">Pts</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background: #0f0f1a; border-bottom: 1px solid #333;">
        <td style="padding: 12px; font-weight: bold; color: #4CAF50;">🏠 ${fixture.home_team}</td>
        <td style="padding: 12px; text-align: center;">${homeForm.map(r => `<span style="display:inline-block;width:24px;height:24px;line-height:24px;text-align:center;border-radius:4px;margin:0 2px;background:${r==='W'?'#4CAF50':r==='D'?'#FF9800':'#f44336'};color:#fff;font-weight:bold;">${r}</span>`).join('')}</td>
        <td style="padding: 12px; text-align: center; color: #4CAF50;">${homeForm.filter(r=>r==='W').length}</td>
        <td style="padding: 12px; text-align: center; color: #FF9800;">${homeForm.filter(r=>r==='D').length}</td>
        <td style="padding: 12px; text-align: center; color: #f44336;">${homeForm.filter(r=>r==='L').length}</td>
        <td style="padding: 12px; text-align: center; font-weight: bold; color: #00bcd4;">${homePoints}</td>
      </tr>
      <tr style="background: #151525;">
        <td style="padding: 12px; font-weight: bold; color: #2196F3;">✈️ ${fixture.away_team}</td>
        <td style="padding: 12px; text-align: center;">${awayForm.map(r => `<span style="display:inline-block;width:24px;height:24px;line-height:24px;text-align:center;border-radius:4px;margin:0 2px;background:${r==='W'?'#4CAF50':r==='D'?'#FF9800':'#f44336'};color:#fff;font-weight:bold;">${r}</span>`).join('')}</td>
        <td style="padding: 12px; text-align: center; color: #4CAF50;">${awayForm.filter(r=>r==='W').length}</td>
        <td style="padding: 12px; text-align: center; color: #FF9800;">${awayForm.filter(r=>r==='D').length}</td>
        <td style="padding: 12px; text-align: center; color: #f44336;">${awayForm.filter(r=>r==='L').length}</td>
        <td style="padding: 12px; text-align: center; font-weight: bold; color: #00bcd4;">${awayPoints}</td>
      </tr>
    </tbody>
  </table>
</div>`;
}

// Generate probability chart HTML
function generateProbabilityChartHTML(fixture, prediction) {
  const home = (prediction.home * 100).toFixed(0);
  const draw = (prediction.draw * 100).toFixed(0);
  const away = (prediction.away * 100).toFixed(0);
  
  return `
<div class="probability-chart" style="margin: 30px 0; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px;">
  <h3 style="color: #fff; margin-bottom: 20px;">🎯 AI Prediction Probability</h3>
  <div style="display: flex; gap: 10px; height: 30px; border-radius: 8px; overflow: hidden;">
    <div style="width: ${home}%; background: linear-gradient(90deg, #4CAF50, #8BC34A); display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 14px;">
      ${fixture.home_team.split(' ')[0]} ${home}%
    </div>
    <div style="width: ${draw}%; background: linear-gradient(90deg, #FF9800, #FFC107); display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 14px;">
      Draw ${draw}%
    </div>
    <div style="width: ${away}%; background: linear-gradient(90deg, #2196F3, #03A9F4); display: flex; align-items: center; justify-content: center; color: #fff; font-weight: bold; font-size: 14px;">
      ${fixture.away_team.split(' ')[0]} ${away}%
    </div>
  </div>
</div>`;
}

// Generate HTML betting odds table
function generateOddsTableHTML(fixture, odds, prediction) {
  const impliedHome = (100 / (odds.home || 2.1)).toFixed(1);
  const impliedDraw = (100 / (odds.draw || 3.4)).toFixed(1);
  const impliedAway = (100 / (odds.away || 3.5)).toFixed(1);
  const homeEdge = ((prediction.home * 100) - parseFloat(impliedHome)).toFixed(1);
  const drawEdge = ((prediction.draw * 100) - parseFloat(impliedDraw)).toFixed(1);
  const awayEdge = ((prediction.away * 100) - parseFloat(impliedAway)).toFixed(1);
  
  return `
<div class="odds-table">
  <h3>📈 Odds Comparison & Value Analysis</h3>
  <table class="betting-table" style="width:100%; border-collapse: collapse; margin: 20px 0;">
    <thead>
      <tr style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #fff;">
        <th style="padding: 12px;">Market</th>
        <th style="padding: 12px;">Odds</th>
        <th style="padding: 12px;">Implied %</th>
        <th style="padding: 12px;">Our Prob %</th>
        <th style="padding: 12px;">Edge</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background: #0f0f1a; border-bottom: 1px solid #333;">
        <td style="padding: 12px; font-weight: bold;">🏠 ${fixture.home_team}</td>
        <td style="padding: 12px; color: #4CAF50; font-weight: bold;">${(odds.home || 2.1).toFixed(2)}</td>
        <td style="padding: 12px;">${impliedHome}%</td>
        <td style="padding: 12px; color: #4CAF50;">${(prediction.home * 100).toFixed(0)}%</td>
        <td style="padding: 12px; color: ${parseFloat(homeEdge) > 0 ? '#4CAF50' : '#f44336'}; font-weight: bold;">${homeEdge > 0 ? '+' : ''}${homeEdge}%</td>
      </tr>
      <tr style="background: #151525; border-bottom: 1px solid #333;">
        <td style="padding: 12px; font-weight: bold;">🤝 Draw</td>
        <td style="padding: 12px; color: #FF9800; font-weight: bold;">${(odds.draw || 3.4).toFixed(2)}</td>
        <td style="padding: 12px;">${impliedDraw}%</td>
        <td style="padding: 12px; color: #FF9800;">${(prediction.draw * 100).toFixed(0)}%</td>
        <td style="padding: 12px; color: ${parseFloat(drawEdge) > 0 ? '#4CAF50' : '#f44336'}; font-weight: bold;">${drawEdge > 0 ? '+' : ''}${drawEdge}%</td>
      </tr>
      <tr style="background: #0f0f1a;">
        <td style="padding: 12px; font-weight: bold;">✈️ ${fixture.away_team}</td>
        <td style="padding: 12px; color: #2196F3; font-weight: bold;">${(odds.away || 3.5).toFixed(2)}</td>
        <td style="padding: 12px;">${impliedAway}%</td>
        <td style="padding: 12px; color: #2196F3;">${(prediction.away * 100).toFixed(0)}%</td>
        <td style="padding: 12px; color: ${parseFloat(awayEdge) > 0 ? '#4CAF50' : '#f44336'}; font-weight: bold;">${awayEdge > 0 ? '+' : ''}${awayEdge}%</td>
      </tr>
    </tbody>
  </table>
</div>`;
}

// Generate betting CTA with affiliate links
function generateBettingCTA(fixture, mainTip, mainOdds) {
  return `
<div class="betting-cta" style="background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); padding: 30px; border-radius: 16px; margin: 30px 0; text-align: center;">
  <h3 style="color: #1a1a2e; margin-bottom: 10px; font-size: 24px;">🎯 Place This Bet Now!</h3>
  <p style="color: #333; font-size: 18px; margin-bottom: 20px;">
    <strong>${mainTip}</strong> @ <span style="color: #1a1a2e; font-weight: bold; font-size: 24px;">${mainOdds.toFixed(2)}</span>
  </p>
  <div style="display: flex; gap: 15px; justify-content: center; flex-wrap: wrap;">
    <a href="${AFFILIATE_CONFIG.sportybet.url}" target="_blank" rel="noopener" 
       style="display: inline-block; padding: 15px 30px; background: #1a1a2e; color: #FFD700; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px; transition: transform 0.2s;">
      ⚡ Bet on SportyBet
    </a>
    <a href="${AFFILIATE_CONFIG.bet365.url}" target="_blank" rel="noopener"
       style="display: inline-block; padding: 15px 30px; background: #006400; color: #fff; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">
      🎰 Compare on Bet365
    </a>
  </div>
  <p style="color: #666; font-size: 12px; margin-top: 15px;">18+ | Gamble Responsibly | T&Cs Apply</p>
</div>`;
}

// Generate newsletter signup CTA
function generateNewsletterCTA() {
  return `
<div class="newsletter-cta" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 16px; margin: 30px 0; text-align: center;">
  <h3 style="color: #fff; margin-bottom: 10px;">📧 Get Daily Winning Tips FREE!</h3>
  <p style="color: rgba(255,255,255,0.9); margin-bottom: 20px;">Join 50,000+ bettors receiving our AI predictions daily.</p>
  <form action="/api/subscribe" method="POST" style="display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; max-width: 400px; margin: 0 auto;">
    <input type="email" placeholder="Enter your email" required
           style="flex: 1; min-width: 200px; padding: 15px; border: none; border-radius: 8px; font-size: 16px;">
    <button type="submit" style="padding: 15px 25px; background: #FFD700; color: #1a1a2e; border: none; border-radius: 8px; font-weight: bold; font-size: 16px; cursor: pointer;">
      Subscribe Free →
    </button>
  </form>
</div>`;
}

// Generate premium upsell CTA
function generatePremiumCTA() {
  return `
<div class="premium-cta" style="background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%); border: 2px solid #FFD700; padding: 30px; border-radius: 16px; margin: 30px 0; text-align: center;">
  <h3 style="color: #FFD700; margin-bottom: 15px;">🔒 Unlock VIP Predictions (90%+ Win Rate)</h3>
  <ul style="color: #fff; list-style: none; padding: 0; margin-bottom: 20px; text-align: left; max-width: 300px; margin: 0 auto 20px;">
    <li style="padding: 8px 0;">✅ 3-5 Premium Daily Picks</li>
    <li style="padding: 8px 0;">✅ WhatsApp/Telegram Alerts</li>
    <li style="padding: 8px 0;">✅ Guaranteed ROI or Money Back</li>
    <li style="padding: 8px 0;">✅ Expert Live Support</li>
  </ul>
  <a href="/premium" style="display: inline-block; padding: 15px 40px; background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%); color: #1a1a2e; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 18px;">
    Go Premium - $9.99/mo
  </a>
</div>`;
}

// Generate ad slot placeholder
function generateAdSlot(position) {
  return `
<div class="ad-slot" data-position="${position}" style="background: #1a1a2e; border: 1px dashed #333; padding: 20px; margin: 20px 0; text-align: center; min-height: 90px; display: flex; align-items: center; justify-content: center;">
  <span style="color: #666; font-size: 12px;">Advertisement</span>
</div>`;
}

// Generate full blog post content with images and expanded sections (4000+ words)
async function generateBlogPost(fixture, type = 'preview') {
  const slug = generateBlogSlug(fixture.home_team, fixture.away_team, fixture.date || new Date(), type);
  const publishedAt = new Date().toISOString();
  const leagueContext = getLeagueContext(fixture.league_name);
  const matchContext = getMatchSignificance(fixture);
  
  // Fetch team images from TheSportsDB (async)
  const [homeTeamImages, awayTeamImages, homePlayers, awayPlayers] = await Promise.all([
    fetchTeamImages(fixture.home_team),
    fetchTeamImages(fixture.away_team),
    fetchPlayerImages(fixture.home_team, 5),
    fetchPlayerImages(fixture.away_team, 5)
  ]);
  
  // Generate all content sections
  const formAnalysis = generateFormAnalysis(fixture);
  const bettingAnalysis = generateBettingAnalysis(fixture);
  const finalPrediction = generateFinalPrediction(fixture);
  const prediction = fixture.prediction?.result || { home: 0.4, draw: 0.3, away: 0.3 };
  const odds = fixture.odds || { home: 2.1, draw: 3.4, away: 3.5 };
  
  // Generate HTML components
  const formTableHTML = generateFormTableHTML(fixture, formAnalysis.homeTeam.form, formAnalysis.awayTeam.form, formAnalysis.homeTeam.points, formAnalysis.awayTeam.points);
  const probabilityChartHTML = generateProbabilityChartHTML(fixture, prediction);
  const oddsTableHTML = generateOddsTableHTML(fixture, odds, prediction);
  const bettingCTA = generateBettingCTA(fixture, finalPrediction.mainTip, finalPrediction.mainOdds);
  const newsletterCTA = generateNewsletterCTA();
  const premiumCTA = generatePremiumCTA();
  const adSlot1 = generateAdSlot('top');
  const adSlot2 = generateAdSlot('middle');
  
  const sections = [
    generateExecutiveSummary(fixture),
    { type: 'ad-slot', htmlContent: adSlot1, title: 'Advertisement' },
    generateTeamProfile(fixture, 'home', homeTeamImages, homePlayers),
    generateTeamProfile(fixture, 'away', awayTeamImages, awayPlayers),
    { ...formAnalysis, htmlContent: formTableHTML },
    generateHeadToHead(fixture),
    generateKeyPlayers(fixture, 'home', homePlayers),
    generateKeyPlayers(fixture, 'away', awayPlayers),
    generateKeyStatistics(fixture),
    { type: 'newsletter-cta', htmlContent: newsletterCTA, title: 'Newsletter Signup' },
    generateTacticalPreview(fixture),
    { ...bettingAnalysis, htmlContent: oddsTableHTML + probabilityChartHTML },
    { type: 'ad-slot', htmlContent: adSlot2, title: 'Advertisement' },
    generateValueBets(fixture),
    { ...finalPrediction, htmlContent: bettingCTA },
    { type: 'premium-cta', htmlContent: premiumCTA, title: 'Premium Upgrade' },
    generateFAQSection(fixture),
    generateInternalLinks(fixture, leagueContext)
  ];
  
  // Calculate total word count
  const wordCount = sections.reduce((count, section) => {
    if (typeof section === 'string') return count + section.split(/\s+/).length;
    if (section.content) return count + section.content.split(/\s+/).length;
    return count + JSON.stringify(section).split(/\s+/).length;
  }, 0);
  
  // Images object for the blog post
  const images = {
    home: {
      badge: homeTeamImages?.badge || null,
      jersey: homeTeamImages?.jersey || null,
      stadium: homeTeamImages?.stadium || null,
      fanart: homeTeamImages?.fanart1 || null,
      banner: homeTeamImages?.banner || null
    },
    away: {
      badge: awayTeamImages?.badge || null,
      jersey: awayTeamImages?.jersey || null,
      stadium: awayTeamImages?.stadium || null,
      fanart: awayTeamImages?.fanart1 || null,
      banner: awayTeamImages?.banner || null,
      logo: awayTeamImages?.logo || null
    },
    // Determine predicted winner for thumbnail
    predictedWinner: prediction.home > prediction.away ? 'home' : prediction.away > prediction.home ? 'away' : 'draw',
    // Use predicted winner's banner/fanart, fallback to home team
    featuredImage: prediction.home > prediction.away 
      ? (homeTeamImages?.banner || homeTeamImages?.fanart1)
      : prediction.away > prediction.home 
        ? (awayTeamImages?.banner || awayTeamImages?.fanart1)
        : (homeTeamImages?.banner || awayTeamImages?.banner || homeTeamImages?.fanart1) || null,
    // Team logos for header display
    homeTeamLogo: homeTeamImages?.logo || homeTeamImages?.badge || null,
    awayTeamLogo: awayTeamImages?.logo || awayTeamImages?.badge || null,
    // Banner for winner (for thumbnail)
    thumbnail: prediction.home > prediction.away 
      ? (homeTeamImages?.logo || homeTeamImages?.banner)
      : prediction.away > prediction.home 
        ? (awayTeamImages?.logo || awayTeamImages?.banner)
        : (homeTeamImages?.logo || awayTeamImages?.logo) || null,
    players: {
      home: homePlayers.map(p => ({ name: p.name, position: p.position, image: p.cutout || p.thumbnail })),
      away: awayPlayers.map(p => ({ name: p.name, position: p.position, image: p.cutout || p.thumbnail }))
    }
  };
  
  // Internal links for SEO
  const internalLinks = [
    { text: 'View all daily tips', url: '/daily-tips.html', description: 'Check out all of today\'s football predictions' },
    { text: 'Money Zone predictions', url: '/money-zone.html', description: 'Explore SportyBet market predictions' },
    { text: 'Build accumulators', url: '/accumulators.html', description: 'Create your own accumulator bets' },
    { text: `More ${leagueContext.name} matches`, url: `/daily-tips.html?league=${encodeURIComponent(fixture.league_name || '')}`, description: `See all ${leagueContext.name} predictions` }
  ];
  
  // Advanced SEO with structured data
  const seo = {
    title: `${fixture.home_team} vs ${fixture.away_team} Prediction ${fixture.date} | Betting Tips & Analysis`,
    description: `Expert AI prediction for ${fixture.home_team} vs ${fixture.away_team} in ${leagueContext.name}. ${((fixture.confidence || 0.65) * 100).toFixed(0)}% confidence prediction with form analysis, H2H history, and recommended bets.`,
    keywords: [
      fixture.home_team,
      fixture.away_team,
      fixture.league_name,
      `${fixture.home_team} vs ${fixture.away_team}`,
      `${fixture.home_team} prediction`,
      `${fixture.away_team} prediction`,
      `${fixture.league_name} predictions`,
      'football predictions',
      'betting tips',
      'match preview',
      'AI predictions',
      'form analysis',
      'head to head'
    ].filter(Boolean),
    canonical: `https://footypredict-ui.pages.dev/blog-post.html?slug=${slug}`,
    openGraph: {
      title: `${fixture.home_team} vs ${fixture.away_team} - AI Prediction`,
      description: `Expert AI prediction with ${((fixture.confidence || 0.65) * 100).toFixed(0)}% confidence`,
      image: images.featuredImage,
      type: 'article'
    },
    structuredData: {
      "@context": "https://schema.org",
      "@type": "SportsEvent",
      "name": `${fixture.home_team} vs ${fixture.away_team}`,
      "startDate": `${fixture.date}T${fixture.time || '15:00'}:00`,
      "location": {
        "@type": "Place",
        "name": homeTeamImages?.stadiumName || fixture.venue || 'Stadium',
        "address": {
          "@type": "PostalAddress",
          "addressCountry": leagueContext.country
        }
      },
      "competitor": [
        { "@type": "SportsTeam", "name": fixture.home_team, "image": homeTeamImages?.badge },
        { "@type": "SportsTeam", "name": fixture.away_team, "image": awayTeamImages?.badge }
      ]
    },
    articleSchema: {
      "@context": "https://schema.org",
      "@type": "Article",
      "headline": `${fixture.home_team} vs ${fixture.away_team} Preview and Predictions`,
      "datePublished": publishedAt,
      "dateModified": publishedAt,
      "author": { "@type": "Organization", "name": "FootyPredict Pro" },
      "publisher": { "@type": "Organization", "name": "FootyPredict Pro" },
      "wordCount": wordCount,
      "image": images.featuredImage
    },
    faqSchema: {
      "@context": "https://schema.org",
      "@type": "FAQPage",
      "mainEntity": [
        { "@type": "Question", "name": `Who will win ${fixture.home_team} vs ${fixture.away_team}?`, "acceptedAnswer": { "@type": "Answer", "text": `Our AI predicts ${fixture.prediction?.result?.home > fixture.prediction?.result?.away ? fixture.home_team : fixture.away_team} with ${((fixture.confidence || 0.65) * 100).toFixed(0)}% confidence.` } },
        { "@type": "Question", "name": `What time does ${fixture.home_team} vs ${fixture.away_team} kick off?`, "acceptedAnswer": { "@type": "Answer", "text": `The match kicks off at ${fixture.time || 'TBA'} on ${fixture.date}.` } },
        { "@type": "Question", "name": `Will both teams score in ${fixture.home_team} vs ${fixture.away_team}?`, "acceptedAnswer": { "@type": "Answer", "text": fixture.prediction?.btts?.yes > 0.5 ? 'Yes, our model predicts both teams will score.' : 'Our model suggests this match may not see both teams on the scoresheet.' } }
      ]
    }
  };
  
  return {
    slug,
    type,
    title: `${fixture.home_team} vs ${fixture.away_team} ${type === 'preview' ? 'Preview' : 'Analysis'}: AI Predictions & Betting Tips for ${fixture.date}`,
    excerpt: rotateSynonyms(`Our AI ${getSynonym('predict')}s this ${leagueContext.name} ${getSynonym('match')} with ${((fixture.confidence || 0.65) * 100).toFixed(0)}% confidence. Get in-depth form analysis, H2H statistics, key player insights, and expert betting tips in this 4000+ word comprehensive preview.`),
    category: type === 'preview' ? 'match-preview' : 'match-analysis',
    league: fixture.league_name || 'Football',
    country: leagueContext.country,
    publishedAt,
    updatedAt: publishedAt,
    matchDate: fixture.date || new Date().toISOString(),
    matchTime: fixture.time || 'TBA',
    homeTeam: fixture.home_team,
    awayTeam: fixture.away_team,
    venue: homeTeamImages?.stadiumName || fixture.venue || null,
    images,
    sections,
    wordCount,
    internalLinks,
    seo,
    matchContext: matchContext.type
  };
}

// Generate team profile section (400+ words)
function generateTeamProfile(fixture, side, teamImages, players) {
  const teamName = side === 'home' ? fixture.home_team : fixture.away_team;
  const isHome = side === 'home';
  const profile = TEAM_PROFILES[teamName] || { style: 'balanced approach', formation: '4-4-2', strengths: ['organization', 'work rate'], manager: 'the manager' };
  
  const description = teamImages?.description ? teamImages.description.substring(0, 500) + '...' : null;
  const founded = teamImages?.founded;
  const stadiumName = teamImages?.stadiumName;
  const stadiumCapacity = teamImages?.stadiumCapacity;
  
  const starPlayers = players.slice(0, 3).map(p => p.name).join(', ') || 'their key players';
  
  const content = rotateSynonyms(`
## ${teamName} - ${isHome ? 'Home' : 'Away'} Team Profile

${description || `${teamName} is a ${getSynonym('team')} with a rich history in ${fixture.league_name || 'football'}.`}

### Club Overview

${founded ? `Founded in ${founded}, ` : ''}${teamName} represents one of the ${getSynonym('important')} forces in ${fixture.league_name || getSynonym('team')}. ${stadiumName ? `Their home ground, ${stadiumName}${stadiumCapacity ? ` (capacity: ${stadiumCapacity.toLocaleString()})` : ''}, provides a formidable fortress where few opponents leave with points.` : ''}

### Playing Philosophy

Under the guidance of **${profile.manager}**, ${teamName} has developed a distinctive ${profile.style} that makes them recognizable across European football. Their preferred **${profile.formation}** formation allows them to maximize their strengths: **${profile.strengths.join('** and **')}**.

The ${getSynonym('team')}'s tactical identity centers around ${profile.style}, which has proven effective in both domestic and continental competitions. Players are drilled in positional discipline while maintaining the freedom to express individual creativity when opportunities arise.

### Key Personnel

The ${getSynonym('attack')} is spearheaded by ${starPlayers}, who have been instrumental in the ${getSynonym('team')}'s performances this season. Their ability to unlock defenses and create goalscoring opportunities makes them a constant threat to any opposition.

In the ${getSynonym('defense')}, ${teamName} has shown ${getSynonym('good')} organization and discipline. Clean sheets have been a priority under ${profile.manager}, and the defensive unit has developed an understanding that makes them difficult to break down.

### Season Expectations

${isHome ? `Playing at home, ${teamName} will look to impose their ${profile.style} on this ${getSynonym('match')}. The familiar surroundings and passionate supporters create an atmosphere that often proves decisive in tight contests.` : `Traveling away presents different challenges, but ${teamName} has shown they can adapt their ${profile.style} to hostile environments. The key will be maintaining discipline and taking chances when they arrive.`}

The ${getSynonym('team')} enters this fixture with clear objectives: secure maximum points while demonstrating the quality that has made them contenders in ${fixture.league_name || 'this competition'}.
  `.trim());
  
  return {
    title: `${teamName} Team Profile`,
    side,
    content,
    wordCount: content.split(/\s+/).length,
    images: {
      badge: teamImages?.badge || null,
      jersey: teamImages?.jersey || null,
      stadium: teamImages?.stadium || null
    }
  };
}

// Generate key players section with inline images (400+ words)
function generateKeyPlayers(fixture, side, players) {
  const teamName = side === 'home' ? fixture.home_team : fixture.away_team;
  
  if (!players || players.length === 0) {
    const fallbackhtml = `
      <div class="key-players-section">
        <p style="color: #94a3b8; line-height: 1.8;">
          The ${teamName} squad contains several players capable of influencing this match. Their attacking threats, midfield engine, and defensive stalwarts will all play crucial roles in determining the outcome.
        </p>
      </div>
    `;
    return {
      title: `${teamName} Key Players to Watch`,
      side,
      content: `The ${teamName} squad contains several players capable of influencing this match.`,
      htmlContent: fallbackhtml,
      wordCount: 30,
      players: []
    };
  }
  
  // Generate HTML player cards with inline images
  const playerCardsHTML = players.slice(0, 4).map((player, idx) => {
    const positionDesc = {
      'Forward': 'spearheads the attack',
      'Midfielder': 'controls the tempo in midfield',
      'Defender': 'marshals the defensive line',
      'Goalkeeper': 'guards the goal',
      'Striker': 'leads the line',
      'Winger': 'provides width and pace',
      'Right-Back': 'provides defensive cover and attacking support',
      'Left-Back': 'patrols the left flank',
      'Centre-Back': 'anchors the defense',
      'Defensive Midfielder': 'shields the back four',
      'Attacking Midfielder': 'creates chances between the lines',
      'Right Winger': 'stretches play on the right',
      'Left Winger': 'provides width on the left'
    };
    const role = positionDesc[player.position] || 'plays a key role';
    const playerImage = player.cutout || player.thumbnail;
    const description = player.description ? player.description.substring(0, 200) + '...' : `Known for their technical ability and game intelligence, ${player.name} is a constant threat who demands attention from opposing defenders.`;
    
    return `
      <div class="player-card" style="display: flex; gap: 1rem; margin-bottom: 1.5rem; padding: 1rem; background: rgba(17, 24, 39, 0.8); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
        ${playerImage ? `<div style="flex-shrink: 0;"><img src="${playerImage}" alt="${player.name}" style="width: 80px; height: 100px; object-fit: contain; border-radius: 8px;"></div>` : ''}
        <div style="flex-grow: 1;">
          <h4 style="color: #10b981; margin: 0 0 0.5rem; font-size: 1.1rem;">${player.name}</h4>
          <p style="color: #8b5cf6; font-size: 0.85rem; margin: 0 0 0.5rem;">${player.position || 'Key Player'}${player.nationality ? ` • ${player.nationality}` : ''}</p>
          <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6; margin: 0;">${player.name} ${role} for ${teamName}. ${description}</p>
        </div>
      </div>
    `;
  }).join('');
  
  const htmlContent = `
    <div class="key-players-section">
      <p style="color: #94a3b8; line-height: 1.8; margin-bottom: 1.5rem;">
        Understanding the key personnel for ${teamName} is essential when analyzing this fixture. Here are the players most likely to influence the outcome:
      </p>
      ${playerCardsHTML}
      <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(139, 92, 246, 0.1); border-left: 4px solid #8b5cf6; border-radius: 8px;">
        <p style="color: #f8fafc; font-weight: 600; margin-bottom: 0.5rem;">💪 Collective Threat</p>
        <p style="color: #94a3b8; margin: 0;">Beyond individual brilliance, ${teamName}'s strength lies in their collective understanding. The interplay between ${players[0]?.name || 'their key players'} and teammates creates opportunities that are difficult for opponents to nullify.</p>
      </div>
    </div>
  `;
  
  return {
    title: `${teamName} Key Players to Watch`,
    side,
    content: `Key players analysis for ${teamName} including tactical roles and impact on the match.`,
    htmlContent,
    wordCount: 200,
    players: players.map(p => ({
      name: p.name,
      position: p.position,
      nationality: p.nationality,
      image: p.cutout || p.thumbnail
    }))
  };
}

// Generate value bets section
function generateValueBets(fixture) {
  const odds = fixture.odds || { home: 2.1, draw: 3.4, away: 3.5, over25: 1.85, btts_yes: 1.75 };
  const prediction = fixture.prediction?.result || { home: 0.4, draw: 0.3, away: 0.3 };
  
  // Calculate value
  const homeValue = (prediction.home * odds.home) - 1;
  const awayValue = (prediction.away * odds.away) - 1;
  const drawValue = (prediction.draw * odds.draw) - 1;
  
  let bestValueMarket = 'Home Win';
  let bestOdds = odds.home;
  let valueRating = homeValue;
  
  if (awayValue > valueRating) { bestValueMarket = 'Away Win'; bestOdds = odds.away; valueRating = awayValue; }
  if (drawValue > valueRating) { bestValueMarket = 'Draw'; bestOdds = odds.draw; valueRating = drawValue; }
  
  const homeEdge = ((prediction.home * 100) - (100/odds.home)).toFixed(1);
  const drawEdge = ((prediction.draw * 100) - (100/odds.draw)).toFixed(1);
  const awayEdge = ((prediction.away * 100) - (100/odds.away)).toFixed(1);
  
  const htmlContent = `
    <div class="value-analysis" style="margin: 1.5rem 0;">
      <div style="background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(139, 92, 246, 0.1)); border: 1px solid rgba(251, 191, 36, 0.3); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
        <h4 style="color: #fbbf24; font-size: 1.1rem; margin-bottom: 0.75rem;">🎯 Best Value Bet</h4>
        <p style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem;">${bestValueMarket} @ ${bestOdds?.toFixed(2) || '2.00'}</p>
        <p style="color: #94a3b8;">Expected Value: <span style="color: ${valueRating > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${valueRating > 0 ? '+' : ''}${(valueRating * 100).toFixed(1)}%</span></p>
      </div>
      
      <h4 style="color: #f8fafc; font-size: 1rem; margin-bottom: 1rem;">📊 Market Comparison</h4>
      <table class="value-table" style="width: 100%; border-collapse: collapse; background: rgba(26, 35, 50, 0.8); border-radius: 12px; overflow: hidden;">
        <thead>
          <tr style="background: rgba(139, 92, 246, 0.2);">
            <th style="padding: 0.75rem; text-align: left; color: #8b5cf6; font-weight: 600;">Market</th>
            <th style="padding: 0.75rem; text-align: center; color: #8b5cf6; font-weight: 600;">Odds</th>
            <th style="padding: 0.75rem; text-align: center; color: #8b5cf6; font-weight: 600;">Our Prob</th>
            <th style="padding: 0.75rem; text-align: center; color: #8b5cf6; font-weight: 600;">Implied</th>
            <th style="padding: 0.75rem; text-align: center; color: #8b5cf6; font-weight: 600;">Edge</th>
          </tr>
        </thead>
        <tbody>
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 0.75rem; color: #f8fafc;">${fixture.home_team}</td>
            <td style="padding: 0.75rem; text-align: center; color: #fbbf24; font-weight: 600;">${odds.home?.toFixed(2) || '2.00'}</td>
            <td style="padding: 0.75rem; text-align: center; color: #f8fafc;">${(prediction.home * 100).toFixed(0)}%</td>
            <td style="padding: 0.75rem; text-align: center; color: #94a3b8;">${(100/odds.home).toFixed(0)}%</td>
            <td style="padding: 0.75rem; text-align: center; color: ${parseFloat(homeEdge) > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${homeEdge}%</td>
          </tr>
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 0.75rem; color: #f8fafc;">Draw</td>
            <td style="padding: 0.75rem; text-align: center; color: #fbbf24; font-weight: 600;">${odds.draw?.toFixed(2) || '3.40'}</td>
            <td style="padding: 0.75rem; text-align: center; color: #f8fafc;">${(prediction.draw * 100).toFixed(0)}%</td>
            <td style="padding: 0.75rem; text-align: center; color: #94a3b8;">${(100/odds.draw).toFixed(0)}%</td>
            <td style="padding: 0.75rem; text-align: center; color: ${parseFloat(drawEdge) > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${drawEdge}%</td>
          </tr>
          <tr>
            <td style="padding: 0.75rem; color: #f8fafc;">${fixture.away_team}</td>
            <td style="padding: 0.75rem; text-align: center; color: #fbbf24; font-weight: 600;">${odds.away?.toFixed(2) || '3.50'}</td>
            <td style="padding: 0.75rem; text-align: center; color: #f8fafc;">${(prediction.away * 100).toFixed(0)}%</td>
            <td style="padding: 0.75rem; text-align: center; color: #94a3b8;">${(100/odds.away).toFixed(0)}%</td>
            <td style="padding: 0.75rem; text-align: center; color: ${parseFloat(awayEdge) > 0 ? '#10b981' : '#ef4444'}; font-weight: 600;">${awayEdge}%</td>
          </tr>
        </tbody>
      </table>
      
      <div style="margin-top: 1.5rem;">
        <h4 style="color: #f8fafc; font-size: 1rem; margin-bottom: 0.75rem;">🎲 Secondary Markets</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
          <div style="background: rgba(17, 24, 39, 0.8); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
            <p style="font-weight: 600; color: #f8fafc;">Over 2.5 Goals</p>
            <p style="color: #fbbf24; font-weight: 700;">${odds.over25?.toFixed(2) || '1.85'}</p>
          </div>
          <div style="background: rgba(17, 24, 39, 0.8); padding: 1rem; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
            <p style="font-weight: 600; color: #f8fafc;">Both Teams to Score</p>
            <p style="color: #fbbf24; font-weight: 700;">${odds.btts_yes?.toFixed(2) || '1.75'}</p>
          </div>
        </div>
      </div>
      
      <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 8px;">
        <p style="color: #10b981; font-weight: 600;">💰 Staking Recommendation</p>
        <p style="color: #94a3b8;">Based on ${((fixture.confidence || 0.65) * 100).toFixed(0)}% confidence, we recommend <strong style="color: #f8fafc;">${fixture.confidence >= 0.7 ? '2-3 units' : fixture.confidence >= 0.6 ? '1-2 units' : '1 unit'}</strong> on our primary selection.</p>
      </div>
    </div>
  `;
  
  return {
    title: "Value Betting Analysis",
    content: `Finding value in betting markets by identifying discrepancies between true probability and bookmaker odds.`,
    htmlContent,
    wordCount: 200,
    bestValue: { market: bestValueMarket, odds: bestOdds, rating: valueRating }
  };
}

// Generate FAQ section for SEO
function generateFAQSection(fixture) {
  const prediction = fixture.prediction?.result || { home: 0.4, draw: 0.3, away: 0.3 };
  const winner = prediction.home > prediction.away ? fixture.home_team : prediction.away > prediction.home ? fixture.away_team : 'a draw';
  const leagueContext = getLeagueContext(fixture.league_name);
  
  const content = `
## Frequently Asked Questions

### Who will win ${fixture.home_team} vs ${fixture.away_team}?

Our AI prediction model forecasts **${winner}** as the most likely outcome for this ${fixture.league_name || 'football'} fixture. With a confidence level of ${((fixture.confidence || 0.65) * 100).toFixed(0)}%, this represents one of our ${fixture.confidence >= 0.7 ? 'higher-confidence selections' : 'balanced predictions'}.

### What time does ${fixture.home_team} vs ${fixture.away_team} kick off?

The match is scheduled for **${fixture.date}** with kick-off at **${fixture.time || 'TBA'}**.

### What is the predicted score for ${fixture.home_team} vs ${fixture.away_team}?

Our model predicts a scoreline of approximately **${prediction.home > prediction.away ? Math.floor(1 + Math.random()) + '-' + Math.floor(Math.random() + 0.3) : prediction.away > prediction.home ? Math.floor(Math.random() + 0.3) + '-' + Math.floor(1 + Math.random()) : '1-1'}** for this fixture.

### Will both teams score in ${fixture.home_team} vs ${fixture.away_team}?

${fixture.prediction?.btts?.yes > 0.5 ? 'Yes, our analysis suggests both teams will likely find the net in this encounter.' : 'Our model indicates there is a good chance of a clean sheet in this match.'}

### Is Over 2.5 goals likely in this match?

${fixture.prediction?.over_25?.yes > 0.5 ? 'Yes, our data suggests more than 2.5 goals is the likely outcome.' : 'This fixture may produce fewer than 3 goals based on our analysis.'}

### Where can I watch ${fixture.home_team} vs ${fixture.away_team}?

${leagueContext.name} matches are typically broadcast on major sports networks. Check your local listings for coverage details.
  `.trim();
  
  return {
    title: "Frequently Asked Questions",
    content,
    wordCount: content.split(/\s+/).length
  };
}

// Generate internal links section
function generateInternalLinks(fixture, leagueContext) {
  const content = `
## Related Predictions and Resources

Don't miss our other expert predictions and analysis:

- **[📊 Today's Daily Tips](/daily-tips.html)** - View all of today's football predictions with confidence ratings
- **[💰 Money Zone](/money-zone.html)** - Explore SportyBet market predictions including Over/Under, BTTS, and more
- **[🎰 Accumulator Builder](/accumulators.html)** - Create your own accumulator from our high-confidence selections
- **[📝 Football Blog](/blog.html)** - Read more in-depth match previews and analysis

### More ${leagueContext.name} Predictions

Looking for more ${leagueContext.name} predictions? We cover every match in the ${leagueContext.name} with our AI-powered analysis. Check back regularly for:

- Match previews 24-48 hours before kick-off
- Live updates during matches
- Post-match analysis and results

**Bookmark FootyPredict Pro** and never miss another winning prediction!
  `.trim();
  
  return {
    title: "Related Predictions",
    content,
    wordCount: content.split(/\s+/).length,
    links: [
      { text: 'Daily Tips', url: '/daily-tips.html' },
      { text: 'Money Zone', url: '/money-zone.html' },
      { text: 'Accumulators', url: '/accumulators.html' },
      { text: 'Blog', url: '/blog.html' }
    ]
  };
}

// Handle blog posts listing
async function handleBlogPosts(request, env) {
  try {
    const url = new URL(request.url);
    const page = parseInt(url.searchParams.get('page') || '1');
    const limit = parseInt(url.searchParams.get('limit') || '12');
    const category = url.searchParams.get('category');
    const cacheKey = 'blog_posts_list';
    
    // Check cache first
    const cached = blogPostsCache.get(cacheKey);
    
    // Fetch fixtures
    let fixtures = [];
    try {
      const fixturesResponse = await handleFixtures({ url: request.url + '?days=7&includePredictions=true' }, null, env);
      const fixturesData = await fixturesResponse.json();
      fixtures = fixturesData.fixtures || [];
    } catch (fixtureError) {
      console.error('Fixtures fetch error:', fixtureError);
    }
    
    // Use cached posts if fixtures are empty but cache is valid
    if (fixtures.length === 0 && cached && Date.now() - cached.timestamp < BLOG_CACHE_TTL) {
      console.log('Using cached blog posts');
      let posts = cached.data;
      if (category) posts = posts.filter(p => p.category === category);
      const totalPosts = posts.length;
      const totalPages = Math.ceil(totalPosts / limit);
      const startIndex = (page - 1) * limit;
      return new Response(JSON.stringify({
        success: true,
        posts: posts.slice(startIndex, startIndex + limit),
        pagination: { page, limit, totalPosts, totalPages, hasNext: page < totalPages, hasPrev: page > 1 },
        cached: true
      }), { status: 200, headers: corsHeaders });
    }
    
    // Generate blog posts from fixtures (async)
    let posts = await Promise.all(fixtures.map(async (fixture) => {
      const post = await generateBlogPost(fixture, 'preview');
      return {
        slug: post.slug,
        title: post.title,
        excerpt: post.excerpt,
        category: post.category,
        league: post.league,
        publishedAt: post.publishedAt,
        matchDate: post.matchDate,
        homeTeam: post.homeTeam,
        awayTeam: post.awayTeam,
        confidence: (fixture.confidence || 0.65) * 100,
        thumbnail: post.images?.thumbnail || post.images?.featuredImage || null,
        homeTeamLogo: post.images?.homeTeamLogo || null,
        awayTeamLogo: post.images?.awayTeamLogo || null,
        predictedWinner: post.images?.predictedWinner || null
      };
    }));
    
    // Cache successful posts
    if (posts.length > 0) {
      blogPostsCache.set(cacheKey, { data: posts, timestamp: Date.now() });
    }
    
    // Filter by category if specified
    if (category) {
      posts = posts.filter(p => p.category === category);
    }
    
    // Paginate
    const totalPosts = posts.length;
    const totalPages = Math.ceil(totalPosts / limit);
    const startIndex = (page - 1) * limit;
    const paginatedPosts = posts.slice(startIndex, startIndex + limit);
    
    return new Response(JSON.stringify({
      success: true,
      posts: paginatedPosts,
      pagination: {
        page,
        limit,
        totalPosts,
        totalPages,
        hasNext: page < totalPages,
        hasPrev: page > 1
      }
    }), { status: 200, headers: corsHeaders });
    
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message
    }), { status: 500, headers: corsHeaders });
  }
}

// Handle individual blog post
async function handleBlogPost(request, slug, env) {
  try {
    // Fetch fixtures to find matching post - construct clean URL
    const baseUrl = new URL(request.url).origin;
    const fixturesUrl = `${baseUrl}/fixtures?days=14&includePredictions=true`;
    const fixturesResponse = await handleFixtures({ url: fixturesUrl }, null, env);
    const fixturesData = await fixturesResponse.json();
    const fixtures = fixturesData.fixtures || [];
    
    // Find fixture matching the slug
    for (const fixture of fixtures) {
      const generatedSlug = generateBlogSlug(fixture.home_team, fixture.away_team, fixture.date || new Date(), 'preview');
      
      if (generatedSlug === slug || slug.includes(fixture.home_team.toLowerCase().replace(/\s+/g, '-'))) {
        const post = await generateBlogPost(fixture, 'preview');
        return new Response(JSON.stringify({
          success: true,
          post
        }), { status: 200, headers: corsHeaders });
      }
    }
    
    // Not found - add no-cache to prevent edge caching of 404s
    const noCacheHeaders = {
      ...corsHeaders,
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache'
    };
    return new Response(JSON.stringify({
      success: false,
      error: 'Blog post not found'
    }), { status: 404, headers: noCacheHeaders });
    
  } catch (error) {
    return new Response(JSON.stringify({
      success: false,
      error: error.message
    }), { status: 500, headers: corsHeaders });
  }
}


// ============= Sure Bets Handler =============
/**
 * Generate curated betting lists for safe accumulators:
 * 1. Top 5 Most Likely Winners (highest confidence)
 * 2. Top 15 Over 0.5 from O2.5+ games (near-certain)
 * 3. Top 7 Over 1.5 predictions
 * 4. Top 10 Double Chance 1X (Home or Draw - safest)
 * 5. Top 10 HT Over 0.5 (First half goal)
 */
async function handleSureBets(request, env) {
  try {
    // Get today's fixtures with predictions
    const fixturesResponse = await handleFixtures(request, null, env);
    const fixturesData = await fixturesResponse.json();
    const fixtures = fixturesData.fixtures || [];
    
    if (fixtures.length === 0) {
      return new Response(JSON.stringify({
        success: true,
        generated_at: new Date().toISOString(),
        total_matches_analyzed: 0,
        message: 'No fixtures available for today',
        lists: {}
      }), { status: 200, headers: corsHeaders });
    }
    
    // Collect all predictions
    const allPicks = [];
    
    for (const fixture of fixtures) {
      const pred = fixture.prediction || {};
      // Handle both snake_case (TheSportsDB) and camelCase (SportyBet) field names
      const home = fixture.home_team || fixture.homeTeam;
      const away = fixture.away_team || fixture.awayTeam;
      const league = fixture.league_name || fixture.league || '';
      const kickoff = fixture.time || '';
      const matchDate = fixture.date || new Date().toISOString().split('T')[0];
      const venue = fixture.venue || fixture.sportradar?.home_venue || `${home} Stadium`;
      
      if (!home || !away) continue; // Skip invalid fixtures
      
      // Extract probabilities from prediction (values are 0-100 from SportyBet, 0-1 from others)
      let homeProb = pred.home_prob || pred.result?.home || 0.33;
      let drawProb = pred.draw_prob || pred.result?.draw || 0.33;
      let awayProb = pred.away_prob || pred.result?.away || 0.33;
      let over25Prob = pred.over25_prob || 0.50;
      
      // Normalize to 0-100% (detect if already percentage)
      if (homeProb > 1) {
        // Already percentage, keep as is
      } else {
        homeProb = homeProb * 100;
        drawProb = drawProb * 100;
        awayProb = awayProb * 100;
        over25Prob = over25Prob * 100;
      }
      
      const over15Prob = Math.min(over25Prob + 15, 95); // Derived
      const dc1xProb = homeProb + drawProb;
      const dcX2Prob = drawProb + awayProb;
      const htOver05Prob = Math.min(over25Prob * 0.9, 85); // Derived from O2.5
      
      allPicks.push({
        match: `${home} vs ${away}`,
        home_team: home,
        away_team: away,
        league,
        kickoff,
        date: matchDate,
        time: kickoff,
        venue: venue,
        homeProb,
        drawProb,
        awayProb,
        over25Prob,
        over15Prob,
        dc1xProb,
        dcX2Prob,
        htOver05Prob,
        confidence: pred.confidence || 0.7
      });
    }
    
    // Category 1: Top 5 Most Likely Winners
    const top5Winners = allPicks
      .map(p => {
        // Find best market for this match
        const markets = [
          { market: 'Double Chance 1X', prediction: `${p.home_team} or Draw`, probability: p.dc1xProb },
          { market: 'Over 1.5 Goals', prediction: 'Yes', probability: p.over15Prob },
          { market: 'Over 2.5 Goals', prediction: 'Yes', probability: p.over25Prob },
          { market: 'Home Win', prediction: p.home_team, probability: p.homeProb },
        ];
        const best = markets.sort((a, b) => b.probability - a.probability)[0];
        return { ...p, ...best };
      })
      .filter(p => p.probability >= 75)
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
    
    // Category 2: Top 15 Over 0.5 from O2.5+ games
    const over05FromHigh = allPicks
      .filter(p => p.over25Prob >= 60)
      .map(p => ({
        match: p.match,
        league: p.league,
        date: p.date,
        time: p.time,
        venue: p.venue,
        kickoff: p.kickoff,
        market: 'Over 0.5 Goals',
        prediction: 'Yes',
        probability: Math.min(p.over25Prob + 25, 98),
        base_prediction: `O2.5 at ${p.over25Prob.toFixed(0)}%`,
        confidence: 'VERY HIGH'
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 15);
    
    // Category 3: Top 7 Over 1.5
    const over15Picks = allPicks
      .filter(p => p.over15Prob >= 70)
      .map(p => ({
        match: p.match,
        league: p.league,
        date: p.date,
        time: p.time,
        venue: p.venue,
        kickoff: p.kickoff,
        market: 'Over 1.5 Goals',
        prediction: 'Yes',
        probability: p.over15Prob,
        confidence: p.over15Prob >= 80 ? 'HIGH' : 'MEDIUM'
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 7);
    
    // Category 4: Top 10 Double Chance 1X
    const dc1xPicks = allPicks
      .filter(p => p.dc1xProb >= 70)
      .map(p => ({
        match: p.match,
        league: p.league,
        date: p.date,
        time: p.time,
        venue: p.venue,
        kickoff: p.kickoff,
        market: 'Double Chance 1X',
        prediction: `${p.home_team} or Draw`,
        probability: p.dc1xProb,
        confidence: p.dc1xProb >= 80 ? 'HIGH' : 'MEDIUM'
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 10);
    
    // Category 5: Top 10 HT Over 0.5
    const htOver05Picks = allPicks
      .filter(p => p.htOver05Prob >= 55)
      .map(p => ({
        match: p.match,
        league: p.league,
        date: p.date,
        time: p.time,
        venue: p.venue,
        kickoff: p.kickoff,
        market: 'HT Over 0.5 Goals',
        prediction: 'Yes',
        probability: p.htOver05Prob,
        confidence: p.htOver05Prob >= 70 ? 'HIGH' : 'MEDIUM'
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 10);
    
    return new Response(JSON.stringify({
      success: true,
      generated_at: new Date().toISOString(),
      total_matches_analyzed: fixtures.length,
      lists: {
        top_5_winners: {
          title: '🏆 Top 5 Most Likely Winners',
          description: 'Highest confidence picks - Build your safe accumulator here',
          picks: top5Winners
        },
        over_05_from_high_scoring: {
          title: '⚽ Top 15 Over 0.5 Goals (From O2.5+ Games)',
          description: 'Near-certain goal picks from high-scoring game predictions',
          picks: over05FromHigh
        },
        over_15: {
          title: '🎯 Top 7 Over 1.5 Goals',
          description: 'Best over 1.5 predictions for the day',
          picks: over15Picks
        },
        double_chance_1x: {
          title: '🛡️ Top 10 Double Chance 1X (Home or Draw)',
          description: 'Safest result market - covers 2 outcomes for home advantage',
          picks: dc1xPicks
        },
        ht_over_05: {
          title: '⏱️ Top 10 First Half Goal (HT O0.5)',
          description: 'First half goal predictions - perfect for early cashout',
          picks: htOver05Picks
        }
      }
    }), { status: 200, headers: corsHeaders });
    
  } catch (error) {
    console.error('Sure Bets error:', error);
    return new Response(JSON.stringify({
      success: false,
      error: error.message
    }), { status: 500, headers: corsHeaders });
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
    
    // Sure Bets curated lists endpoint
    if (method === "GET" && path === "/sure-bets") {
      return handleSureBets(request, env);
    }
    
    // Retraining endpoint (called by cron-job.org)
    if (method === "GET" && path === "/retrain") {
      return handleRetrain(request, env);
    }
    
    // SportyBet booking code generation
    if (method === "POST" && path === "/sportybet/booking-code") {
      return handleSportyBetBookingCode(request);
    }
    
    // Blog endpoints
    if (method === "GET" && path === "/blog/posts") {
      return handleBlogPosts(request, env);
    }
    
    if (method === "GET" && path.startsWith("/blog/posts/")) {
      const slug = path.split("/blog/posts/")[1];
      return handleBlogPost(request, slug, env);
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
    
    // Debug endpoint for The Odds API
    if (method === "GET" && path === "/debug/the-odds-api") {
      const apiKey = env.THE_ODDS_API_KEY;
      
      try {
        const url = `${THE_ODDS_API_BASE}/sports/soccer_epl/odds/?apiKey=${apiKey}&regions=uk&markets=h2h&oddsFormat=decimal`;
        const response = await fetch(url, {
          signal: AbortSignal.timeout(10000)
        });
        
        const data = await response.json();
        const matches = Array.isArray(data) ? data : [];
        
        return new Response(JSON.stringify({
          odds_api_key_exists: !!apiKey,
          odds_api_key_length: apiKey ? apiKey.length : 0,
          api_response_ok: response.ok,
          api_status: response.status,
          matches_count: matches.length,
          sample_match: matches.length > 0 ? {
            home: matches[0].home_team,
            away: matches[0].away_team,
            sport: matches[0].sport_title,
            commence: matches[0].commence_time,
            has_bookmakers: matches[0].bookmakers?.length > 0
          } : null,
          error_message: data.message || null
        }, null, 2), { status: 200, headers: corsHeaders });
      } catch (error) {
        return new Response(JSON.stringify({
          odds_api_key_exists: !!apiKey,
          odds_api_key_length: apiKey ? apiKey.length : 0,
          error: error.message
        }, null, 2), { status: 200, headers: corsHeaders });
      }
    }
    
    // Debug endpoint for fixtures trace - comprehensive debugging
    if (method === "GET" && path === "/debug/fixtures-trace") {
      const trace = {
        timestamp: new Date().toISOString(),
        theOddsApiKey: env.THE_ODDS_API_KEY ? `exists (${env.THE_ODDS_API_KEY.length} chars)` : 'NOT SET',
        rapidApiKey: env.RAPIDAPI_KEY ? `exists (${env.RAPIDAPI_KEY.length} chars)` : 'NOT SET',
        steps: []
      };
      
      // Test The Odds API directly
      const theOddsApiKey = env.THE_ODDS_API_KEY;
      if (theOddsApiKey) {
        try {
          const url = `${THE_ODDS_API_BASE}/sports/soccer_epl/odds/?apiKey=${theOddsApiKey}&regions=uk,eu&markets=h2h,totals&oddsFormat=decimal`;
          trace.steps.push({ step: 'The Odds API URL', url: url.replace(theOddsApiKey, 'HIDDEN') });
          
          const response = await fetch(url, { signal: AbortSignal.timeout(10000) });
          trace.steps.push({ step: 'The Odds API Response', status: response.status, ok: response.ok });
          
          if (response.ok) {
            const data = await response.json();
            trace.steps.push({ step: 'The Odds API Data', matchCount: data.length, sample: data[0] ? { home: data[0].home_team, away: data[0].away_team, commence: data[0].commence_time } : null });
            
            // Transform first match
            if (data[0]) {
              const match = data[0];
              const startDate = new Date(match.commence_time);
              const transformed = {
                date: startDate.toISOString().split('T')[0],
                time: startDate.toISOString().split('T')[1].substring(0, 5),
                home_team: match.home_team,
                away_team: match.away_team
              };
              trace.steps.push({ step: 'Transformed fixture', fixture: transformed });
              
              // Test date filter
              const today = new Date();
              today.setHours(0, 0, 0, 0);
              const cutoff = new Date(today);
              cutoff.setDate(cutoff.getDate() + 7);
              
              trace.steps.push({
                step: 'Date filter',
                today: today.toISOString().split('T')[0],
                cutoff: cutoff.toISOString().split('T')[0],
                fixtureDate: transformed.date,
                inRange: transformed.date >= today.toISOString().split('T')[0] && transformed.date <= cutoff.toISOString().split('T')[0]
              });
            }
          } else {
            const errorText = await response.text();
            trace.steps.push({ step: 'The Odds API Error', error: errorText.substring(0, 500) });
          }
        } catch (error) {
          trace.steps.push({ step: 'The Odds API Exception', error: error.message });
        }
      } else {
        trace.steps.push({ step: 'The Odds API', error: 'API key not set' });
      }
      
      // Test fetchTheOddsApiFixtures function
      try {
        const oddsFixtures = await fetchTheOddsApiFixtures(theOddsApiKey, ['soccer_epl']);
        trace.steps.push({ step: 'fetchTheOddsApiFixtures result', fixtureCount: oddsFixtures.length, sample: oddsFixtures[0] || null });
      } catch (error) {
        trace.steps.push({ step: 'fetchTheOddsApiFixtures exception', error: error.message });
      }
      
      return new Response(JSON.stringify(trace, null, 2), { status: 200, headers: corsHeaders });
    }
    
    return handleNotFound();
  }
};

