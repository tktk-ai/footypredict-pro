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

// ============= Fixtures Fetcher =============

// OpenLigaDB - Free API for German leagues with real upcoming fixtures
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
function predictMatch(homeTeam, awayTeam, options = {}) {
  const timestamp = new Date().toISOString();
  const matchId = `${homeTeam.toLowerCase().replace(/\s+/g, '_')}_vs_${awayTeam.toLowerCase().replace(/\s+/g, '_')}_${Date.now()}`;
  
  // Use odds if available to improve predictions
  let homeProb, drawProb, awayProb;
  
  if (options.home_odds && options.draw_odds && options.away_odds) {
    // Convert odds to probabilities
    const totalInv = 1/options.home_odds + 1/options.draw_odds + 1/options.away_odds;
    homeProb = (1/options.home_odds) / totalInv;
    drawProb = (1/options.draw_odds) / totalInv;
    awayProb = (1/options.away_odds) / totalInv;
  } else {
    // Base probabilities with home advantage
    const homeAdvantage = 0.10;
    homeProb = 0.35 + homeAdvantage;
    drawProb = 0.28;
    awayProb = 0.27;
    
    const total = homeProb + drawProb + awayProb;
    homeProb /= total;
    drawProb /= total;
    awayProb /= total;
  }
  
  // Goals predictions based on league averages
  const over25Prob = 0.52 + (Math.random() * 0.08 - 0.04);
  const bttsProb = 0.48 + (Math.random() * 0.08 - 0.04);
  
  // Confidence based on data availability
  const confidence = options.home_odds ? 0.75 : 0.65;
  
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
async function handleFixtures(request, specificLeague = null) {
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
  
  // Fetch today's LIVE matches for 24/7 coverage (runs in parallel)
  const todayPromise = includeToday ? fetchTodaysMatches() : Promise.resolve([]);
  const yesterdayPromise = includeRecent ? fetchYesterdaysMatches() : Promise.resolve([]);
  const fetchPromises = Object.entries(leagues).map(async ([leagueId, leagueInfo]) => {
    const sportsDbId = SPORTSDB_LEAGUES[leagueId];
    const fixtures = [];
    
    if (sportsDbId) {
      try {
        const matches = await fetchTheSportsDBFixtures(sportsDbId);
        
        for (const match of matches) {
          if (!match.date) continue;
          
          // Use string comparison for dates (YYYY-MM-DD format) to avoid timezone issues
          const todayStr = today.toISOString().split('T')[0];
          const cutoffStr = cutoff.toISOString().split('T')[0];
          const matchDateStr = match.date;
          
          // Include scheduled matches within date range
          if (match.status === 'scheduled' && matchDateStr >= todayStr && matchDateStr <= cutoffStr) {
            // Use league name from API if available (more accurate), else use our mapping
            const actualLeagueName = match.league_from_api || leagueInfo.name;
            const actualCountry = match.country_from_api || leagueInfo.country;
            
            const fixture = {
              ...match,
              league: leagueId,
              league_name: actualLeagueName,
              country: actualCountry
            };
            
            // Add predictions
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
    
    // Also get recent results from Football-Data.co.uk if requested
    if (includeRecent && leagueInfo.file) {
      try {
        const historicalMatches = await fetchFootballDataCSV(leagueInfo.file);
        const finished = historicalMatches
          .filter(m => m.status === 'finished')
          .slice(-10); // Last 10 results
        
        for (const match of finished) {
          fixtures.push({
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
    
    return fixtures;
  });
  
  // Wait for all fetches including today's matches
  const [results, todayMatches, yesterdayMatches] = await Promise.all([
    Promise.all(fetchPromises),
    todayPromise,
    yesterdayPromise
  ]);
  
  // Add scheduled fixtures from leagues
  results.forEach(fixtures => allFixtures.push(...fixtures));
  
  // Add today's live/scheduled matches with predictions
  for (const match of todayMatches) {
    if (includePredictions && match.status !== 'finished') {
      const prediction = predictMatch(match.home_team, match.away_team, {
        league_name: match.league_name,
        date: match.date,
        time: match.time
      });
      match.prediction = prediction.predictions;
      match.confidence = prediction.confidence;
    }
    allFixtures.push(match);
  }
  
  // Add yesterday's results if requested
  yesterdayMatches.forEach(match => allFixtures.push(match));
  
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
    data_source: "Football-Data.co.uk",
    prediction_features: ["odds-based", "home-advantage", "combo-generation"],
    timestamp: new Date().toISOString()
  }), { status: 200, headers: corsHeaders });
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
      return handleFixtures(request);
    }
    
    if (method === "GET" && path.startsWith("/fixtures/")) {
      const league = path.split("/")[2];
      return handleFixtures(request, league);
    }
    
    if (method === "GET" && path === "/leagues") {
      return handleLeagues();
    }
    
    // Prediction endpoint
    if (method === "POST" && path === "/predict") {
      return handlePredict(request);
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
          predict: "POST /predict"
        },
        timestamp: new Date().toISOString()
      }), { status: 200, headers: corsHeaders });
    }
    
    return handleNotFound();
  }
};

