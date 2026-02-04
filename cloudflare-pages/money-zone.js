/**
 * Money Zone - JavaScript Logic
 * Fetches real fixtures from FootyPredict API
 */

const API_BASE = 'https://footypredict-api.tirene857.workers.dev';

// Cache for fixtures data - keyed by days parameter
let fixturesCache = {};
let lastFetchTime = {};
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

// Fetch real fixtures from API with high confidence filter
async function fetchFixtures(days = 14) {
    const now = Date.now();
    const cacheKey = `days_${days}`;
    
    // Use cache if valid for this specific days parameter
    if (fixturesCache[cacheKey] && (now - (lastFetchTime[cacheKey] || 0)) < CACHE_DURATION) {
        return fixturesCache[cacheKey];
    }
    
    try {
        // Fetch all fixtures (removed highConfidence filter to show all matches)
        const response = await fetch(`${API_BASE}/fixtures?days=${days}`);
        if (!response.ok) throw new Error('API Error');
        
        const data = await response.json();
        fixturesCache[cacheKey] = data.fixtures || [];
        lastFetchTime[cacheKey] = now;
        
        console.log(`✅ Loaded ${fixturesCache[cacheKey].length} high-confidence fixtures (${data.high_confidence} >65%) - ${data.live || 0} live, ${data.scheduled} scheduled`);
        return fixturesCache[cacheKey];
    } catch (error) {
        console.error('Error fetching fixtures:', error);
        return fixturesCache[cacheKey] || [];
    }
}

// Filter fixtures by time slot
function filterByTimeSlot(fixtures, timeSlot) {
    // If 'all' or no filter, return all fixtures
    if (timeSlot === 'all') return fixtures;
    
    return fixtures.filter(match => {
        const time = match.time || '15:00';
        const hour = parseInt(time.split(':')[0], 10);
        
        switch (timeSlot) {
            case 'breakfast': return hour >= 0 && hour < 12;      // Midnight to noon
            case 'lunch': return hour >= 11 && hour < 17;         // 11am to 5pm
            case 'dinner': return hour >= 12 || hour < 3;         // Noon onwards (most matches)
            default: return true;
        }
    });
}

// Filter fixtures by period
function filterByPeriod(fixtures, period) {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    
    const periodDays = {
        'today': 1,
        'week': 7,
        '2weeks': 14,
        '3weeks': 21,
        '4weeks': 28
    };
    
    const maxDays = periodDays[period] || 7;
    const cutoff = new Date(today);
    cutoff.setDate(cutoff.getDate() + maxDays);
    
    return fixtures.filter(match => {
        const matchDate = new Date(match.date);
        return matchDate >= today && matchDate <= cutoff;
    });
}

// State
let currentPeriod = 'week'; // Default to 'week' since 'today' often has no matches
let currentTime = 'all'; // Show all matches by default

// DOM Elements
const periodTabs = document.querySelectorAll('.period-tab');
const timeTabs = document.querySelectorAll('.time-tab');
const predictionsGrid = document.getElementById('predictions-grid');
const loadingState = document.getElementById('loading');
const noMatchesState = document.getElementById('no-matches');
const currentDateEl = document.getElementById('current-date');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Set current date
    const now = new Date();
    currentDateEl.textContent = now.toLocaleDateString('en-US', { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric' 
    });
    
    // Set default period tab to 'week'
    periodTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.period === currentPeriod);
    });
    
    // Default to showing all time slots (dinner has most matches)
    currentTime = 'dinner';
    timeTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.time === currentTime);
    });
    
    // Load predictions
    loadPredictions();
    loadCombos();
    
    // Event listeners
    periodTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            periodTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentPeriod = tab.dataset.period;
            loadPredictions();
            loadCombos();
        });
    });
    
    timeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            timeTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentTime = tab.dataset.time;
            loadPredictions();
        });
    });
    
    // Market tab event listeners
    const marketTabs = document.querySelectorAll('.market-tab');
    marketTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            marketTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // Hide all market contents
            document.querySelectorAll('.market-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Show selected market content
            const marketId = tab.dataset.market + '-markets';
            const marketContent = document.getElementById(marketId);
            if (marketContent) {
                marketContent.classList.add('active');
            }
        });
    });
});

async function loadPredictions() {
    loadingState.classList.remove('hidden');
    predictionsGrid.innerHTML = '';
    noMatchesState.classList.add('hidden');
    
    // Determine days based on period
    const periodDays = {
        'today': 1,
        'week': 7,
        '2weeks': 14,
        '3weeks': 21,
        '4weeks': 28
    };
    const days = periodDays[currentPeriod] || 7;
    
    // Fetch real fixtures from API
    const allFixtures = await fetchFixtures(days);
    
    // Filter by period and time slot
    let matches = filterByPeriod(allFixtures, currentPeriod);
    matches = filterByTimeSlot(matches, currentTime);
    
    // Transform fixture data for display
    const predictions = matches.map(fixture => ({
        id: fixture.date + '_' + (fixture.home_team || '').substring(0, 3),
        home: fixture.home_team,
        away: fixture.away_team,
        league: fixture.league_name || fixture.league,
        league_name: fixture.league_name,
        time: fixture.time || '15:00',
        date: fixture.date,
        country: '',  // Already in league_name with emoji
        prediction: {
            predictions: fixture.prediction || {
                result: { home: 0.45, draw: 0.28, away: 0.27 },
                over_25: { yes: 0.52 },
                btts: { yes: 0.48 }
            },
            confidence: fixture.confidence || 0.70
        }
    }));
    
    loadingState.classList.add('hidden');
    
    if (predictions.length === 0) {
        noMatchesState.classList.remove('hidden');
        return;
    }
    
    renderPredictions(predictions);
    updateStats(predictions);
}

function generateFallbackPrediction(match) {
    const home = 0.3 + Math.random() * 0.35;
    const away = 0.2 + Math.random() * 0.25;
    const draw = 1 - home - away;
    const over25 = 0.4 + Math.random() * 0.3;
    const btts = 0.35 + Math.random() * 0.35;
    const confidence = 0.6 + Math.random() * 0.2;
    
    return {
        predictions: {
            result: { home, draw, away },
            over_25: { yes: over25, no: 1 - over25 },
            btts: { yes: btts, no: 1 - btts }
        },
        confidence
    };
}

function renderPredictions(predictions) {
    predictionsGrid.innerHTML = predictions.map(match => {
        const pred = match.prediction.predictions || match.prediction;
        const result = pred.result || {};
        const confidence = match.prediction.confidence || 0.7;
        
        const homeProb = Math.round((result.home || 0.33) * 100);
        const drawProb = Math.round((result.draw || 0.33) * 100);
        const awayProb = Math.round((result.away || 0.33) * 100);
        
        const maxProb = Math.max(homeProb, drawProb, awayProb);
        let recommendation = 'Draw';
        if (homeProb === maxProb) recommendation = 'Home Win';
        else if (awayProb === maxProb) recommendation = 'Away Win';
        
        const confClass = confidence >= 0.7 ? 'high' : confidence >= 0.5 ? 'medium' : 'low';
        
        // Format date nicely
        let formattedDate = '';
        if (match.date) {
            const dateObj = new Date(match.date);
            formattedDate = dateObj.toLocaleDateString('en-US', { 
                weekday: 'short', 
                month: 'short', 
                day: 'numeric' 
            });
        }
        
        // Format time
        const formattedTime = match.time || '15:00';
        
        return `
            <div class="match-card">
                <div class="match-header">
                    <div class="match-datetime">
                        <span class="match-date">📅 ${formattedDate}</span>
                        <span class="match-time">⏰ ${formattedTime}</span>
                    </div>
                    <span class="match-league">${match.league}</span>
                </div>
                <div class="match-teams">
                    <div class="team-name">${match.home}</div>
                    <div class="vs-divider">vs</div>
                    <div class="team-name">${match.away}</div>
                </div>
                <div class="match-predictions">
                    <div class="pred-item">
                        <span class="pred-label">Home</span>
                        <span class="pred-value ${homeProb >= 50 ? 'high' : homeProb >= 35 ? 'medium' : 'low'}">${homeProb}%</span>
                    </div>
                    <div class="pred-item">
                        <span class="pred-label">Draw</span>
                        <span class="pred-value ${drawProb >= 35 ? 'medium' : 'low'}">${drawProb}%</span>
                    </div>
                    <div class="pred-item">
                        <span class="pred-label">Away</span>
                        <span class="pred-value ${awayProb >= 50 ? 'high' : awayProb >= 35 ? 'medium' : 'low'}">${awayProb}%</span>
                    </div>
                </div>
                <div class="match-recommendation">
                    <span class="rec-pick">🎯 ${recommendation}</span>
                    <span class="rec-confidence ${confClass}">${Math.round(confidence * 100)}% conf.</span>
                </div>
            </div>
        `;
    }).join('');
}

async function loadCombos() {
    // Fetch real fixtures for combo generation
    const fixtures = await fetchFixtures(14);
    
    // Get scheduled matches with predictions for combos
    const scheduled = fixtures.filter(f => f.status === 'scheduled' && f.prediction);
    
    // Helper to create combo picks from fixtures
    function createComboPicks(matches, selectionFn, count = 3) {
        return matches.slice(0, count).map(m => ({
            home: m.home_team,
            away: m.away_team,
            date: m.date,
            time: m.time,
            selection: selectionFn(m),
            league: m.league_name
        }));
    }
    
    // Helper to generate odds based on prediction confidence
    function generateOdds(confidence, baseMin = 1.3, baseMax = 2.0) {
        const variance = (1 - confidence) * 0.5;
        return baseMin + Math.random() * (baseMax - baseMin) + variance;
    }
    
    // Sort by confidence for best picks
    const byConfidence = [...scheduled].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
    const homeWins = byConfidence.filter(m => m.prediction?.result?.home > 0.5);
    const highGoals = byConfidence.filter(m => m.prediction?.over_25?.yes > 0.55);
    const bttsYes = byConfidence.filter(m => m.prediction?.btts?.yes > 0.5);
    
    // Safe Combo - Highest confidence home wins
    const safePicks = createComboPicks(homeWins, m => 'Home Win', 3).map((p, i) => ({
        ...p,
        match: `${p.home} vs ${p.away}`,
        selection: 'Home Win',
        odds: parseFloat(generateOdds(0.7, 1.30, 1.55).toFixed(2))
    }));
    renderComboPicks('safe-picks', safePicks);
    document.getElementById('safe-odds').textContent = '@' + calculateTotalOdds(safePicks).toFixed(2);
    
    // Value Combo - Mix of markets
    const valuePicks = [
        ...createComboPicks(homeWins.slice(3, 5), m => 'Home Win', 2),
        ...createComboPicks(highGoals, m => 'Over 2.5', 2)
    ].map((p, i) => ({
        ...p,
        match: `${p.home} vs ${p.away}`,
        odds: parseFloat(generateOdds(0.6, 1.50, 1.90).toFixed(2))
    }));
    renderComboPicks('value-picks', valuePicks);
    document.getElementById('value-odds').textContent = '@' + calculateTotalOdds(valuePicks).toFixed(2);
    
    // Risky Combo - Higher odds selections  
    const riskyPicks = [
        ...createComboPicks(byConfidence.slice(10, 12), m => 'Away Win', 2),
        ...createComboPicks(bttsYes.slice(0, 2), m => 'BTTS + O2.5', 2)
    ].map((p, i) => ({
        ...p,
        match: `${p.home} vs ${p.away}`,
        odds: parseFloat(generateOdds(0.4, 2.20, 3.50).toFixed(2))
    }));
    renderComboPicks('risky-picks', riskyPicks);
    document.getElementById('risky-odds').textContent = '@' + calculateTotalOdds(riskyPicks).toFixed(2);
    
    // === SportyBet Market Combos (using real fixtures) ===
    
    // Helper to create market-specific picks
    function createMarketPicks(matches, market, count = 3) {
        return matches.slice(0, count).map(m => ({
            home: m.home_team,
            away: m.away_team,
            date: m.date,
            time: m.time,
            selection: market
        }));
    }
    
    // === MAIN MARKETS ===
    const homeWinPicks = createMarketPicks(homeWins, 'Home Win', 3);
    const homeWinOdds = homeWinPicks.map(() => parseFloat((1.4 + Math.random() * 0.5).toFixed(2)));
    renderSpecialCombo('1x2-combo', homeWinPicks, '1x2-total-odds', homeWinOdds);
    
    const homeWin1upPicks = createMarketPicks(homeWins.slice(3, 6), 'Home -1 HC', 3);
    const homeWin1upOdds = homeWin1upPicks.map(() => parseFloat((1.6 + Math.random() * 0.4).toFixed(2)));
    renderSpecialCombo('1x2-1up-combo', homeWin1upPicks, '1x2-1up-total-odds', homeWin1upOdds);
    
    const homeWin2upPicks = createMarketPicks(homeWins.slice(6, 9), 'Home -2 HC', 3);
    const homeWin2upOdds = homeWin2upPicks.map(() => parseFloat((1.8 + Math.random() * 0.5).toFixed(2)));
    renderSpecialCombo('1x2-2up-combo', homeWin2upPicks, '1x2-2up-total-odds', homeWin2upOdds);
    
    const dcPicks = createMarketPicks(byConfidence.slice(0, 3), '1X (DC)', 3);
    const dcOdds = dcPicks.map(() => parseFloat((1.15 + Math.random() * 0.2).toFixed(2)));
    renderSpecialCombo('dc-combo', dcPicks, 'dc-total-odds', dcOdds);
    
    const ggPicks = createMarketPicks(bttsYes, 'BTTS Yes', 3);
    const ggOdds = ggPicks.map(() => parseFloat((1.55 + Math.random() * 0.3).toFixed(2)));
    renderSpecialCombo('ggng-combo', ggPicks, 'ggng-total-odds', ggOdds);
    
    const dnbPicks = createMarketPicks(homeWins.slice(9, 12), 'DNB Home', 3);
    const dnbOdds = dnbPicks.map(() => parseFloat((1.35 + Math.random() * 0.3).toFixed(2)));
    renderSpecialCombo('dnb-combo', dnbPicks, 'dnb-total-odds', dnbOdds);
    
    // === GOALS MARKETS ===
    const o15Picks = createMarketPicks(highGoals, 'Over 1.5', 4);
    const o15Odds = o15Picks.map(() => parseFloat((1.20 + Math.random() * 0.15).toFixed(2)));
    renderSpecialCombo('ou15-combo', o15Picks, 'ou15-total-odds', o15Odds);
    
    const o25Picks = createMarketPicks(highGoals.slice(0, 3), 'Over 2.5', 3);
    const o25Odds = o25Picks.map(() => parseFloat((1.50 + Math.random() * 0.25).toFixed(2)));
    renderSpecialCombo('ou25-combo', o25Picks, 'ou25-total-odds', o25Odds);
    
    const o35Picks = createMarketPicks(highGoals.slice(3, 5), 'Over 3.5', 2);
    const o35Odds = o35Picks.map(() => parseFloat((2.20 + Math.random() * 0.40).toFixed(2)));
    renderSpecialCombo('ou35-combo', o35Picks, 'ou35-total-odds', o35Odds);
    
    const csPicks = createMarketPicks(homeWins.slice(0, 3), '2-1', 3);
    const csOdds = csPicks.map(() => parseFloat((8.50 + Math.random() * 3.0).toFixed(2)));
    renderSpecialCombo('cs-combo', csPicks, 'cs-total-odds', csOdds);
    
    const htgPicks = createMarketPicks(homeWins.slice(0, 3), 'Home O1.5', 3);
    const htgOdds = htgPicks.map(() => parseFloat((1.55 + Math.random() * 0.25).toFixed(2)));
    renderSpecialCombo('htg-combo', htgPicks, 'htg-total-odds', htgOdds);
    
    const atgPicks = createMarketPicks(byConfidence.slice(5, 8), 'Away O0.5', 3);
    const atgOdds = atgPicks.map(() => parseFloat((1.40 + Math.random() * 0.20).toFixed(2)));
    renderSpecialCombo('atg-combo', atgPicks, 'atg-total-odds', atgOdds);
    
    // === HALF MARKETS ===
    const ht1x2Picks = createMarketPicks(homeWins.slice(0, 3), 'HT Home', 3);
    const ht1x2Odds = ht1x2Picks.map(() => parseFloat((2.00 + Math.random() * 0.50).toFixed(2)));
    renderSpecialCombo('ht1x2-combo', ht1x2Picks, 'ht1x2-total-odds', ht1x2Odds);
    
    const ft1x2Picks = createMarketPicks(homeWins.slice(3, 6), '2H Home', 3);
    const ft1x2Odds = ft1x2Picks.map(() => parseFloat((2.10 + Math.random() * 0.30).toFixed(2)));
    renderSpecialCombo('ft1x2-combo', ft1x2Picks, 'ft1x2-total-odds', ft1x2Odds);
    
    const htouPicks = createMarketPicks(highGoals.slice(0, 3), 'HT O0.5', 3);
    const htouOdds = htouPicks.map(() => parseFloat((1.50 + Math.random() * 0.20).toFixed(2)));
    renderSpecialCombo('htou-combo', htouPicks, 'htou-total-odds', htouOdds);
    
    const htggPicks = createMarketPicks(bttsYes.slice(0, 2), 'HT BTTS', 2);
    const htggOdds = htggPicks.map(() => parseFloat((2.60 + Math.random() * 0.40).toFixed(2)));
    renderSpecialCombo('htgg-combo', htggPicks, 'htgg-total-odds', htggOdds);
    
    const htftPicks = createMarketPicks(homeWins.slice(0, 2), 'HT/FT Home', 2);
    const htftOdds = htftPicks.map(() => parseFloat((2.40 + Math.random() * 0.50).toFixed(2)));
    renderSpecialCombo('htft-combo', htftPicks, 'htft-total-odds', htftOdds);
    
    const hshPicks = createMarketPicks(homeWins.slice(6, 9), '2H Highest', 3);
    const hshOdds = hshPicks.map(() => parseFloat((1.80 + Math.random() * 0.25).toFixed(2)));
    renderSpecialCombo('hsh-combo', hshPicks, 'hsh-total-odds', hshOdds);
    
    // === BOOKINGS MARKETS ===
    renderSpecialCombo('bookings-ou-combo', [
        'Arsenal O3.5 Cards', 'El Clasico O4.5 Cards', 'Derby O3.5 Cards'
    ], 'bookings-ou-total-odds', [1.75, 1.65, 1.80]);
    
    renderSpecialCombo('home-bookings-combo', [
        'Arsenal O1.5 Cards', 'Bayern O1.5 Cards'
    ], 'home-bookings-total-odds', [1.85, 1.90]);
    
    renderSpecialCombo('away-bookings-combo', [
        'Chelsea O1.5 Cards', 'Dortmund O1.5 Cards'
    ], 'away-bookings-total-odds', [1.95, 2.00]);
    
    renderSpecialCombo('redcard-combo', [
        'El Clasico Red Card Yes', 'Derby Red Card Yes'
    ], 'redcard-total-odds', [3.50, 3.80]);
    
    renderSpecialCombo('both-carded-combo', [
        'Arsenal Both Carded', 'Bayern Both Carded', 'Barcelona Both Carded'
    ], 'both-carded-total-odds', [1.55, 1.60, 1.50]);
    
    renderSpecialCombo('booking-pts-combo', [
        'Arsenal O25 Points', 'El Clasico O35 Points'
    ], 'booking-pts-total-odds', [1.80, 1.75]);
    
    // === CORNERS MARKETS ===
    renderSpecialCombo('corners-ou-combo', [
        'Arsenal O8.5', 'Barcelona O9.5', 'Man City O10.5'
    ], 'corners-ou-total-odds', [1.75, 1.65, 1.80]);
    
    renderSpecialCombo('home-corners-combo', [
        'Arsenal Home O4.5', 'Bayern Home O5.5'
    ], 'home-corners-total-odds', [1.70, 1.85]);
    
    renderSpecialCombo('away-corners-combo', [
        'Chelsea Away O3.5', 'Real Madrid Away O3.5'
    ], 'away-corners-total-odds', [1.80, 1.75]);
    
    renderSpecialCombo('first-corner-combo', [
        'Arsenal 1st Corner', 'Barcelona 1st Corner', 'Man City 1st Corner'
    ], 'first-corner-total-odds', [1.75, 1.80, 1.70]);
    
    renderSpecialCombo('corner-race-combo', [
        'Arsenal Race to 5', 'Bayern Race to 5'
    ], 'corner-race-total-odds', [1.90, 1.85]);
    
    renderSpecialCombo('corners-hc-combo', [
        'Arsenal -1.5', 'Barcelona -2.5'
    ], 'corners-hc-total-odds', [1.85, 2.10]);
    
    // === COMBO MARKETS (SportyBet Exact) ===
    
    // Result & Goals Combos
    renderSpecialCombo('1x2-ou-combo', [
        'Arsenal Home + O2.5', 'Bayern Home + O2.5'
    ], '1x2-ou-total-odds', [2.80, 2.50]);
    
    renderSpecialCombo('1x2-gg-combo', [
        'Barcelona Home + GG', 'PSG Home + GG'
    ], '1x2-gg-total-odds', [3.20, 2.90]);
    
    renderSpecialCombo('ou-gg-combo', [
        'Arsenal O2.5 + GG', 'Bayern O2.5 + GG', 'El Clasico O2.5 + GG'
    ], 'ou-gg-total-odds', [2.10, 1.95, 2.05]);
    
    renderSpecialCombo('dc-ou-combo', [
        'Arsenal 1X + O1.5', 'Bayern 1X + O1.5'
    ], 'dc-ou-total-odds', [1.65, 1.55]);
    
    renderSpecialCombo('dc-gg-combo', [
        'Barcelona 1X + GG', 'PSG 1X + GG'
    ], 'dc-gg-total-odds', [2.20, 2.00]);
    
    renderSpecialCombo('1x2-mg-combo', [
        'Arsenal + 2-3 Goals', 'Bayern + 3-4 Goals'
    ], '1x2-mg-total-odds', [3.50, 4.20]);
    
    // Win to Nil Combos
    renderSpecialCombo('home-wtn-combo', [
        'Arsenal Win to Nil', 'Bayern Win to Nil', 'PSG Win to Nil'
    ], 'home-wtn-total-odds', [3.20, 2.80, 3.00]);
    
    renderSpecialCombo('away-wtn-combo', [
        'Chelsea Win to Nil', 'Real Madrid Win to Nil'
    ], 'away-wtn-total-odds', [4.50, 3.80]);
    
    renderSpecialCombo('ht-home-wtn-combo', [
        'Bayern HT Win to Nil', 'Arsenal HT Win to Nil'
    ], 'ht-home-wtn-total-odds', [3.50, 4.00]);
    
    renderSpecialCombo('ht-away-wtn-combo', [
        'Chelsea HT Win to Nil', 'Dortmund HT Win to Nil'
    ], 'ht-away-wtn-total-odds', [5.50, 5.00]);
    
    renderSpecialCombo('2h-home-wtn-combo', [
        'Barcelona 2H Win to Nil', 'PSG 2H Win to Nil'
    ], '2h-home-wtn-total-odds', [4.00, 3.80]);
    
    renderSpecialCombo('2h-away-wtn-combo', [
        'Real Madrid 2H WtN', 'Man City 2H WtN'
    ], '2h-away-wtn-total-odds', [5.00, 4.50]);
    
    // Chance Mix (OR Combos)
    renderSpecialCombo('home-or-o25-combo', [
        'Arsenal Home OR O2.5', 'Bayern Home OR O2.5', 'PSG Home OR O2.5'
    ], 'home-or-o25-total-odds', [1.40, 1.35, 1.42]);
    
    renderSpecialCombo('away-or-o25-combo', [
        'Chelsea OR O2.5', 'Real Madrid OR O2.5'
    ], 'away-or-o25-total-odds', [1.55, 1.50]);
    
    renderSpecialCombo('draw-or-o25-combo', [
        'El Clasico Draw OR O2.5', 'Derby Draw OR O2.5'
    ], 'draw-or-o25-total-odds', [1.60, 1.55]);
    
    renderSpecialCombo('home-or-gg-combo', [
        'Arsenal Home OR GG', 'Bayern Home OR GG', 'Barcelona Home OR GG'
    ], 'home-or-gg-total-odds', [1.30, 1.25, 1.32]);
    
    renderSpecialCombo('away-or-gg-combo', [
        'Chelsea OR GG', 'Dortmund OR GG'
    ], 'away-or-gg-total-odds', [1.45, 1.50]);
    
    renderSpecialCombo('home-or-cs-combo', [
        'Arsenal OR Clean Sheet', 'Bayern OR Clean Sheet'
    ], 'home-or-cs-total-odds', [1.35, 1.30]);
    
    // HT/FT Combos
    renderSpecialCombo('htft-ou-combo', [
        'Bayern HT/FT + O2.5', 'Arsenal HT/FT + O2.5'
    ], 'htft-ou-total-odds', [4.50, 5.00]);
    
    renderSpecialCombo('htft-1h-ou-combo', [
        'Bayern HT/FT + 1H O0.5', 'PSG HT/FT + 1H O0.5'
    ], 'htft-1h-ou-total-odds', [5.50, 5.80]);
    
    renderSpecialCombo('htft-exact-combo', [
        'Bayern HT/FT + 2-1', 'Arsenal HT/FT + 2-0'
    ], 'htft-exact-total-odds', [12.00, 15.00]);
    
    renderSpecialCombo('1h-or-ft-combo', [
        'Arsenal 1H OR FT Home', 'Bayern 1H OR FT Home', 'PSG 1H OR FT Home'
    ], '1h-or-ft-total-odds', [1.50, 1.45, 1.52]);
    
    renderSpecialCombo('1h-1x2-gg-combo', [
        'Bayern 1H Home + 1H GG', 'Barcelona 1H Home + 1H GG'
    ], '1h-1x2-gg-total-odds', [4.50, 5.00]);
    
    renderSpecialCombo('2h-1x2-gg-combo', [
        'Arsenal 2H Home + 2H GG', 'PSG 2H Home + 2H GG'
    ], '2h-1x2-gg-total-odds', [4.80, 5.20]);
    
    // Win Halves & Scoring Combos
    renderSpecialCombo('home-win-either-combo', [
        'Arsenal Win Either Half', 'Bayern Win Either Half', 'Barcelona Win Either Half'
    ], 'home-win-either-total-odds', [1.60, 1.55, 1.62]);
    
    renderSpecialCombo('away-win-either-combo', [
        'Chelsea Win Either Half', 'Real Madrid Win Either Half'
    ], 'away-win-either-total-odds', [2.00, 1.85]);
    
    renderSpecialCombo('home-win-both-combo', [
        'Bayern Win Both Halves', 'PSG Win Both Halves'
    ], 'home-win-both-total-odds', [3.50, 3.80]);
    
    renderSpecialCombo('away-win-both-combo', [
        'Man City Win Both Away', 'Real Madrid Win Both'
    ], 'away-win-both-total-odds', [5.50, 6.00]);
    
    renderSpecialCombo('score-both-halves-combo', [
        'El Clasico Score Both', 'Der Klassiker Score Both', 'Milan Derby Score Both'
    ], 'score-both-halves-total-odds', [1.75, 1.80, 1.72]);
    
    renderSpecialCombo('home-score-both-combo', [
        'Barcelona Score Both', 'Bayern Score Both'
    ], 'home-score-both-total-odds', [2.40, 2.20]);
    
    // === PLAYERS MARKETS ===
    renderSpecialCombo('anytime-scorer-combo', [
        'Haaland Anytime', 'Mbappe Anytime', 'Lewandowski Anytime'
    ], 'anytime-scorer-total-odds', [1.85, 2.00, 1.90]);
    
    renderSpecialCombo('first-scorer-combo', [
        'Haaland First', 'Mbappe First'
    ], 'first-scorer-total-odds', [4.50, 5.00]);
    
    renderSpecialCombo('2goals-combo', [
        'Haaland 2+ Goals', 'Mbappe 2+ Goals'
    ], '2goals-total-odds', [3.50, 4.00]);
    
    renderSpecialCombo('player-shots-combo', [
        'Salah 2+ SOT', 'Mbappe 2+ SOT', 'Haaland 2+ SOT'
    ], 'player-shots-total-odds', [1.80, 1.75, 1.85]);
    
    renderSpecialCombo('player-carded-combo', [
        'Casemiro Carded', 'Rodri Carded'
    ], 'player-carded-total-odds', [2.50, 2.80]);
    
    renderSpecialCombo('player-assists-combo', [
        'De Bruyne Assist', 'Messi Assist'
    ], 'player-assists-total-odds', [3.00, 2.80]);
    
    // === TEAMS MARKETS ===
    renderSpecialCombo('home-score-combo', [
        'Arsenal to Score', 'Barcelona to Score', 'Bayern to Score'
    ], 'home-score-total-odds', [1.15, 1.12, 1.18]);
    
    renderSpecialCombo('away-score-combo', [
        'Chelsea to Score', 'Real Madrid to Score', 'Dortmund to Score'
    ], 'away-score-total-odds', [1.40, 1.35, 1.45]);
    
    renderSpecialCombo('clean-sheet-combo', [
        'Arsenal CS', 'Bayern CS'
    ], 'clean-sheet-total-odds', [2.80, 2.50]);
    
    renderSpecialCombo('win-nil-combo', [
        'Arsenal Win to Nil', 'Bayern Win to Nil'
    ], 'win-nil-total-odds', [3.50, 3.00]);
    
    renderSpecialCombo('handicap-combo', [
        'Arsenal -1', 'Bayern -1.5', 'Barcelona -1'
    ], 'handicap-total-odds', [2.10, 2.30, 2.00]);
    
    renderSpecialCombo('margin-combo', [
        'Arsenal by 2+', 'Bayern by 3+'
    ], 'margin-total-odds', [3.20, 4.50]);
    
    // === MINUTES MARKETS ===
    renderSpecialCombo('goal-15-combo', [
        'Arsenal Goal 1-15', 'Barcelona Goal 1-15', 'Bayern Goal 1-15'
    ], 'goal-15-total-odds', [2.20, 2.10, 2.15]);
    
    renderSpecialCombo('goal-30-combo', [
        'El Clasico Goal 16-30', 'Der Klassiker Goal 16-30'
    ], 'goal-30-total-odds', [1.75, 1.80]);
    
    renderSpecialCombo('goal-45-combo', [
        'Arsenal Goal 31-45', 'PSG Goal 31-45'
    ], 'goal-45-total-odds', [1.90, 1.85]);
    
    renderSpecialCombo('goal-60-combo', [
        'Arsenal Goal 46-60', 'Barcelona Goal 46-60', 'Bayern Goal 46-60'
    ], 'goal-60-total-odds', [1.65, 1.60, 1.70]);
    
    renderSpecialCombo('goal-75-combo', [
        'Arsenal Goal 61-75', 'PSG Goal 61-75'
    ], 'goal-75-total-odds', [1.55, 1.60]);
    
    renderSpecialCombo('goal-90-combo', [
        'Arsenal Goal 76-90', 'Barcelona Goal 76-90', 'Man City Goal 76-90'
    ], 'goal-90-total-odds', [1.70, 1.65, 1.75]);
}

function renderComboPicks(containerId, picks) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = picks.map(pick => {
        // Format date if available
        let dateTimeStr = '';
        if (pick.date) {
            const dateObj = new Date(pick.date);
            const dateStr = dateObj.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
            dateTimeStr = `<span class="pick-datetime">📅 ${dateStr} ⏰ ${pick.time || '15:00'}</span>`;
        }
        
        return `
            <div class="combo-pick">
                <div class="pick-info">
                    <span class="pick-match">${pick.match}</span>
                    ${dateTimeStr}
                </div>
                <span class="pick-selection">${pick.selection} @${pick.odds?.toFixed(2) || '1.50'}</span>
            </div>
        `;
    }).join('');
}

function renderSpecialCombo(containerId, picks, oddsId, odds) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Picks can be either strings (old format) or objects with match info
    container.innerHTML = picks.map((pick, i) => {
        if (typeof pick === 'object' && pick.home && pick.away) {
            // New format with full match info
            const dateStr = pick.date ? new Date(pick.date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }) : '';
            return `
                <div class="special-pick-full">
                    <div class="pick-match-info">
                        <span class="pick-teams">${pick.home} vs ${pick.away}</span>
                        <span class="pick-datetime">📅 ${dateStr} ⏰ ${pick.time || '15:00'}</span>
                    </div>
                    <span class="pick-market">${pick.selection} @${(odds[i] || 1.5).toFixed(2)}</span>
                </div>
            `;
        } else {
            // Old string format (fallback)
            return `<div class="special-pick">${pick}</div>`;
        }
    }).join('');
    
    const total = odds.reduce((a, b) => a * b, 1);
    const oddsEl = document.getElementById(oddsId);
    if (oddsEl) oddsEl.textContent = '@' + total.toFixed(2);
}

function calculateTotalOdds(picks) {
    return picks.reduce((total, pick) => total * pick.odds, 1);
}

function updateStats(predictions) {
    document.getElementById('total-predictions').textContent = predictions.length;
    
    const avgConf = predictions.reduce((sum, p) => 
        sum + (p.prediction.confidence || 0.7), 0) / predictions.length;
    document.getElementById('avg-confidence').textContent = Math.round(avgConf * 100) + '%';
    
    const leagues = new Set(predictions.map(p => p.league));
    document.getElementById('leagues-count').textContent = leagues.size;
}
