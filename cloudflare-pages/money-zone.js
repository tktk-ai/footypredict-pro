/**
 * Money Zone - JavaScript Logic
 * Handles time-based predictions and combo generation
 */

const API_BASE = 'https://footypredict-api.tirene857.workers.dev';

// Sample match data for different time periods (in production, this would come from an API)
const sampleMatches = {
    today: {
        breakfast: [
            { id: 1, home: 'Melbourne Victory', away: 'Sydney FC', league: 'A-League', time: '05:00', country: '🇦🇺' },
            { id: 2, home: 'Ulsan HD', away: 'Jeonbuk Motors', league: 'K-League', time: '08:00', country: '🇰🇷' },
            { id: 3, home: 'Yokohama Marinos', away: 'Vissel Kobe', league: 'J-League', time: '09:30', country: '🇯🇵' },
        ],
        lunch: [
            { id: 4, home: 'Al Hilal', away: 'Al Nassr', league: 'Saudi Pro', time: '12:00', country: '🇸🇦' },
            { id: 5, home: 'Mamelodi Sundowns', away: 'Orlando Pirates', league: 'PSL', time: '13:00', country: '🇿🇦' },
            { id: 6, home: 'Raja Casablanca', away: 'Wydad AC', league: 'Botola Pro', time: '14:30', country: '🇲🇦' },
            { id: 7, home: 'Zamalek', away: 'Al Ahly', league: 'Egyptian Premier', time: '15:00', country: '🇪🇬' },
        ],
        dinner: [
            { id: 8, home: 'Arsenal', away: 'Chelsea', league: 'Premier League', time: '17:30', country: '🏴󠁧󠁢󠁥󠁮󠁧󠁿' },
            { id: 9, home: 'Barcelona', away: 'Real Madrid', league: 'La Liga', time: '20:00', country: '🇪🇸' },
            { id: 10, home: 'Bayern Munich', away: 'Dortmund', league: 'Bundesliga', time: '18:30', country: '🇩🇪' },
            { id: 11, home: 'PSG', away: 'Marseille', league: 'Ligue 1', time: '21:00', country: '🇫🇷' },
            { id: 12, home: 'Juventus', away: 'AC Milan', league: 'Serie A', time: '20:45', country: '🇮🇹' },
            { id: 13, home: 'Flamengo', away: 'Palmeiras', league: 'Brasileirao', time: '22:00', country: '🇧🇷' },
        ]
    },
    week: {
        breakfast: generateWeekMatches('breakfast', 7),
        lunch: generateWeekMatches('lunch', 7),
        dinner: generateWeekMatches('dinner', 7)
    },
    '2weeks': {
        breakfast: generateWeekMatches('breakfast', 14),
        lunch: generateWeekMatches('lunch', 14),
        dinner: generateWeekMatches('dinner', 14)
    },
    '3weeks': {
        breakfast: generateWeekMatches('breakfast', 21),
        lunch: generateWeekMatches('lunch', 21),
        dinner: generateWeekMatches('dinner', 21)
    },
    '4weeks': {
        breakfast: generateWeekMatches('breakfast', 28),
        lunch: generateWeekMatches('lunch', 28),
        dinner: generateWeekMatches('dinner', 28)
    }
};

function generateWeekMatches(timeSlot, days) {
    const teams = {
        breakfast: [
            ['Melbourne Victory', 'Sydney FC', 'A-League', '🇦🇺'],
            ['Ulsan HD', 'Jeonbuk Motors', 'K-League', '🇰🇷'],
            ['Yokohama Marinos', 'Vissel Kobe', 'J-League', '🇯🇵'],
            ['Beijing Guoan', 'Shanghai Port', 'CSL', '🇨🇳'],
        ],
        lunch: [
            ['Al Hilal', 'Al Nassr', 'Saudi Pro', '🇸🇦'],
            ['Mamelodi Sundowns', 'Orlando Pirates', 'PSL', '🇿🇦'],
            ['Zamalek', 'Al Ahly', 'Egyptian Premier', '🇪🇬'],
            ['Raja Casablanca', 'Wydad AC', 'Botola Pro', '🇲🇦'],
        ],
        dinner: [
            ['Arsenal', 'Chelsea', 'Premier League', '🏴󠁧󠁢󠁥󠁮󠁧󠁿'],
            ['Barcelona', 'Real Madrid', 'La Liga', '🇪🇸'],
            ['Bayern Munich', 'Dortmund', 'Bundesliga', '🇩🇪'],
            ['PSG', 'Marseille', 'Ligue 1', '🇫🇷'],
            ['Juventus', 'AC Milan', 'Serie A', '🇮🇹'],
            ['Flamengo', 'Palmeiras', 'Brasileirao', '🇧🇷'],
        ]
    };
    
    const times = {
        breakfast: ['05:00', '06:30', '08:00', '09:30', '10:00'],
        lunch: ['11:30', '12:00', '13:00', '14:00', '15:00'],
        dinner: ['16:30', '17:30', '18:30', '20:00', '21:00', '22:00']
    };
    
    const matches = [];
    const teamList = teams[timeSlot];
    const timeList = times[timeSlot];
    
    for (let d = 0; d < days; d++) {
        const date = new Date();
        date.setDate(date.getDate() + d);
        const dateStr = date.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit' });
        
        const numMatches = Math.floor(Math.random() * 3) + 2;
        for (let i = 0; i < numMatches && i < teamList.length; i++) {
            const [home, away, league, country] = teamList[i];
            matches.push({
                id: `${d}-${i}`,
                home,
                away,
                league,
                time: timeList[i % timeList.length],
                date: dateStr,
                country
            });
        }
    }
    
    return matches;
}

// State
let currentPeriod = 'today';
let currentTime = 'dinner'; // Default based on current time

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
    
    // Determine default time tab based on current hour
    const hour = now.getHours();
    if (hour < 11) currentTime = 'breakfast';
    else if (hour < 16) currentTime = 'lunch';
    else currentTime = 'dinner';
    
    // Set active time tab
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
});

async function loadPredictions() {
    loadingState.classList.remove('hidden');
    predictionsGrid.innerHTML = '';
    noMatchesState.classList.add('hidden');
    
    // Simulate API delay
    await new Promise(r => setTimeout(r, 500));
    
    const matches = sampleMatches[currentPeriod]?.[currentTime] || [];
    
    if (matches.length === 0) {
        loadingState.classList.add('hidden');
        noMatchesState.classList.remove('hidden');
        return;
    }
    
    // Generate predictions for each match
    const predictions = await Promise.all(matches.map(async match => {
        try {
            const response = await fetch(`${API_BASE}/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    home_team: match.home,
                    away_team: match.away,
                    league: match.league
                })
            });
            const data = await response.json();
            return { ...match, prediction: data };
        } catch (error) {
            // Fallback prediction
            return {
                ...match,
                prediction: generateFallbackPrediction(match)
            };
        }
    }));
    
    loadingState.classList.add('hidden');
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
        
        return `
            <div class="match-card">
                <div class="match-header">
                    <span class="match-time">${match.date ? match.date + ' ' : ''}${match.time}</span>
                    <span class="match-league">${match.country} ${match.league}</span>
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

function loadCombos() {
    // Safe Combo - High confidence picks
    const safePicks = [
        { match: 'Arsenal vs Chelsea', selection: 'Over 1.5 Goals', odds: 1.35 },
        { match: 'Barcelona vs Real Madrid', selection: 'BTTS Yes', odds: 1.55 },
        { match: 'Bayern vs Dortmund', selection: 'Home Win', odds: 1.45 },
    ];
    renderComboPicks('safe-picks', safePicks);
    document.getElementById('safe-odds').textContent = '@' + calculateTotalOdds(safePicks).toFixed(2);
    
    // Value Combo
    const valuePicks = [
        { match: 'PSG vs Marseille', selection: 'Home Win', odds: 1.60 },
        { match: 'Juventus vs AC Milan', selection: 'Draw', odds: 3.20 },
        { match: 'Al Hilal vs Al Nassr', selection: 'Over 2.5', odds: 1.85 },
        { match: 'Flamengo vs Palmeiras', selection: 'BTTS Yes', odds: 1.70 },
    ];
    renderComboPicks('value-picks', valuePicks);
    document.getElementById('value-odds').textContent = '@' + calculateTotalOdds(valuePicks).toFixed(2);
    
    // Risky Combo
    const riskyPicks = [
        { match: 'Melbourne vs Sydney', selection: 'Away Win', odds: 2.80 },
        { match: 'Zamalek vs Al Ahly', selection: 'Draw', odds: 3.40 },
        { match: 'Raja vs Wydad', selection: 'Under 1.5', odds: 2.50 },
        { match: 'Ulsan vs Jeonbuk', selection: 'BTTS + Over 2.5', odds: 2.20 },
    ];
    renderComboPicks('risky-picks', riskyPicks);
    document.getElementById('risky-odds').textContent = '@' + calculateTotalOdds(riskyPicks).toFixed(2);
    
    // Special combos
    renderSpecialCombo('over25-combo', [
        'Arsenal vs Chelsea',
        'Bayern vs Dortmund',
        'PSG vs Marseille'
    ], 'over25-total-odds', [1.55, 1.65, 1.70]);
    
    renderSpecialCombo('btts-combo', [
        'Barcelona vs Real Madrid',
        'Juventus vs AC Milan',
        'Flamengo vs Palmeiras'
    ], 'btts-total-odds', [1.65, 1.75, 1.60]);
    
    renderSpecialCombo('home-combo', [
        'Arsenal vs Chelsea',
        'Bayern vs Dortmund',
        'Al Hilal vs Al Nassr'
    ], 'home-total-odds', [1.85, 1.45, 1.55]);
}

function renderComboPicks(containerId, picks) {
    const container = document.getElementById(containerId);
    container.innerHTML = picks.map(pick => `
        <div class="combo-pick">
            <span class="pick-match">${pick.match}</span>
            <span class="pick-selection">${pick.selection}</span>
        </div>
    `).join('');
}

function renderSpecialCombo(containerId, matches, oddsId, odds) {
    const container = document.getElementById(containerId);
    container.innerHTML = matches.map(m => `<div class="special-pick">${m}</div>`).join('');
    const total = odds.reduce((a, b) => a * b, 1);
    document.getElementById(oddsId).textContent = '@' + total.toFixed(2);
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
