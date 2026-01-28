/**
 * FootyPredict Pro - Frontend JavaScript
 * Connects to Cloudflare Worker API
 */

const API_BASE = 'https://footypredict-api.tirene857.workers.dev';

// DOM Elements
const form = document.getElementById('prediction-form');
const homeTeamInput = document.getElementById('home-team');
const awayTeamInput = document.getElementById('away-team');
const leagueSelect = document.getElementById('league');
const predictBtn = document.getElementById('predict-btn');
const btnText = predictBtn.querySelector('.btn-text');
const btnLoader = predictBtn.querySelector('.btn-loader');
const resultsDiv = document.getElementById('results');
const errorDiv = document.getElementById('error-message');
const errorText = document.getElementById('error-text');

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const homeTeam = homeTeamInput.value.trim();
    const awayTeam = awayTeamInput.value.trim();
    const league = leagueSelect.value || undefined;
    
    if (!homeTeam || !awayTeam) {
        showError('Please enter both team names');
        return;
    }
    
    await getPrediction(homeTeam, awayTeam, league);
});

async function getPrediction(homeTeam, awayTeam, league) {
    // Show loading state
    setLoading(true);
    hideError();
    resultsDiv.classList.add('hidden');
    
    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                home_team: homeTeam,
                away_team: awayTeam,
                league: league
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        displayResults(data);
        
    } catch (error) {
        console.error('Prediction error:', error);
        showError(error.message || 'Failed to get prediction. Please try again.');
    } finally {
        setLoading(false);
    }
}

function displayResults(data) {
    // Update match title
    document.getElementById('match-title').textContent = 
        `${data.home_team} vs ${data.away_team}`;
    
    // Update confidence
    const confidence = Math.round(data.confidence * 100);
    const confidenceBadge = document.getElementById('confidence');
    confidenceBadge.textContent = `${confidence}% Confidence`;
    confidenceBadge.style.background = confidence >= 70 ? 'var(--success)' : 
                                       confidence >= 50 ? 'var(--warning)' : 'var(--error)';
    
    // Update result probabilities
    const predictions = data.predictions;
    if (predictions.result) {
        updateProbability('home', predictions.result.home);
        updateProbability('draw', predictions.result.draw);
        updateProbability('away', predictions.result.away);
        document.getElementById('result-recommendation').textContent = 
            predictions.result.recommendation;
    }
    
    // Update goals markets
    if (predictions.over_25) {
        document.getElementById('over25-prob').textContent = 
            `${Math.round(predictions.over_25.yes * 100)}%`;
        document.getElementById('over25-rec').textContent = 
            predictions.over_25.recommendation;
    }
    
    if (predictions.btts) {
        document.getElementById('btts-prob').textContent = 
            `${Math.round(predictions.btts.yes * 100)}%`;
        document.getElementById('btts-rec').textContent = 
            predictions.btts.recommendation;
    }
    
    // Update footer
    document.getElementById('model-version').textContent = 
        `Model v${data.model_version || '1.0.0'}`;
    document.getElementById('timestamp').textContent = 
        new Date(data.timestamp).toLocaleString();
    
    // Show results
    resultsDiv.classList.remove('hidden');
}

function updateProbability(type, value) {
    const percentage = Math.round(value * 100);
    document.getElementById(`prob-${type}`).style.width = `${percentage}%`;
    document.getElementById(`prob-${type}-val`).textContent = `${percentage}%`;
}

function setLoading(isLoading) {
    predictBtn.disabled = isLoading;
    btnText.textContent = isLoading ? 'Analyzing...' : 'Get Prediction';
    btnLoader.classList.toggle('hidden', !isLoading);
}

function showError(message) {
    errorText.textContent = message;
    errorDiv.classList.remove('hidden');
}

function hideError() {
    errorDiv.classList.add('hidden');
}

// Copy API endpoint helper
function copyEndpoint() {
    const curlCommand = `curl -X POST ${API_BASE}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"home_team":"Arsenal","away_team":"Chelsea","league":"Premier League"}'`;
    
    navigator.clipboard.writeText(curlCommand).then(() => {
        const btn = document.querySelector('.btn-copy');
        const originalText = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = originalText, 2000);
    });
}

// Add some sample autocomplete data
const popularTeams = [
    'Arsenal', 'Chelsea', 'Liverpool', 'Manchester United', 'Manchester City',
    'Tottenham', 'Newcastle', 'Aston Villa', 'West Ham', 'Brighton',
    'Barcelona', 'Real Madrid', 'Atletico Madrid', 'Sevilla',
    'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen',
    'PSG', 'Marseille', 'Lyon', 'Monaco',
    'Juventus', 'AC Milan', 'Inter Milan', 'Napoli', 'Roma'
];

// Simple autocomplete (optional enhancement)
function setupAutocomplete(input) {
    input.addEventListener('input', () => {
        const value = input.value.toLowerCase();
        if (value.length < 2) return;
        
        const matches = popularTeams.filter(team => 
            team.toLowerCase().includes(value)
        );
        
        // Could add dropdown here if desired
    });
}

setupAutocomplete(homeTeamInput);
setupAutocomplete(awayTeamInput);

// Health check on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE}/`);
        if (response.ok) {
            console.log('✅ API is healthy');
        }
    } catch (error) {
        console.warn('⚠️ API health check failed:', error);
    }
}

checkAPIHealth();
