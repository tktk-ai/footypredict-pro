/**
 * FootyPredict Pro - Frontend JavaScript
 * Connects to Cloudflare Worker API
 */

// ============= Service Worker Registration =============
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('✅ SW registered:', registration.scope);
                
                // Check for updates
                registration.addEventListener('updatefound', () => {
                    const newWorker = registration.installing;
                    newWorker.addEventListener('statechange', () => {
                        if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                            // New content available, show refresh prompt
                            if (confirm('New version available! Reload to update?')) {
                                window.location.reload();
                            }
                        }
                    });
                });
            })
            .catch((error) => {
                console.log('❌ SW registration failed:', error);
            });
    });
}

// ============= Mobile Menu Toggle =============
(function() {
    const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
    const nav = document.querySelector('.nav');
    const overlay = document.querySelector('.mobile-menu-overlay');
    
    if (mobileMenuToggle && nav) {
        mobileMenuToggle.addEventListener('click', function() {
            mobileMenuToggle.classList.toggle('active');
            nav.classList.toggle('active');
            if (overlay) overlay.classList.toggle('active');
            document.body.style.overflow = nav.classList.contains('active') ? 'hidden' : '';
        });
        
        // Close menu when clicking overlay
        if (overlay) {
            overlay.addEventListener('click', function() {
                mobileMenuToggle.classList.remove('active');
                nav.classList.remove('active');
                overlay.classList.remove('active');
                document.body.style.overflow = '';
            });
        }
        
        // Close menu when clicking a nav link
        nav.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', function() {
                mobileMenuToggle.classList.remove('active');
                nav.classList.remove('active');
                if (overlay) overlay.classList.remove('active');
                document.body.style.overflow = '';
            });
        });
    }
})();

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

// Sure Wins Section
async function loadSureWins() {
    const loadingEl = document.getElementById('sure-wins-loading');
    const gridEl = document.getElementById('sure-wins-grid');
    const noWinsEl = document.getElementById('no-sure-wins');
    
    // Check if elements exist (only on index page)
    if (!loadingEl || !gridEl) return;
    
    try {
        // Fetch fixtures with predictions
        const response = await fetch(`${API_BASE}/fixtures?days=3&includePredictions=true`);
        if (!response.ok) throw new Error('Failed to fetch fixtures');
        
        const data = await response.json();
        const fixtures = data.fixtures || [];
        
        // Filter for high-confidence matches (80%+)
        const sureWins = fixtures.filter(f => {
            const confidence = f.confidence || 0;
            return confidence >= 0.80; // 80% threshold for Sure Wins
        }).slice(0, 12); // Show max 12
        
        loadingEl.classList.add('hidden');
        
        if (sureWins.length === 0) {
            noWinsEl.classList.remove('hidden');
            return;
        }
        
        // Render sure wins
        gridEl.innerHTML = sureWins.map(match => {
            const confidence = Math.round((match.confidence || 0) * 100);
            const prediction = match.prediction?.result;
            const odds = match.odds || {};
            
            // Determine the pick
            let pick = 'Draw';
            if (prediction) {
                if (prediction.home > prediction.away && prediction.home > prediction.draw) {
                    pick = match.home_team || 'Home Win';
                } else if (prediction.away > prediction.home && prediction.away > prediction.draw) {
                    pick = match.away_team || 'Away Win';
                } else {
                    pick = 'Draw';
                }
            }
            
            return `
                <div class="sure-win-card">
                    <span class="sure-win-badge">🔥 ${confidence}%</span>
                    <div class="sure-win-teams">
                        <h4>${match.home_team} vs ${match.away_team}</h4>
                        <div class="sure-win-meta">
                            <span>📅 ${match.date}</span>
                            <span>⏰ ${match.time || 'TBA'}</span>
                            <span>🏆 ${match.league_name || 'League'}</span>
                        </div>
                    </div>
                    <div class="sure-win-prediction">
                        <span class="pick">🎯 ${pick}</span>
                        <span class="confidence">${confidence}% sure</span>
                    </div>
                    ${odds.home ? `
                        <div class="sure-win-odds">
                            <span>H: ${odds.home}</span>
                            <span>D: ${odds.draw}</span>
                            <span>A: ${odds.away}</span>
                            ${odds.over25 ? `<span>O2.5: ${odds.over25}</span>` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
        
        gridEl.classList.remove('hidden');
        
    } catch (error) {
        console.error('Sure Wins error:', error);
        loadingEl.classList.add('hidden');
        noWinsEl.classList.remove('hidden');
    }
}

// Load Sure Wins on page load
loadSureWins();

// ============= Social Sharing Functions =============
let lastPrediction = null;

// Store prediction data for sharing
function storePredictionForSharing(data) {
    lastPrediction = data;
}

function shareOnTwitter() {
    const matchTitle = document.getElementById('match-title')?.textContent || 'Football Match';
    const confidence = document.getElementById('confidence')?.textContent || '';
    const text = `⚽ ${matchTitle}\n${confidence}\n\nGet AI predictions at FootyPredict Pro! 🎯\n\n`;
    const url = window.location.href;
    window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`, '_blank');
}

function shareOnFacebook() {
    const url = window.location.href;
    window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`, '_blank');
}

function shareOnWhatsApp() {
    const matchTitle = document.getElementById('match-title')?.textContent || 'Football Match';
    const confidence = document.getElementById('confidence')?.textContent || '';
    const text = `⚽ *${matchTitle}*\n${confidence}\n\nGet AI predictions at: ${window.location.href}`;
    window.open(`https://wa.me/?text=${encodeURIComponent(text)}`, '_blank');
}

function copyPredictionLink() {
    const matchTitle = document.getElementById('match-title')?.textContent || '';
    const confidence = document.getElementById('confidence')?.textContent || '';
    const text = `⚽ ${matchTitle} - ${confidence}\n${window.location.href}`;
    
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.share-btn.copy');
        const originalContent = btn.innerHTML;
        btn.innerHTML = '<span>✅</span>';
        setTimeout(() => btn.innerHTML = originalContent, 2000);
    });
}

// ============= Newsletter Subscription =============
async function subscribeNewsletter(event) {
    event.preventDefault();
    
    const emailInput = document.getElementById('newsletter-email');
    const submitBtn = document.getElementById('subscribe-btn');
    const successDiv = document.getElementById('newsletter-success');
    const form = document.getElementById('newsletter-form');
    const email = emailInput.value.trim();
    
    if (!email) return;
    
    // Show loading
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').textContent = 'Subscribing...';
    
    try {
        // Store email (you can replace this with your actual API endpoint)
        // For now, we'll store in localStorage as a demo
        const subscribers = JSON.parse(localStorage.getItem('footypredict_subscribers') || '[]');
        if (!subscribers.includes(email)) {
            subscribers.push(email);
            localStorage.setItem('footypredict_subscribers', JSON.stringify(subscribers));
        }
        
        // Show success
        form.classList.add('hidden');
        successDiv.classList.remove('hidden');
        
        console.log('📧 New subscriber:', email);
        
    } catch (error) {
        console.error('Newsletter subscription error:', error);
        alert('Subscription failed. Please try again.');
    } finally {
        submitBtn.disabled = false;
        submitBtn.querySelector('.btn-text').textContent = 'Subscribe Free';
    }
}

// ============= Push Notifications =============
async function enablePushNotifications() {
    const pushBtn = document.getElementById('push-btn');
    const pushBtnText = document.getElementById('push-btn-text');
    
    if (!('Notification' in window)) {
        alert('Your browser does not support push notifications.');
        return;
    }
    
    if (!('serviceWorker' in navigator)) {
        alert('Service Worker is not supported in your browser.');
        return;
    }
    
    pushBtn.disabled = true;
    pushBtnText.textContent = 'Enabling...';
    
    try {
        const permission = await Notification.requestPermission();
        
        if (permission === 'granted') {
            // Get service worker registration
            const registration = await navigator.serviceWorker.ready;
            
            // Subscribe to push (for demo, we're just enabling notifications)
            pushBtnText.textContent = '✅ Notifications Enabled';
            pushBtn.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
            
            // Store subscription status
            localStorage.setItem('footypredict_push_enabled', 'true');
            
            // Show a test notification
            new Notification('⚽ FootyPredict Pro', {
                body: 'You will now receive alerts for high-confidence predictions!',
                icon: '/icons/icon.svg'
            });
            
            console.log('🔔 Push notifications enabled');
            
        } else if (permission === 'denied') {
            pushBtnText.textContent = 'Notifications Blocked';
            pushBtn.style.background = '#ef4444';
            alert('Notifications were blocked. Please enable them in your browser settings.');
        } else {
            pushBtnText.textContent = 'Enable Notifications';
            pushBtn.disabled = false;
        }
        
    } catch (error) {
        console.error('Push notification error:', error);
        pushBtnText.textContent = 'Enable Notifications';
        pushBtn.disabled = false;
    }
}

// Check if push is already enabled
function checkPushStatus() {
    const pushBtn = document.getElementById('push-btn');
    const pushBtnText = document.getElementById('push-btn-text');
    
    if (!pushBtn || !pushBtnText) return;
    
    if (localStorage.getItem('footypredict_push_enabled') === 'true' && Notification.permission === 'granted') {
        pushBtnText.textContent = '✅ Notifications Enabled';
        pushBtn.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
        pushBtn.disabled = true;
    }
}

// Run on page load
document.addEventListener('DOMContentLoaded', checkPushStatus);
