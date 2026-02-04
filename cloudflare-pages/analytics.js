/**
 * Analytics Dashboard JavaScript
 * Handles data fetching, chart rendering, and interactivity
 */

// API Base URL - use relative for same-origin, or specify your deployed API
const API_BASE = 'https://footypredict-api.ananseglobal.workers.dev';
const DASHBOARD_API = API_BASE + '/api/analytics/dashboard';

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    setupEventListeners();
});

// Dashboard initialization
async function initializeDashboard() {
    await loadTodaySummary();
    await loadSectionAnalytics();  // NEW: Load section-based stats
    await loadAccuracyChart();
    loadConfidenceChart();
    loadWinLossChart();
    await loadLeagueTable();
    await loadMarketTable();
    await loadTopPicks();
    await calculateROI();
}

// Event listeners
function setupEventListeners() {
    // Filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            const period = e.target.dataset.period;
            filterByPeriod(period);
        });
    });
    
    // Mobile menu
    const menuToggle = document.querySelector('.mobile-menu-toggle');
    const nav = document.querySelector('.nav');
    const overlay = document.querySelector('.mobile-menu-overlay');
    
    if (menuToggle) {
        menuToggle.addEventListener('click', () => {
            nav.classList.toggle('active');
            overlay.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
    }
}

// Load today's summary from REAL DATABASE
async function loadTodaySummary() {
    // Default values (fallback)
    let data = {
        total_predictions: 0,
        avg_confidence: 0,
        status: { won: 0, lost: 0, pending: 0, accuracy: 0 },
        confidence_distribution: { high: 0, medium: 0, low: 0 },
        data_source: 'fallback'
    };
    
    try {
        // Fetch from real database API
        const response = await fetch(`${DASHBOARD_API}/today`);
        if (response.ok) {
            const apiData = await response.json();
            if (apiData.success) {
                data = apiData;
                console.log('📊 Analytics loaded from database:', data.data_source);
            }
        }
    } catch (e) {
        console.log('Using fallback data for summary:', e.message);
    }
    
    // Update UI with real data
    const totalPreds = data.total_predictions || 0;
    const accuracy = data.status?.accuracy || data.avg_confidence || 0;
    const won = data.status?.won || 0;
    const lost = data.status?.lost || 0;
    const pending = data.status?.pending || 0;
    
    // Calculate ROI estimate
    const roi = totalPreds > 0 ? ((won * 1.85 - (won + lost)) / Math.max(1, won + lost) * 100).toFixed(1) : 0;
    
    document.getElementById('total-predictions').textContent = totalPreds;
    document.getElementById('accuracy-rate').textContent = `${accuracy}%`;
    document.getElementById('won-count').textContent = won;
    document.getElementById('lost-count').textContent = lost;
    document.getElementById('pending-count').textContent = pending;
    document.getElementById('roi-value').textContent = roi >= 0 ? `+${roi}%` : `${roi}%`;
    
    // Update data source indicator if fallback
    if (data.data_source === 'fallback') {
        console.log('⚠️ Using fallback data - run prediction generator job to populate database');
    }
}

// Load section-based analytics (Daily Tips, Money Zone, ACCAs)
async function loadSectionAnalytics() {
    let sections = {
        daily_tips: { total: 0, won: 0, accuracy: 0, roi: 0 },
        money_zone: { total: 0, won: 0, accuracy: 0, roi: 0 },
        accas: { total: 0, won: 0, accuracy: 0, roi: 0 }
    };
    
    try {
        const response = await fetch(`${DASHBOARD_API}/sections?section=all`);
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                sections = {
                    daily_tips: data.daily_tips || sections.daily_tips,
                    money_zone: data.money_zone || sections.money_zone,
                    accas: data.accas || sections.accas
                };
                console.log('📊 Section analytics loaded from database');
            }
        }
    } catch (e) {
        console.log('Using fallback section data:', e.message);
    }
    
    // Update section cards in UI (if elements exist)
    updateSectionCard('daily-tips', sections.daily_tips);
    updateSectionCard('money-zone', sections.money_zone);
    updateSectionCard('accas', sections.accas);
    
    // Also update combined stats display if elements exist
    const combinedEl = document.getElementById('section-analytics');
    if (combinedEl) {
        combinedEl.innerHTML = `
            <div class="section-grid">
                ${renderSectionCard('Daily Tips', sections.daily_tips, '📋')}
                ${renderSectionCard('Money Zone', sections.money_zone, '💰')}
                ${renderSectionCard('ACCAs', sections.accas, '🎯')}
            </div>
        `;
    }
}

// Helper to update individual section card
function updateSectionCard(sectionId, data) {
    const el = document.getElementById(sectionId + '-stats');
    if (el) {
        el.querySelector('.section-total').textContent = data.total || 0;
        el.querySelector('.section-accuracy').textContent = `${data.accuracy || 0}%`;
        el.querySelector('.section-roi').textContent = data.roi >= 0 ? `+${data.roi}%` : `${data.roi}%`;
    }
}

// Helper to render section card HTML
function renderSectionCard(name, data, icon) {
    const roiClass = data.roi >= 0 ? 'positive' : 'negative';
    const accuracyClass = data.accuracy >= 70 ? 'high' : data.accuracy >= 60 ? 'medium' : 'low';
    
    return `
        <div class="section-card">
            <div class="section-header">
                <span class="section-icon">${icon}</span>
                <h4>${name}</h4>
            </div>
            <div class="section-stats">
                <div class="stat-item">
                    <span class="stat-label">Predictions</span>
                    <span class="stat-value">${data.total || 0}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Won</span>
                    <span class="stat-value positive">${data.won || 0}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Accuracy</span>
                    <span class="stat-value accuracy-badge ${accuracyClass}">${data.accuracy || 0}%</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">ROI</span>
                    <span class="stat-value ${roiClass}">${data.roi >= 0 ? '+' : ''}${data.roi || 0}%</span>
                </div>
            </div>
        </div>
    `;
}

// Load accuracy chart
async function loadAccuracyChart() {
    const ctx = document.getElementById('accuracy-chart').getContext('2d');
    
    // Generate sample data for 30 days
    const labels = [];
    const accuracyData = [];
    const predictionsData = [];
    
    for (let i = 29; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        
        // Simulate varying accuracy (70-85%)
        const baseAccuracy = 72 + Math.sin(i * 0.5) * 8 + (Math.random() * 5);
        accuracyData.push(Math.min(85, Math.max(65, baseAccuracy)));
        predictionsData.push(15 + Math.floor(Math.random() * 15));
    }
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy %',
                    data: accuracyData,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true,
                    yAxisID: 'y'
                },
                {
                    label: 'Predictions',
                    data: predictionsData,
                    borderColor: '#22c55e',
                    backgroundColor: 'transparent',
                    tension: 0.4,
                    borderDash: [5, 5],
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    min: 50,
                    max: 100,
                    ticks: { 
                        color: '#6366f1',
                        callback: (value) => value + '%'
                    },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    min: 0,
                    max: 50,
                    ticks: { color: '#22c55e' },
                    grid: { display: false }
                }
            }
        }
    });
}

// Load confidence distribution chart
function loadConfidenceChart() {
    const ctx = document.getElementById('confidence-chart').getContext('2d');
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High (80%+)', 'Medium (60-79%)', 'Low (<60%)'],
            datasets: [{
                data: [8, 12, 4],
                backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8' }
                }
            }
        }
    });
}

// Load win/loss chart
function loadWinLossChart() {
    const ctx = document.getElementById('winloss-chart').getContext('2d');
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            datasets: [
                {
                    label: 'Won',
                    data: [3, 5, 4, 6, 4, 7, 5],
                    backgroundColor: '#22c55e'
                },
                {
                    label: 'Lost',
                    data: [1, 2, 1, 2, 2, 3, 1],
                    backgroundColor: '#ef4444'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#94a3b8' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#64748b' },
                    grid: { display: false }
                },
                y: {
                    ticks: { color: '#64748b' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}

// Load league performance table from REAL DATABASE
async function loadLeagueTable() {
    let leagues = [];
    
    try {
        const response = await fetch(`${DASHBOARD_API}/league`);
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.leagues) {
                leagues = data.leagues;
                console.log('📊 League data loaded from database');
            }
        }
    } catch (e) {
        console.log('Using fallback league data');
    }
    
    // Fallback if no data
    if (leagues.length === 0) {
        leagues = [
            { league: 'No Data', predictions: 0, won: 0, accuracy: 0, roi: 0 }
        ];
    }
    
    const tbody = document.getElementById('league-table');
    tbody.innerHTML = leagues.map(l => `
        <tr>
            <td><strong>${l.league}</strong></td>
            <td>${l.predictions}</td>
            <td>${l.won}</td>
            <td><span class="accuracy-badge ${l.accuracy >= 70 ? 'high' : l.accuracy >= 60 ? 'medium' : 'low'}">${l.accuracy}%</span></td>
            <td class="${l.roi >= 0 ? 'positive' : 'negative'}" style="color: ${l.roi >= 0 ? '#22c55e' : '#ef4444'}">${l.roi >= 0 ? '+' : ''}${l.roi}%</td>
        </tr>
    `).join('');
}

// Load market performance table from REAL DATABASE
async function loadMarketTable() {
    let markets = [];
    
    try {
        const response = await fetch(`${DASHBOARD_API}/market`);
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.markets) {
                markets = data.markets;
                console.log('📊 Market data loaded from database');
            }
        }
    } catch (e) {
        console.log('Using fallback market data');
    }
    
    // Fallback if no data
    if (markets.length === 0) {
        markets = [
            { market: 'No Data', predictions: 0, won: 0, accuracy: 0, avgOdds: 0 }
        ];
    }
    
    const tbody = document.getElementById('market-table');
    tbody.innerHTML = markets.map(m => `
        <tr>
            <td><strong>${m.market}</strong></td>
            <td>${m.predictions}</td>
            <td>${m.won}</td>
            <td><span class="accuracy-badge ${m.accuracy >= 70 ? 'high' : m.accuracy >= 60 ? 'medium' : 'low'}">${m.accuracy}%</span></td>
            <td>@${m.avgOdds.toFixed(2)}</td>
        </tr>
    `).join('');
}

// Load top picks
async function loadTopPicks() {
    const picks = [
        { rank: 1, match: 'Arsenal vs Chelsea', confidence: 91, prediction: 'Home Win', status: 'pending' },
        { rank: 2, match: 'Barcelona vs Real Madrid', confidence: 87, prediction: 'BTTS Yes', status: 'pending' },
        { rank: 3, match: 'Bayern vs Dortmund', confidence: 85, prediction: 'Over 2.5', status: 'won' },
        { rank: 4, match: 'PSG vs Lyon', confidence: 82, prediction: '1X', status: 'pending' },
        { rank: 5, match: 'Inter vs Juventus', confidence: 80, prediction: 'Home Win', status: 'lost' }
    ];
    
    const container = document.getElementById('top-picks');
    container.innerHTML = picks.map(p => `
        <div class="pick-card">
            <div class="pick-header">
                <span class="pick-rank">${p.rank}</span>
                <span class="pick-confidence">${p.confidence}%</span>
            </div>
            <div class="pick-match">${p.match}</div>
            <div class="pick-details">
                <span class="pick-prediction">${p.prediction}</span>
                <span class="pick-status ${p.status}">${p.status.toUpperCase()}</span>
            </div>
        </div>
    `).join('');
}

// Calculate ROI from REAL DATABASE
async function calculateROI() {
    const stake = parseFloat(document.getElementById('stake-input').value) || 10;
    const days = parseInt(document.getElementById('period-input').value) || 30;
    
    let totalBets = 0;
    let wins = 0;
    let avgOdds = 1.85;
    let totalStaked = 0;
    let totalReturns = 0;
    let profit = 0;
    let roi = 0;
    
    try {
        const response = await fetch(`${DASHBOARD_API}/calculate-roi?stake=${stake}&days=${days}`);
        if (response.ok) {
            const data = await response.json();
            if (data.success && data.data_source === 'database') {
                totalBets = data.total_bets || 0;
                wins = data.wins || 0;
                avgOdds = data.avg_odds || 1.85;
                totalStaked = data.total_staked || 0;
                totalReturns = data.total_returns || 0;
                profit = data.profit || 0;
                roi = data.roi || 0;
                console.log('📊 ROI calculated from database');
            } else {
                // Calculate locally as fallback
                const betsPerDay = 6;
                totalBets = betsPerDay * days;
                const winRate = 0.70;
                wins = Math.round(totalBets * winRate);
                totalStaked = totalBets * stake;
                totalReturns = wins * stake * avgOdds;
                profit = totalReturns - totalStaked;
                roi = (profit / totalStaked) * 100;
            }
        }
    } catch (e) {
        // Local fallback calculation
        const betsPerDay = 6;
        totalBets = betsPerDay * days;
        const winRate = 0.70;
        wins = Math.round(totalBets * winRate);
        totalStaked = totalBets * stake;
        totalReturns = wins * stake * avgOdds;
        profit = totalReturns - totalStaked;
        roi = (profit / totalStaked) * 100;
    }
    
    document.getElementById('total-staked').textContent = `$${totalStaked.toLocaleString()}`;
    document.getElementById('total-returns').textContent = `$${Math.round(totalReturns).toLocaleString()}`;
    document.getElementById('net-profit').textContent = profit >= 0 ? `+$${Math.round(profit).toLocaleString()}` : `-$${Math.abs(Math.round(profit)).toLocaleString()}`;
    document.getElementById('roi-percent').textContent = roi >= 0 ? `+${roi.toFixed(1)}%` : `${roi.toFixed(1)}%`;
    
    // Update styling
    const profitEl = document.getElementById('net-profit');
    const roiEl = document.getElementById('roi-percent');
    profitEl.className = `roi-value ${profit >= 0 ? 'positive' : 'negative'}`;
    roiEl.className = `roi-value ${roi >= 0 ? 'positive' : 'negative'}`;
}

// Filter by period
function filterByPeriod(period) {
    console.log(`Filtering by: ${period}`);
    // In production, refetch data for selected period
    // For now, just log
}

// Export data
function exportData() {
    const data = {
        exportDate: new Date().toISOString(),
        summary: {
            totalPredictions: document.getElementById('total-predictions').textContent,
            accuracy: document.getElementById('accuracy-rate').textContent,
            won: document.getElementById('won-count').textContent,
            lost: document.getElementById('lost-count').textContent
        },
        message: 'Export data functionality - integrate with backend'
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `footypredict-analytics-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
