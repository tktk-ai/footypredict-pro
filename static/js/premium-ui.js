/**
 * Premium UI JavaScript Enhancements
 * Phase 6: Cutting-edge UI/UX features
 */

// =====================
// Theme Toggle (Dark/Light)
// =====================

class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'dark';
        this.init();
    }
    
    init() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.createToggle();
    }
    
    createToggle() {
        const toggle = document.createElement('button');
        toggle.className = 'theme-toggle';
        toggle.innerHTML = this.theme === 'dark' ? '🌙' : '☀️';
        toggle.title = 'Toggle theme';
        toggle.onclick = () => this.toggle();
        document.body.appendChild(toggle);
    }
    
    toggle() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', this.theme);
        localStorage.setItem('theme', this.theme);
        document.querySelector('.theme-toggle').innerHTML = 
            this.theme === 'dark' ? '🌙' : '☀️';
    }
}

// =====================
// AI Confidence Gauge
// =====================

class ConfidenceGauge {
    constructor(container, value = 0.5) {
        this.container = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;
        this.value = value;
        this.render();
    }
    
    render() {
        if (!this.container) return;
        
        this.container.innerHTML = `
            <div class="confidence-gauge">
                <div class="gauge-arc"></div>
                <div class="gauge-needle" style="transform: translateX(-50%) rotate(${this.getRotation()}deg)"></div>
                <div class="gauge-center"></div>
                <div class="gauge-value">${Math.round(this.value * 100)}%</div>
                <div class="gauge-label">AI Confidence</div>
            </div>
        `;
    }
    
    getRotation() {
        // Map 0-1 to -90 to 90 degrees
        return -90 + (this.value * 180);
    }
    
    update(value) {
        this.value = Math.max(0, Math.min(1, value));
        const needle = this.container.querySelector('.gauge-needle');
        const valueEl = this.container.querySelector('.gauge-value');
        
        if (needle) {
            needle.style.transform = `translateX(-50%) rotate(${this.getRotation()}deg)`;
        }
        if (valueEl) {
            valueEl.textContent = `${Math.round(this.value * 100)}%`;
        }
    }
}

// =====================
// Real-Time Odds Ticker
// =====================

class OddsTicker {
    constructor() {
        this.odds = [];
        this.container = null;
        this.init();
    }
    
    init() {
        this.container = document.createElement('div');
        this.container.className = 'odds-ticker';
        this.container.innerHTML = '<div class="ticker-content"></div>';
        
        const header = document.querySelector('header, nav, .navbar');
        if (header) {
            header.after(this.container);
        } else {
            document.body.prepend(this.container);
        }
        
        this.fetchOdds();
        setInterval(() => this.fetchOdds(), 60000); // Update every minute
    }
    
    async fetchOdds() {
        try {
            const response = await fetch('/api/live-odds');
            if (response.ok) {
                const data = await response.json();
                this.updateDisplay(data.odds || []);
            }
        } catch (e) {
            // Use fallback display
            this.updateDisplay(this.getFallbackOdds());
        }
    }
    
    getFallbackOdds() {
        return [
            { home: 'Liverpool', away: 'Arsenal', odds: { home: 2.1, draw: 3.4, away: 3.5 } },
            { home: 'Man City', away: 'Chelsea', odds: { home: 1.5, draw: 4.2, away: 6.0 } },
            { home: 'Bayern', away: 'Dortmund', odds: { home: 1.8, draw: 3.8, away: 4.5 } },
            { home: 'Real Madrid', away: 'Barcelona', odds: { home: 2.4, draw: 3.3, away: 2.9 } },
        ];
    }
    
    updateDisplay(odds) {
        const content = this.container.querySelector('.ticker-content');
        if (!content) return;
        
        let html = '';
        // Duplicate for seamless scroll
        for (let i = 0; i < 2; i++) {
            odds.forEach(match => {
                html += `
                    <div class="ticker-item">
                        <span class="team-names">${match.home} vs ${match.away}</span>
                        <span class="odds">H: ${match.odds?.home?.toFixed(2) || '2.00'}</span>
                        <span class="odds">D: ${match.odds?.draw?.toFixed(2) || '3.00'}</span>
                        <span class="odds">A: ${match.odds?.away?.toFixed(2) || '3.00'}</span>
                    </div>
                `;
            });
        }
        
        content.innerHTML = html;
    }
}

// =====================
// Injury Alert System
// =====================

class InjuryAlerts {
    constructor() {
        this.alertQueue = [];
        this.isShowing = false;
        this.init();
    }
    
    init() {
        this.container = document.createElement('div');
        this.container.className = 'injury-alert';
        document.body.appendChild(this.container);
    }
    
    show(injury) {
        this.alertQueue.push(injury);
        if (!this.isShowing) {
            this.processQueue();
        }
    }
    
    processQueue() {
        if (this.alertQueue.length === 0) {
            this.isShowing = false;
            return;
        }
        
        this.isShowing = true;
        const injury = this.alertQueue.shift();
        
        this.container.innerHTML = `
            <div class="alert-header">
                <span class="alert-icon">🏥</span>
                <span>Injury Alert</span>
            </div>
            <div class="player-name">${injury.player}</div>
            <div class="injury-details">${injury.team} - ${injury.type}</div>
        `;
        
        this.container.classList.add('show');
        
        setTimeout(() => {
            this.container.classList.remove('show');
            setTimeout(() => this.processQueue(), 500);
        }, 4000);
    }
}

// =====================
// Voice Predictions
// =====================

class VoicePrediction {
    constructor() {
        this.synth = window.speechSynthesis;
        this.enabled = false;
    }
    
    toggle() {
        this.enabled = !this.enabled;
        return this.enabled;
    }
    
    speak(text) {
        if (!this.enabled || !this.synth) return;
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 1;
        this.synth.speak(utterance);
    }
    
    announcePrediction(homeTeam, awayTeam, prediction) {
        const text = `Prediction for ${homeTeam} versus ${awayTeam}: ${prediction}`;
        this.speak(text);
    }
}

// =====================
// Skeleton Loading
// =====================

function showSkeletonLoading(container, count = 3) {
    const el = typeof container === 'string' 
        ? document.querySelector(container) 
        : container;
    
    if (!el) return;
    
    let html = '';
    for (let i = 0; i < count; i++) {
        html += `
            <div class="skeleton skeleton-card">
                <div class="skeleton skeleton-text"></div>
                <div class="skeleton skeleton-text short"></div>
            </div>
        `;
    }
    el.innerHTML = html;
}

// =====================
// Initialize on Load
// =====================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize all premium features
    window.themeManager = new ThemeManager();
    window.oddsTicker = new OddsTicker();
    window.injuryAlerts = new InjuryAlerts();
    window.voicePrediction = new VoicePrediction();
    
    console.log('✨ Premium UI initialized');
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ThemeManager,
        ConfidenceGauge,
        OddsTicker,
        InjuryAlerts,
        VoicePrediction
    };
}
