# Prediction Tracking API Documentation

## Overview

The FootyPredict Pro prediction tracking system allows you to:

- Track predictions automatically
- Verify results and calculate accuracy
- View leaderboards and statistics
- Generate performance reports

## API Endpoints

### 📊 Statistics

#### `GET /api/tracker/stats`

Get overall prediction tracking statistics.

**Response:**

```json
{
  "success": true,
  "total_predictions": 32,
  "verified": 32,
  "pending": 0,
  "won": 22,
  "lost": 10,
  "accuracy": 68.8,
  "by_league": {
    "bundesliga": { "accuracy": 75.0, "total": 8 },
    "premier_league": { "accuracy": 75.0, "total": 8 }
  },
  "by_confidence": {
    "high_75+": { "accuracy": 100.0, "total": 6, "won": 6 },
    "medium_55_75": { "accuracy": 50.0, "total": 8, "won": 4 }
  }
}
```

#### `GET /api/monitor/stats`

Dashboard statistics (integrates with tracker).

---

### 📝 Track & Verify Predictions

#### `POST /api/tracker/add`

Add a new prediction to track.

**Request Body:**

```json
{
  "home": "Bayern Munich",
  "away": "Borussia Dortmund",
  "league": "bundesliga",
  "prediction": "home",
  "confidence": 0.85,
  "date": "2025-01-25"
}
```

#### `POST /api/tracker/verify`

Verify a prediction with actual result.

**Request Body:**

```json
{
  "home": "Bayern Munich",
  "away": "Borussia Dortmund",
  "score": "2-1",
  "outcome": "home"
}
```

Or by ID:

```json
{
  "id": "pred_20250125_0001",
  "score": "2-1",
  "outcome": "home"
}
```

#### `GET /api/tracker/pending`

Get predictions awaiting results.

#### `GET /api/tracker/recent?limit=20`

Get recent tracked predictions.

---

### 🏆 Leaderboard

#### `GET /api/bet-tracker/leaderboard`

Get rankings by league performance.

**Response:**

```json
{
  "success": true,
  "leaderboard": [
    {
      "rank": 1,
      "username": "Bundesliga Tracker",
      "accuracy": 75.0,
      "predictions": 8,
      "streak": 3,
      "roi": 11.2
    }
  ]
}
```

---

### 🔧 Utilities

#### `POST /api/tracker/seed`

Seed sample predictions for demo/testing.

#### `POST /api/tracker/auto-track`

Auto-track today's predictions.

**Request Body:**

```json
{
  "league": "bundesliga"
}
```

---

## Usage Examples

### cURL Examples

```bash
# Get stats
curl http://localhost:5000/api/tracker/stats

# Add a prediction
curl -X POST http://localhost:5000/api/tracker/add \
  -H "Content-Type: application/json" \
  -d '{"home":"Bayern","away":"Dortmund","league":"bundesliga","prediction":"home","confidence":0.85}'

# Verify a prediction
curl -X POST http://localhost:5000/api/tracker/verify \
  -H "Content-Type: application/json" \
  -d '{"home":"Bayern","away":"Dortmund","score":"2-1","outcome":"home"}'

# Seed sample data
curl -X POST http://localhost:5000/api/tracker/seed

# Auto-track today's matches
curl -X POST http://localhost:5000/api/tracker/auto-track \
  -H "Content-Type: application/json" \
  -d '{"league":"premier_league"}'
```

### JavaScript Examples

```javascript
// Add prediction
fetch("/api/tracker/add", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    home: "Liverpool",
    away: "Arsenal",
    league: "premier_league",
    prediction: "draw",
    confidence: 0.67,
  }),
});

// Get stats
fetch("/api/tracker/stats")
  .then((r) => r.json())
  .then((data) => console.log("Accuracy:", data.accuracy + "%"));
```

---

## Data Flow

1. **Make Prediction** → Call `/api/tracker/add`
2. **Match Played** → Results become available
3. **Verify Result** → Call `/api/tracker/verify`
4. **Track Performance** → View `/api/tracker/stats`
5. **Leaderboard** → Rankings at `/api/bet-tracker/leaderboard`
