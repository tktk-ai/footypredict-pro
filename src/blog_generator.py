"""
Automated Blog Post Generator

Generates SEO-optimized blog posts based on daily predictions.
Runs automatically after model training/prediction generation.

Post Types:
- Daily Match Preview (morning)
- Top Picks Analysis
- Sure Wins Spotlight
- Weekly Review
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib

logger = logging.getLogger(__name__)

# Blog posts storage path
BLOG_DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'blog_posts')
os.makedirs(BLOG_DATA_PATH, exist_ok=True)


class BlogPostGenerator:
    """Generate automated blog posts from predictions."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load blog post templates."""
        return {
            'daily_preview': {
                'title_template': "⚽ {date} Football Predictions: {match_count} Matches Analyzed",
                'intro_template': "Our AI has analyzed {match_count} matches for {date}. Here are our top predictions with confidence ratings.",
            },
            'top_picks': {
                'title_template': "🔥 Top {count} Sure Bets for {date} - AI Analysis",
                'intro_template': "These are the highest confidence predictions for {date}. Each pick has been verified by our advanced AI model.",
            },
            'weekly_review': {
                'title_template': "📊 Weekly Football Predictions Review: {start_date} to {end_date}",
                'intro_template': "Let's review our prediction performance for the week. We achieved {accuracy}% accuracy across {total} matches.",
            }
        }
    
    def generate_daily_preview(self, predictions: List[Dict], date: str = None) -> Dict:
        """Generate a daily match preview blog post."""
        if not date:
            date = datetime.now().strftime('%B %d, %Y')
        
        date_slug = datetime.now().strftime('%Y-%m-%d')
        
        # Sort predictions by confidence
        sorted_preds = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Generate content sections
        sections = []
        
        # High confidence section (80%+)
        high_conf = [p for p in sorted_preds if p.get('confidence', 0) >= 0.80]
        if high_conf:
            sections.append({
                'title': '🎯 High Confidence Picks',
                'description': 'These matches have 80%+ confidence ratings',
                'matches': self._format_matches(high_conf[:5])
            })
        
        # Medium confidence section (60-79%)
        med_conf = [p for p in sorted_preds if 0.60 <= p.get('confidence', 0) < 0.80]
        if med_conf:
            sections.append({
                'title': '💎 Value Picks',
                'description': 'Good value selections with solid confidence',
                'matches': self._format_matches(med_conf[:5])
            })
        
        # Goals predictions
        goals_preds = [p for p in sorted_preds if p.get('goals_prediction')]
        if goals_preds:
            sections.append({
                'title': '⚽ Goals Market Analysis',
                'description': 'Over/Under and BTTS predictions',
                'matches': self._format_goals_predictions(goals_preds[:5])
            })
        
        # Generate meta
        post_id = f"daily-preview-{date_slug}"
        
        post = {
            'id': post_id,
            'slug': post_id,
            'type': 'daily_preview',
            'title': f"⚽ {date} Football Predictions: {len(predictions)} Matches Analyzed",
            'excerpt': f"AI-powered predictions for {len(predictions)} matches today. Our top pick has {int(high_conf[0]['confidence']*100) if high_conf else 0}% confidence.",
            'date': datetime.now().isoformat(),
            'date_display': date,
            'author': 'FootyPredict AI',
            'category': 'Daily Predictions',
            'tags': ['football predictions', 'daily tips', 'ai betting tips', date_slug],
            'featured_image': '/icons/icon.svg',
            'stats': {
                'total_matches': len(predictions),
                'high_confidence': len(high_conf),
                'avg_confidence': sum(p.get('confidence', 0) for p in predictions) / len(predictions) if predictions else 0
            },
            'sections': sections,
            'top_pick': high_conf[0] if high_conf else None,
            'seo': {
                'meta_title': f"Football Predictions {date} - AI Analysis & Tips",
                'meta_description': f"Expert AI football predictions for {date}. {len(predictions)} matches analyzed with {len(high_conf)} high-confidence picks.",
                'keywords': ['football predictions', 'betting tips', 'ai predictions', 'sure wins']
            }
        }
        
        # Save post
        self._save_post(post)
        
        return post
    
    def generate_top_picks(self, predictions: List[Dict], count: int = 5) -> Dict:
        """Generate a top picks spotlight post."""
        date = datetime.now().strftime('%B %d, %Y')
        date_slug = datetime.now().strftime('%Y-%m-%d')
        
        # Get top picks by confidence
        top = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)[:count]
        
        picks = []
        for i, pred in enumerate(top, 1):
            confidence = int(pred.get('confidence', 0) * 100)
            home = pred.get('home_team', 'Home')
            away = pred.get('away_team', 'Away')
            
            pick = {
                'rank': i,
                'match': f"{home} vs {away}",
                'home_team': home,
                'away_team': away,
                'league': pred.get('league', 'Unknown'),
                'kickoff': pred.get('kickoff', ''),
                'confidence': confidence,
                'prediction': pred.get('prediction', 'Home Win'),
                'analysis': self._generate_match_analysis(pred),
                'odds': pred.get('odds', {})
            }
            picks.append(pick)
        
        post_id = f"top-picks-{date_slug}"
        
        post = {
            'id': post_id,
            'slug': post_id,
            'type': 'top_picks',
            'title': f"🔥 Top {count} Sure Bets for {date} - AI Analysis",
            'excerpt': f"Our {count} highest confidence picks for today. #1 rated at {picks[0]['confidence']}% confidence.",
            'date': datetime.now().isoformat(),
            'date_display': date,
            'author': 'FootyPredict AI',
            'category': 'Top Picks',
            'tags': ['top picks', 'sure bets', 'high confidence', date_slug],
            'featured_image': '/icons/icon.svg',
            'picks': picks,
            'seo': {
                'meta_title': f"Top {count} Football Betting Tips {date} - Sure Wins",
                'meta_description': f"AI-verified top {count} football predictions for {date}. Highest confidence: {picks[0]['confidence']}%.",
                'keywords': ['top picks', 'sure bets', 'football tips', 'winning predictions']
            }
        }
        
        self._save_post(post)
        return post
    
    def generate_weekly_review(self, results: List[Dict], start_date: str, end_date: str) -> Dict:
        """Generate a weekly performance review post."""
        date_slug = datetime.now().strftime('%Y-W%W')
        
        # Calculate stats
        total = len(results)
        wins = len([r for r in results if r.get('correct', False)])
        accuracy = (wins / total * 100) if total > 0 else 0
        
        # Performance by league
        by_league = {}
        for r in results:
            league = r.get('league', 'Unknown')
            if league not in by_league:
                by_league[league] = {'total': 0, 'wins': 0}
            by_league[league]['total'] += 1
            if r.get('correct'):
                by_league[league]['wins'] += 1
        
        league_stats = [
            {
                'league': league,
                'total': stats['total'],
                'wins': stats['wins'],
                'accuracy': round(stats['wins'] / stats['total'] * 100, 1) if stats['total'] > 0 else 0
            }
            for league, stats in by_league.items()
        ]
        league_stats.sort(key=lambda x: x['accuracy'], reverse=True)
        
        post_id = f"weekly-review-{date_slug}"
        
        post = {
            'id': post_id,
            'slug': post_id,
            'type': 'weekly_review',
            'title': f"📊 Weekly Football Predictions Review: {start_date} to {end_date}",
            'excerpt': f"This week we achieved {accuracy:.1f}% accuracy across {total} predictions.",
            'date': datetime.now().isoformat(),
            'author': 'FootyPredict AI',
            'category': 'Weekly Review',
            'tags': ['weekly review', 'accuracy report', 'prediction results', date_slug],
            'featured_image': '/icons/icon.svg',
            'stats': {
                'total': total,
                'wins': wins,
                'losses': total - wins,
                'accuracy': round(accuracy, 1)
            },
            'league_performance': league_stats,
            'top_picks_review': results[:5],
            'seo': {
                'meta_title': f"Football Predictions Review - Week Performance {accuracy:.1f}%",
                'meta_description': f"Weekly prediction review: {wins}/{total} correct ({accuracy:.1f}% accuracy). See league breakdown.",
                'keywords': ['prediction review', 'accuracy', 'football betting results']
            }
        }
        
        self._save_post(post)
        return post
    
    def _format_matches(self, predictions: List[Dict]) -> List[Dict]:
        """Format match predictions for display."""
        formatted = []
        for pred in predictions:
            formatted.append({
                'home_team': pred.get('home_team', 'Home'),
                'away_team': pred.get('away_team', 'Away'),
                'league': pred.get('league', ''),
                'kickoff': pred.get('kickoff', ''),
                'prediction': pred.get('prediction', ''),
                'confidence': int(pred.get('confidence', 0) * 100),
                'odds': pred.get('odds', {})
            })
        return formatted
    
    def _format_goals_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Format goals market predictions."""
        formatted = []
        for pred in predictions:
            goals = pred.get('goals_prediction', {})
            formatted.append({
                'home_team': pred.get('home_team', 'Home'),
                'away_team': pred.get('away_team', 'Away'),
                'over_25': goals.get('over_25', 0),
                'btts': goals.get('btts', 0),
                'prediction': 'Over 2.5' if goals.get('over_25', 0) > 0.6 else 'Under 2.5'
            })
        return formatted
    
    def _generate_match_analysis(self, prediction: Dict) -> str:
        """Generate a brief analysis for a match."""
        home = prediction.get('home_team', 'Home')
        away = prediction.get('away_team', 'Away')
        confidence = int(prediction.get('confidence', 0) * 100)
        pred_type = prediction.get('prediction', 'Home Win')
        
        templates = [
            f"Our AI model gives this match a {confidence}% confidence rating. {home} has shown strong recent form.",
            f"With {confidence}% confidence, this is one of our top selections for today.",
            f"Statistical analysis favors {pred_type} at {confidence}% probability.",
        ]
        
        # Simple hash to pick a template consistently
        hash_val = int(hashlib.md5(f"{home}{away}".encode()).hexdigest()[:8], 16)
        return templates[hash_val % len(templates)]
    
    def _save_post(self, post: Dict) -> str:
        """Save blog post to JSON file."""
        filename = f"{post['id']}.json"
        filepath = os.path.join(BLOG_DATA_PATH, filename)
        
        with open(filepath, 'w') as f:
            json.dump(post, f, indent=2)
        
        logger.info(f"Saved blog post: {filename}")
        return filepath
    
    def get_recent_posts(self, limit: int = 10) -> List[Dict]:
        """Get recent blog posts."""
        posts = []
        
        if os.path.exists(BLOG_DATA_PATH):
            for filename in os.listdir(BLOG_DATA_PATH):
                if filename.endswith('.json'):
                    filepath = os.path.join(BLOG_DATA_PATH, filename)
                    try:
                        with open(filepath, 'r') as f:
                            post = json.load(f)
                            posts.append(post)
                    except Exception as e:
                        logger.error(f"Error loading post {filename}: {e}")
        
        # Sort by date, newest first
        posts.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return posts[:limit]
    
    def get_post_by_slug(self, slug: str) -> Optional[Dict]:
        """Get a blog post by its slug."""
        filepath = os.path.join(BLOG_DATA_PATH, f"{slug}.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        return None


# Global instance
blog_generator = BlogPostGenerator()


def generate_daily_blog_posts(predictions: List[Dict] = None) -> Dict:
    """
    Generate all daily blog posts.
    Called after prediction generation.
    """
    from src.scheduler import prediction_cache
    
    # Get predictions from cache if not provided
    if predictions is None:
        predictions = prediction_cache.get_all_predictions(limit=100)
    
    if not predictions:
        logger.warning("No predictions available for blog generation")
        return {'error': 'No predictions available'}
    
    results = {}
    
    # Generate daily preview
    try:
        preview = blog_generator.generate_daily_preview(predictions)
        results['daily_preview'] = preview['id']
        logger.info(f"Generated daily preview: {preview['id']}")
    except Exception as e:
        logger.error(f"Error generating daily preview: {e}")
        results['daily_preview_error'] = str(e)
    
    # Generate top picks
    try:
        top_picks = blog_generator.generate_top_picks(predictions, count=5)
        results['top_picks'] = top_picks['id']
        logger.info(f"Generated top picks: {top_picks['id']}")
    except Exception as e:
        logger.error(f"Error generating top picks: {e}")
        results['top_picks_error'] = str(e)
    
    results['total_predictions'] = len(predictions)
    results['generated_at'] = datetime.now().isoformat()
    
    return results


def get_blog_posts_api() -> List[Dict]:
    """API endpoint to get blog posts."""
    return blog_generator.get_recent_posts(limit=20)


def get_blog_post_api(slug: str) -> Optional[Dict]:
    """API endpoint to get a single blog post."""
    return blog_generator.get_post_by_slug(slug)
