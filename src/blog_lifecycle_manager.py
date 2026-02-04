"""
Blog Lifecycle Manager v1.0

Manages blog post lifecycle based on prediction outcomes:
- Publishes posts when predictions go live
- Updates posts with match results
- Archives posts when predictions fail
- Handles scheduled content generation

Implements Google leak insights:
- Content freshness signals
- User engagement tracking
- Site authority building through consistent publishing
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

from .blog_content_engine import BlogContentEngine, BlogPost, generate_blog_for_prediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostStatus(Enum):
    """Blog post status values."""
    DRAFT = 'draft'
    SCHEDULED = 'scheduled'
    PUBLISHED = 'published'
    UPDATED = 'updated'  # Updated with result
    ARCHIVED = 'archived'  # Hidden due to failed prediction


class PredictionResult(Enum):
    """Prediction outcome values."""
    PENDING = 'pending'
    WIN = 'win'
    LOSS = 'loss'
    PUSH = 'push'  # Void/cancelled


class BlogLifecycleManager:
    """
    Manages blog posts based on prediction results.
    
    - Publishes when predictions go live
    - Archives when predictions fail
    - Updates with match results
    """
    
    def __init__(
        self,
        data_dir: str = 'data/blog_posts',
        archive_dir: str = 'data/blog_posts/archived'
    ):
        self.data_dir = data_dir
        self.archive_dir = archive_dir
        self.content_engine = BlogContentEngine(data_dir)
        self._ensure_dirs()
        
        # Index file for quick lookups
        self.index_file = os.path.join(data_dir, 'index.json')
        self._load_index()
    
    def _ensure_dirs(self):
        """Create required directories."""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)
    
    def _load_index(self):
        """Load or create the post index."""
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                'posts': {},  # post_id -> metadata
                'by_match': {},  # match_id -> post_id
                'by_prediction': {},  # prediction_id -> post_id
                'by_date': {},  # date -> [post_ids]
                'stats': {
                    'total_posts': 0,
                    'published': 0,
                    'archived': 0,
                    'wins': 0,
                    'losses': 0
                }
            }
    
    def _save_index(self):
        """Save the post index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    # =========================================================================
    # CORE LIFECYCLE METHODS
    # =========================================================================
    
    def generate_and_publish(
        self,
        prediction: Dict,
        template_style: str = None,
        publish_immediately: bool = True
    ) -> BlogPost:
        """
        Generate and optionally publish a blog post for a prediction.
        
        This is the main entry point for auto-posting when predictions go live.
        
        Args:
            prediction: The prediction data
            template_style: Optional specific template style
            publish_immediately: Whether to publish now or schedule
            
        Returns:
            The generated BlogPost
        """
        # Generate the blog post
        post = self.content_engine.generate_blog_post(prediction, template_style)
        
        # Set status
        if publish_immediately:
            post.status = PostStatus.PUBLISHED.value
            logger.info(f"Publishing blog post: {post.title}")
        else:
            post.status = PostStatus.SCHEDULED.value
            logger.info(f"Scheduled blog post: {post.title}")
        
        # Save the post
        self.content_engine.save_post(post)
        
        # Update index
        self._index_post(post)
        
        return post
    
    def generate_batch(
        self,
        predictions: List[Dict],
        publish: bool = True
    ) -> List[BlogPost]:
        """
        Generate blog posts for multiple predictions.
        
        Args:
            predictions: List of prediction data
            publish: Whether to publish immediately
            
        Returns:
            List of generated BlogPosts
        """
        posts = []
        
        for i, prediction in enumerate(predictions):
            # Vary templates across batch
            template_style = None  # Auto-select with variety
            
            try:
                post = self.generate_and_publish(
                    prediction,
                    template_style=template_style,
                    publish_immediately=publish
                )
                posts.append(post)
                logger.info(f"Generated post {i+1}/{len(predictions)}: {post.slug}")
            except Exception as e:
                logger.error(f"Error generating post for prediction: {e}")
                continue
        
        return posts
    
    def mark_result(
        self,
        match_id: str,
        result: PredictionResult,
        actual_score: str = None
    ) -> Optional[BlogPost]:
        """
        Update a blog post with the match result.
        
        If the prediction was wrong (LOSS), archive the post.
        If correct (WIN), update with success badge.
        
        Args:
            match_id: The match ID
            result: The prediction outcome
            actual_score: Optional actual match score
            
        Returns:
            Updated BlogPost or None
        """
        # Find post by match ID
        post_id = self.index.get('by_match', {}).get(match_id)
        
        if not post_id:
            logger.warning(f"No blog post found for match {match_id}")
            return None
        
        # Load the post
        post = self.content_engine.load_post(post_id)
        
        if not post:
            logger.error(f"Could not load post {post_id}")
            return None
        
        # Update post based on result
        post.result = result.value
        post.updated_at = datetime.now().isoformat()
        
        if result == PredictionResult.LOSS:
            # Archive failed prediction
            return self.archive_post(post_id, reason="Prediction failed")
        
        elif result == PredictionResult.WIN:
            # Update with success badge
            post.status = PostStatus.UPDATED.value
            post = self._add_result_to_content(post, result, actual_score)
            self.content_engine.save_post(post)
            
            # Update stats
            self.index['stats']['wins'] = self.index['stats'].get('wins', 0) + 1
            self._save_index()
            
            logger.info(f"Post {post_id} marked as WIN")
            return post
        
        elif result == PredictionResult.PUSH:
            # Void - just update status
            post.status = PostStatus.UPDATED.value
            self.content_engine.save_post(post)
            logger.info(f"Post {post_id} marked as PUSH (void)")
            return post
        
        return post
    
    def archive_post(
        self,
        post_id: str,
        reason: str = "Manual archive"
    ) -> Optional[BlogPost]:
        """
        Archive a blog post (soft delete).
        
        Moves the post to archived status and removes from main listings.
        
        Args:
            post_id: The post ID to archive
            reason: Reason for archiving
            
        Returns:
            Archived BlogPost or None
        """
        # Load the post
        post = self.content_engine.load_post(post_id)
        
        if not post:
            logger.error(f"Could not load post {post_id} for archiving")
            return None
        
        # Update status
        post.status = PostStatus.ARCHIVED.value
        post.updated_at = datetime.now().isoformat()
        
        # Move to archive directory
        source_path = os.path.join(self.data_dir, f"{post_id}.json")
        archive_path = os.path.join(self.archive_dir, f"{post_id}.json")
        
        with open(archive_path, 'w') as f:
            json.dump(post.to_dict(), f, indent=2)
        
        # Remove from main directory
        if os.path.exists(source_path):
            os.remove(source_path)
        
        # Update index
        if post_id in self.index.get('posts', {}):
            self.index['posts'][post_id]['status'] = PostStatus.ARCHIVED.value
        
        self.index['stats']['archived'] = self.index['stats'].get('archived', 0) + 1
        self.index['stats']['losses'] = self.index['stats'].get('losses', 0) + 1
        self._save_index()
        
        logger.info(f"Archived post {post_id}: {reason}")
        return post
    
    def restore_post(self, post_id: str) -> Optional[BlogPost]:
        """
        Restore an archived blog post.
        
        Args:
            post_id: The post ID to restore
            
        Returns:
            Restored BlogPost or None
        """
        archive_path = os.path.join(self.archive_dir, f"{post_id}.json")
        
        if not os.path.exists(archive_path):
            logger.error(f"Archived post {post_id} not found")
            return None
        
        with open(archive_path, 'r') as f:
            data = json.load(f)
        
        post = BlogPost(**data)
        post.status = PostStatus.PUBLISHED.value
        post.updated_at = datetime.now().isoformat()
        
        # Save back to main directory
        self.content_engine.save_post(post)
        
        # Remove from archive
        os.remove(archive_path)
        
        # Update index
        if post_id in self.index.get('posts', {}):
            self.index['posts'][post_id]['status'] = PostStatus.PUBLISHED.value
        self._save_index()
        
        logger.info(f"Restored post {post_id}")
        return post
    
    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================
    
    def _index_post(self, post: BlogPost):
        """Add post to the index."""
        self.index['posts'][post.id] = {
            'id': post.id,
            'slug': post.slug,
            'title': post.title,
            'match_id': post.match_id,
            'prediction_id': post.prediction_id,
            'home_team': post.home_team,
            'away_team': post.away_team,
            'match_date': post.match_date,
            'status': post.status,
            'created_at': post.created_at,
            'template_style': post.template_style
        }
        
        # Index by match ID
        self.index['by_match'][post.match_id] = post.id
        
        # Index by prediction ID
        self.index['by_prediction'][post.prediction_id] = post.id
        
        # Index by date
        if post.match_date not in self.index['by_date']:
            self.index['by_date'][post.match_date] = []
        if post.id not in self.index['by_date'][post.match_date]:
            self.index['by_date'][post.match_date].append(post.id)
        
        # Update stats
        self.index['stats']['total_posts'] = len(self.index['posts'])
        if post.status == PostStatus.PUBLISHED.value:
            self.index['stats']['published'] = self.index['stats'].get('published', 0) + 1
        
        self._save_index()
    
    # =========================================================================
    # CONTENT UPDATES
    # =========================================================================
    
    def _add_result_to_content(
        self,
        post: BlogPost,
        result: PredictionResult,
        actual_score: str = None
    ) -> BlogPost:
        """Add result banner to blog post content."""
        
        if result == PredictionResult.WIN:
            badge_class = 'success'
            badge_text = '✅ PREDICTION CORRECT!'
            message = f"Our prediction was spot on! Final score: {actual_score}" if actual_score else "Our prediction was correct!"
        elif result == PredictionResult.LOSS:
            badge_class = 'failed'
            badge_text = '❌ PREDICTION MISSED'
            message = f"Unfortunately, our prediction didn't hit. Final score: {actual_score}" if actual_score else "This prediction didn't work out."
        else:
            badge_class = 'void'
            badge_text = '⚪ MATCH VOID'
            message = "This match was cancelled or void."
        
        result_banner = f"""
        <div class="result-banner {badge_class}">
            <span class="badge">{badge_text}</span>
            <p>{message}</p>
            <time datetime="{datetime.now().isoformat()}">Updated: {datetime.now().strftime('%d %b %Y %H:%M')}</time>
        </div>
        """
        
        # Insert at beginning of content
        post.content_html = result_banner + post.content_html
        
        # Update text content
        post.content_text = f"{badge_text}\n{message}\n\n" + post.content_text
        
        return post
    
    # =========================================================================
    # RETRIEVAL METHODS
    # =========================================================================
    
    def get_published_posts(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict]:
        """Get list of published posts (for listing page)."""
        published = [
            meta for meta in self.index.get('posts', {}).values()
            if meta.get('status') == PostStatus.PUBLISHED.value
        ]
        
        # Sort by date, newest first
        published.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return published[offset:offset+limit]
    
    def get_posts_by_date(self, date: str) -> List[Dict]:
        """Get posts for a specific match date."""
        post_ids = self.index.get('by_date', {}).get(date, [])
        
        return [
            self.index['posts'][pid]
            for pid in post_ids
            if pid in self.index.get('posts', {})
        ]
    
    def get_post_by_slug(self, slug: str) -> Optional[BlogPost]:
        """Get a full post by its slug."""
        for meta in self.index.get('posts', {}).values():
            if meta.get('slug') == slug:
                return self.content_engine.load_post(meta['id'])
        return None
    
    def get_post_by_match(self, match_id: str) -> Optional[BlogPost]:
        """Get a post by match ID."""
        post_id = self.index.get('by_match', {}).get(match_id)
        if post_id:
            return self.content_engine.load_post(post_id)
        return None
    
    def get_stats(self) -> Dict:
        """Get publishing statistics."""
        stats = self.index.get('stats', {}).copy()
        
        # Calculate win rate
        total_results = stats.get('wins', 0) + stats.get('losses', 0)
        if total_results > 0:
            stats['win_rate'] = round(stats.get('wins', 0) / total_results * 100, 1)
        else:
            stats['win_rate'] = 0
        
        return stats
    
    # =========================================================================
    # MAINTENANCE
    # =========================================================================
    
    def check_and_update_results(self, match_results: Dict[str, Dict]) -> Dict:
        """
        Batch check match results and update posts.
        
        Args:
            match_results: Dict mapping match_id -> {result: str, score: str}
            
        Returns:
            Summary of updates
        """
        summary = {'updated': 0, 'archived': 0, 'errors': 0}
        
        for match_id, result_data in match_results.items():
            try:
                result_str = result_data.get('result', 'pending')
                result = PredictionResult[result_str.upper()]
                score = result_data.get('score')
                
                updated = self.mark_result(match_id, result, score)
                
                if updated:
                    if result == PredictionResult.LOSS:
                        summary['archived'] += 1
                    else:
                        summary['updated'] += 1
            except Exception as e:
                logger.error(f"Error updating result for {match_id}: {e}")
                summary['errors'] += 1
        
        return summary
    
    def cleanup_old_posts(self, days: int = 30) -> int:
        """
        Archive posts older than specified days.
        
        Args:
            days: Number of days after which to archive
            
        Returns:
            Number of posts archived
        """
        cutoff = datetime.now() - timedelta(days=days)
        archived_count = 0
        
        for post_id, meta in list(self.index.get('posts', {}).items()):
            if meta.get('status') == PostStatus.ARCHIVED.value:
                continue
            
            created = meta.get('created_at', '')
            if created:
                try:
                    created_dt = datetime.fromisoformat(created)
                    if created_dt < cutoff:
                        self.archive_post(post_id, reason="Old post cleanup")
                        archived_count += 1
                except ValueError:
                    continue
        
        return archived_count


# ============================================================================
# MODULE-LEVEL INSTANCE
# ============================================================================

blog_manager = BlogLifecycleManager()


def auto_publish_prediction(prediction: Dict) -> BlogPost:
    """Convenience function to auto-publish a prediction."""
    return blog_manager.generate_and_publish(prediction, publish_immediately=True)


def archive_failed_prediction(match_id: str) -> Optional[BlogPost]:
    """Convenience function to archive a failed prediction."""
    return blog_manager.mark_result(match_id, PredictionResult.LOSS)


def mark_prediction_win(match_id: str, actual_score: str = None) -> Optional[BlogPost]:
    """Convenience function to mark a prediction as won."""
    return blog_manager.mark_result(match_id, PredictionResult.WIN, actual_score)
