"""
Ads Manager - Google AdSense & Affiliate Integration

Manages ad placements and affiliate links for monetization.
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# GOOGLE ADSENSE CONFIGURATION
# ============================================================================

ADSENSE_CONFIG = {
    'client_id': 'ca-pub-XXXXXXXXXX',  # Replace with actual pub ID
    'enabled': True,
    'auto_ads': True,
    'placements': {
        'header_banner': {
            'slot': 'XXXXXXXXXX',
            'format': 'auto',
            'size': '728x90',
            'position': 'top',
            'lazy_load': False
        },
        'sidebar_rectangle': {
            'slot': 'XXXXXXXXXX',
            'format': 'auto',
            'size': '300x250',
            'position': 'sidebar',
            'lazy_load': True
        },
        'in_article': {
            'slot': 'XXXXXXXXXX',
            'format': 'fluid',
            'layout': 'in-article',
            'position': 'content',
            'lazy_load': True
        },
        'footer_banner': {
            'slot': 'XXXXXXXXXX',
            'format': 'auto',
            'size': '728x90',
            'position': 'footer',
            'lazy_load': True
        },
        'mobile_anchor': {
            'slot': 'XXXXXXXXXX',
            'format': 'auto',
            'position': 'anchor',
            'enabled_mobile_only': True
        }
    }
}


# ============================================================================
# AFFILIATE NETWORKS
# ============================================================================

AFFILIATE_NETWORKS = {
    'bet365': {
        'name': 'Bet365',
        'url': 'https://www.bet365.com/#/AS/B/',
        'tracking_param': 'aff_id=XXXXX',
        'bonus': 'Up to $100 in Bet Credits',
        'bonus_code': None,
        'cta_primary': 'Get Best Odds at Bet365 →',
        'cta_secondary': 'Claim Your Bet365 Bonus',
        'logo': '/static/images/affiliates/bet365.png',
        'rating': 4.8,
        'priority': 1,
        'geo_allowed': ['US', 'UK', 'AU', 'CA', 'DE', 'NG', 'GH', 'KE'],
        'geo_blocked': ['FR', 'ES', 'IT']
    },
    'betway': {
        'name': 'Betway',
        'url': 'https://www.betway.com/',
        'tracking_param': 'affid=XXXXX',
        'bonus': '100% Match up to $250',
        'bonus_code': 'WELCOME250',
        'cta_primary': 'Join Betway Now →',
        'cta_secondary': 'Get $250 Bonus',
        'logo': '/static/images/affiliates/betway.png',
        'rating': 4.6,
        'priority': 2,
        'geo_allowed': ['UK', 'DE', 'NG', 'GH', 'KE', 'ZA'],
        'geo_blocked': ['US', 'AU', 'FR']
    },
    '1xbet': {
        'name': '1xBet',
        'url': 'https://1xbet.com/',
        'tracking_param': 'partner=XXXXX',
        'bonus': '100% up to €130',
        'bonus_code': 'FOOTYPRO',
        'cta_primary': 'Join 1xBet →',
        'cta_secondary': 'Get €130 Bonus',
        'logo': '/static/images/affiliates/1xbet.png',
        'rating': 4.5,
        'priority': 3,
        'geo_allowed': ['NG', 'GH', 'KE', 'TZ', 'UG'],
        'geo_blocked': ['US', 'UK', 'AU']
    },
    'sportybet': {
        'name': 'SportyBet',
        'url': 'https://www.sportybet.com/',
        'tracking_param': 'ref=XXXXX',
        'bonus': '100% First Deposit Bonus',
        'bonus_code': None,
        'cta_primary': 'Bet on SportyBet →',
        'cta_secondary': 'Get First Deposit Bonus',
        'logo': '/static/images/affiliates/sportybet.png',
        'rating': 4.4,
        'priority': 4,
        'geo_allowed': ['NG', 'GH', 'KE', 'TZ', 'UG', 'ZM'],
        'geo_blocked': ['US', 'UK', 'AU', 'EU']
    },
    'parimatch': {
        'name': 'Parimatch',
        'url': 'https://www.parimatch.com/',
        'tracking_param': 'promo=XXXXX',
        'bonus': 'Up to $150 Welcome Bonus',
        'bonus_code': None,
        'cta_primary': 'Join Parimatch →',
        'cta_secondary': 'Claim $150 Bonus',
        'logo': '/static/images/affiliates/parimatch.png',
        'rating': 4.3,
        'priority': 5,
        'geo_allowed': ['NG', 'KE', 'TZ', 'IN'],
        'geo_blocked': ['US', 'UK', 'AU', 'EU']
    }
}


# ============================================================================
# ADS MANAGER CLASS
# ============================================================================

class AdsManager:
    """
    Manages ad placements and affiliate links.
    """
    
    def __init__(self):
        self.adsense = ADSENSE_CONFIG
        self.affiliates = AFFILIATE_NETWORKS
    
    # ========================================================================
    # ADSENSE METHODS
    # ========================================================================
    
    def get_ad_code(self, placement: str, lazy_load: bool = True) -> str:
        """
        Generate AdSense ad code for a specific placement.
        """
        if not self.adsense['enabled']:
            return '<!-- Ads disabled -->'
        
        config = self.adsense['placements'].get(placement)
        if not config:
            return '<!-- Unknown ad placement -->'
        
        client = self.adsense['client_id']
        slot = config['slot']
        format_type = config.get('format', 'auto')
        
        # Build ad code
        lazy_attr = 'loading="lazy"' if lazy_load and config.get('lazy_load', True) else ''
        
        if config.get('layout') == 'in-article':
            return f'''
            <ins class="adsbygoogle"
                 style="display:block; text-align:center;"
                 data-ad-layout="in-article"
                 data-ad-format="fluid"
                 data-ad-client="{client}"
                 data-ad-slot="{slot}"
                 {lazy_attr}></ins>
            <script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>
            '''
        else:
            return f'''
            <ins class="adsbygoogle"
                 style="display:block"
                 data-ad-client="{client}"
                 data-ad-slot="{slot}"
                 data-ad-format="{format_type}"
                 data-full-width-responsive="true"
                 {lazy_attr}></ins>
            <script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>
            '''
    
    def get_auto_ads_script(self) -> str:
        """Get the auto ads script for the head section."""
        if not self.adsense['enabled'] or not self.adsense['auto_ads']:
            return ''
        
        return f'''
        <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client={self.adsense['client_id']}"
             crossorigin="anonymous"></script>
        '''
    
    # ========================================================================
    # AFFILIATE METHODS
    # ========================================================================
    
    def get_affiliate_link(self, network: str) -> Optional[str]:
        """Get tracked affiliate link for a network."""
        affiliate = self.affiliates.get(network)
        if not affiliate:
            return None
        
        url = affiliate['url']
        tracking = affiliate.get('tracking_param', '')
        
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}{tracking}" if tracking else url
    
    def get_affiliate_cta(self, network: str, cta_type: str = 'primary') -> Dict:
        """Get affiliate CTA with all details."""
        affiliate = self.affiliates.get(network)
        if not affiliate:
            return {}
        
        cta_key = f'cta_{cta_type}'
        
        return {
            'name': affiliate['name'],
            'url': self.get_affiliate_link(network),
            'cta': affiliate.get(cta_key, affiliate.get('cta_primary')),
            'bonus': affiliate.get('bonus', ''),
            'bonus_code': affiliate.get('bonus_code'),
            'logo': affiliate.get('logo'),
            'rating': affiliate.get('rating')
        }
    
    def get_affiliates_for_geo(self, country_code: str) -> List[Dict]:
        """
        Get affiliates allowed for a specific country.
        Returns sorted by priority.
        """
        allowed = []
        
        for key, affiliate in self.affiliates.items():
            geo_allowed = affiliate.get('geo_allowed', [])
            geo_blocked = affiliate.get('geo_blocked', [])
            
            # Check if allowed
            if country_code in geo_blocked:
                continue
            
            if '*' in geo_allowed or country_code in geo_allowed:
                allowed.append({
                    'key': key,
                    **affiliate,
                    'url': self.get_affiliate_link(key)
                })
        
        # Sort by priority
        allowed.sort(key=lambda x: x.get('priority', 99))
        
        return allowed
    
    def get_odds_comparison_table(self, match: Dict, country_code: str = 'NG') -> List[Dict]:
        """
        Generate odds comparison table with affiliate links.
        """
        affiliates = self.get_affiliates_for_geo(country_code)
        
        # Simulated odds (would come from real API)
        comparison = []
        for affiliate in affiliates[:5]:
            comparison.append({
                'bookmaker': affiliate['name'],
                'logo': affiliate.get('logo'),
                'url': affiliate['url'],
                'bonus': affiliate.get('bonus'),
                'home_odds': 1.80 + (hash(affiliate['name']) % 20) / 100,
                'draw_odds': 3.40 + (hash(affiliate['name'] + 'draw') % 30) / 100,
                'away_odds': 4.00 + (hash(affiliate['name'] + 'away') % 40) / 100
            })
        
        return comparison
    
    # ========================================================================
    # CTA GENERATION
    # ========================================================================
    
    def generate_cta_html(self, network: str, style: str = 'box') -> str:
        """Generate HTML for affiliate CTA."""
        cta = self.get_affiliate_cta(network)
        if not cta:
            return ''
        
        if style == 'box':
            return f'''
            <div class="affiliate-cta-box">
                <div class="cta-content">
                    <h4>🎁 {cta['bonus']}</h4>
                    {f"<p>Use code: <strong>{cta['bonus_code']}</strong></p>" if cta.get('bonus_code') else ''}
                    <a href="{cta['url']}" class="cta-button" target="_blank" rel="nofollow sponsored noopener">
                        {cta['cta']}
                    </a>
                </div>
            </div>
            '''
        elif style == 'inline':
            return f'''
            <a href="{cta['url']}" class="affiliate-inline-cta" target="_blank" rel="nofollow sponsored noopener">
                {cta['cta']} - {cta['bonus']}
            </a>
            '''
        elif style == 'button':
            return f'''
            <a href="{cta['url']}" class="affiliate-button" target="_blank" rel="nofollow sponsored noopener">
                {cta['cta']}
            </a>
            '''
        
        return ''
    
    def generate_comparison_html(self, match: Dict, country_code: str = 'NG') -> str:
        """Generate odds comparison table HTML."""
        comparison = self.get_odds_comparison_table(match, country_code)
        
        if not comparison:
            return ''
        
        html = '''
        <div class="odds-comparison">
            <h3>📊 Best Odds Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th>Bookmaker</th>
                        <th>Home</th>
                        <th>Draw</th>
                        <th>Away</th>
                        <th>Bonus</th>
                    </tr>
                </thead>
                <tbody>
        '''
        
        for row in comparison:
            html += f'''
                <tr>
                    <td>
                        <a href="{row['url']}" target="_blank" rel="nofollow sponsored">
                            {row['bookmaker']}
                        </a>
                    </td>
                    <td>{row['home_odds']:.2f}</td>
                    <td>{row['draw_odds']:.2f}</td>
                    <td>{row['away_odds']:.2f}</td>
                    <td><small>{row['bonus']}</small></td>
                </tr>
            '''
        
        html += '''
                </tbody>
            </table>
            <p class="disclaimer">*Odds subject to change. 18+ | Gamble Responsibly</p>
        </div>
        '''
        
        return html


# ============================================================================
# SITEMAP GENERATOR
# ============================================================================

class SitemapGenerator:
    """Generate XML sitemap for SEO."""
    
    def __init__(self, base_url: str = 'https://footypredict.pro'):
        self.base_url = base_url
    
    def generate_sitemap(self, posts: List[Dict], static_pages: List[str] = None) -> str:
        """Generate XML sitemap."""
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml += '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        
        # Static pages
        static = static_pages or ['/', '/blog', '/smart-accas', '/vip', '/live', '/about', '/contact']
        for page in static:
            xml += f'''
            <url>
                <loc>{self.base_url}{page}</loc>
                <changefreq>daily</changefreq>
                <priority>0.8</priority>
            </url>
            '''
        
        # Blog posts
        for post in posts:
            slug = post.get('slug', '')
            date = post.get('date', datetime.now().isoformat())[:10]
            xml += f'''
            <url>
                <loc>{self.base_url}/blog/{slug}</loc>
                <lastmod>{date}</lastmod>
                <changefreq>weekly</changefreq>
                <priority>0.6</priority>
            </url>
            '''
        
        xml += '</urlset>'
        return xml
    
    def save_sitemap(self, posts: List[Dict], filepath: str = 'static/sitemap.xml'):
        """Save sitemap to file."""
        sitemap = self.generate_sitemap(posts)
        
        import os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write(sitemap)
        
        logger.info(f"Sitemap saved to {filepath}")
        return filepath


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

ads_manager = AdsManager()
sitemap_generator = SitemapGenerator()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_ad_code(placement: str) -> str:
    """Shortcut to get ad code."""
    return ads_manager.get_ad_code(placement)


def get_affiliate_cta(network: str) -> Dict:
    """Shortcut to get affiliate CTA."""
    return ads_manager.get_affiliate_cta(network)


def get_geo_affiliates(country: str) -> List[Dict]:
    """Get affiliates for a country."""
    return ads_manager.get_affiliates_for_geo(country)


def regenerate_sitemap():
    """Regenerate sitemap with latest posts."""
    from src.seo_blog_generator import seo_blog_generator
    posts = seo_blog_generator.get_recent_posts(limit=100)
    return sitemap_generator.save_sitemap(posts)
