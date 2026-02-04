"""
Complete League Configuration

All leagues from around the world including:
- Top 5 European leagues + lower divisions
- All UEFA competitions
- National team competitions
- South American leagues
- Asian leagues
- African leagues
- North American leagues
- Oceania leagues
"""

# Complete league configuration with API sources
LEAGUES = {
    # ============================================
    # GERMANY (Free via OpenLigaDB)
    # ============================================
    'bundesliga': {
        'name': 'Bundesliga',
        'country': '🇩🇪',
        'api': 'openligadb',
        'code': 'bl1',
        'tier': 'free'
    },
    'bundesliga2': {
        'name': '2. Bundesliga',
        'country': '🇩🇪',
        'api': 'openligadb',
        'code': 'bl2',
        'tier': 'free'
    },
    '3liga': {
        'name': '3. Liga',
        'country': '🇩🇪',
        'api': 'openligadb',
        'code': 'bl3',
        'tier': 'free'
    },
    'dfb_pokal': {
        'name': 'DFB-Pokal',
        'country': '🇩🇪',
        'api': 'openligadb',
        'code': 'dfb',
        'tier': 'free'
    },
    
    # ============================================
    # ENGLAND
    # ============================================
    'premier_league': {
        'name': 'Premier League',
        'country': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'api': 'football-data',
        'code': 'PL',
        'tier': 'premium'
    },
    'championship': {
        'name': 'Championship',
        'country': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'api': 'football-data',
        'code': 'ELC',
        'tier': 'premium'
    },
    'league_one': {
        'name': 'League One',
        'country': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'api': 'thesportsdb',
        'code': 'league_one',
        'tier': 'free'
    },
    'league_two': {
        'name': 'League Two',
        'country': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'api': 'thesportsdb',
        'code': 'league_two',
        'tier': 'free'
    },
    'fa_cup': {
        'name': 'FA Cup',
        'country': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'api': 'thesportsdb',
        'code': 'fa_cup',
        'tier': 'free'
    },
    'efl_cup': {
        'name': 'EFL Cup',
        'country': '🏴󠁧󠁢󠁥󠁮󠁧󠁿',
        'api': 'thesportsdb',
        'code': 'efl_cup',
        'tier': 'free'
    },
    
    # ============================================
    # SPAIN
    # ============================================
    'la_liga': {
        'name': 'La Liga',
        'country': '🇪🇸',
        'api': 'football-data',
        'code': 'PD',
        'tier': 'premium'
    },
    'la_liga2': {
        'name': 'La Liga 2',
        'country': '🇪🇸',
        'api': 'thesportsdb',
        'code': 'laliga2',
        'tier': 'free'
    },
    'copa_del_rey': {
        'name': 'Copa del Rey',
        'country': '🇪🇸',
        'api': 'thesportsdb',
        'code': 'copa_del_rey',
        'tier': 'free'
    },
    
    # ============================================
    # ITALY
    # ============================================
    'serie_a': {
        'name': 'Serie A',
        'country': '🇮🇹',
        'api': 'football-data',
        'code': 'SA',
        'tier': 'premium'
    },
    'serie_b': {
        'name': 'Serie B',
        'country': '🇮🇹',
        'api': 'thesportsdb',
        'code': 'serie_b',
        'tier': 'free'
    },
    'coppa_italia': {
        'name': 'Coppa Italia',
        'country': '🇮🇹',
        'api': 'thesportsdb',
        'code': 'coppa_italia',
        'tier': 'free'
    },
    
    # ============================================
    # FRANCE
    # ============================================
    'ligue_1': {
        'name': 'Ligue 1',
        'country': '🇫🇷',
        'api': 'football-data',
        'code': 'FL1',
        'tier': 'premium'
    },
    'ligue_2': {
        'name': 'Ligue 2',
        'country': '🇫🇷',
        'api': 'thesportsdb',
        'code': 'ligue_2',
        'tier': 'free'
    },
    'coupe_de_france': {
        'name': 'Coupe de France',
        'country': '🇫🇷',
        'api': 'thesportsdb',
        'code': 'coupe_de_france',
        'tier': 'free'
    },
    
    # ============================================
    # NETHERLANDS
    # ============================================
    'eredivisie': {
        'name': 'Eredivisie',
        'country': '🇳🇱',
        'api': 'football-data',
        'code': 'DED',
        'tier': 'premium'
    },
    'eerste_divisie': {
        'name': 'Eerste Divisie',
        'country': '🇳🇱',
        'api': 'thesportsdb',
        'code': 'eerste_divisie',
        'tier': 'free'
    },
    
    # ============================================
    # PORTUGAL
    # ============================================
    'primeira_liga': {
        'name': 'Primeira Liga',
        'country': '🇵🇹',
        'api': 'football-data',
        'code': 'PPL',
        'tier': 'premium'
    },
    'liga_portugal_2': {
        'name': 'Liga Portugal 2',
        'country': '🇵🇹',
        'api': 'thesportsdb',
        'code': 'liga_portugal_2',
        'tier': 'free'
    },
    'taca_de_portugal': {
        'name': 'Taça de Portugal',
        'country': '🇵🇹',
        'api': 'thesportsdb',
        'code': 'taca_portugal',
        'tier': 'free'
    },
    
    # ============================================
    # BELGIUM
    # ============================================
    'jupiler_pro': {
        'name': 'Jupiler Pro League',
        'country': '🇧🇪',
        'api': 'thesportsdb',
        'code': 'jupiler',
        'tier': 'free'
    },
    
    # ============================================
    # TURKEY
    # ============================================
    'super_lig': {
        'name': 'Süper Lig',
        'country': '🇹🇷',
        'api': 'thesportsdb',
        'code': 'super_lig',
        'tier': 'free'
    },
    'tff_first': {
        'name': 'TFF First League',
        'country': '🇹🇷',
        'api': 'thesportsdb',
        'code': 'tff_first',
        'tier': 'free'
    },
    
    # ============================================
    # GREECE
    # ============================================
    'super_league_greece': {
        'name': 'Super League Greece',
        'country': '🇬🇷',
        'api': 'thesportsdb',
        'code': 'super_league_greece',
        'tier': 'free'
    },
    
    # ============================================
    # SCOTLAND
    # ============================================
    'scottish_prem': {
        'name': 'Scottish Premiership',
        'country': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
        'api': 'thesportsdb',
        'code': 'scottish_prem',
        'tier': 'free'
    },
    'scottish_championship': {
        'name': 'Scottish Championship',
        'country': '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
        'api': 'thesportsdb',
        'code': 'scottish_champ',
        'tier': 'free'
    },
    
    # ============================================
    # AUSTRIA
    # ============================================
    'austrian_bundesliga': {
        'name': 'Austrian Bundesliga',
        'country': '🇦🇹',
        'api': 'thesportsdb',
        'code': 'austrian_bundesliga',
        'tier': 'free'
    },
    
    # ============================================
    # SWITZERLAND
    # ============================================
    'swiss_super_league': {
        'name': 'Swiss Super League',
        'country': '🇨🇭',
        'api': 'thesportsdb',
        'code': 'swiss_super',
        'tier': 'free'
    },
    
    # ============================================
    # RUSSIA
    # ============================================
    'russian_premier': {
        'name': 'Russian Premier League',
        'country': '🇷🇺',
        'api': 'thesportsdb',
        'code': 'russian_premier',
        'tier': 'free'
    },
    
    # ============================================
    # UKRAINE
    # ============================================
    'ukrainian_premier': {
        'name': 'Ukrainian Premier League',
        'country': '🇺🇦',
        'api': 'thesportsdb',
        'code': 'ukrainian_premier',
        'tier': 'free'
    },
    
    # ============================================
    # POLAND
    # ============================================
    'ekstraklasa': {
        'name': 'Ekstraklasa',
        'country': '🇵🇱',
        'api': 'thesportsdb',
        'code': 'ekstraklasa',
        'tier': 'free'
    },
    
    # ============================================
    # CZECH REPUBLIC
    # ============================================
    'czech_first_league': {
        'name': 'Czech First League',
        'country': '🇨🇿',
        'api': 'thesportsdb',
        'code': 'czech_first',
        'tier': 'free'
    },
    
    # ============================================
    # CROATIA
    # ============================================
    'hnl': {
        'name': 'Hrvatska Nogometna Liga',
        'country': '🇭🇷',
        'api': 'thesportsdb',
        'code': 'hnl',
        'tier': 'free'
    },
    
    # ============================================
    # SERBIA
    # ============================================
    'serbian_superliga': {
        'name': 'Serbian SuperLiga',
        'country': '🇷🇸',
        'api': 'thesportsdb',
        'code': 'serbian_super',
        'tier': 'free'
    },
    
    # ============================================
    # DENMARK
    # ============================================
    'danish_superliga': {
        'name': 'Danish Superliga',
        'country': '🇩🇰',
        'api': 'thesportsdb',
        'code': 'danish_super',
        'tier': 'free'
    },
    
    # ============================================
    # SWEDEN
    # ============================================
    'allsvenskan': {
        'name': 'Allsvenskan',
        'country': '🇸🇪',
        'api': 'thesportsdb',
        'code': 'allsvenskan',
        'tier': 'free'
    },
    
    # ============================================
    # NORWAY
    # ============================================
    'eliteserien': {
        'name': 'Eliteserien',
        'country': '🇳🇴',
        'api': 'thesportsdb',
        'code': 'eliteserien',
        'tier': 'free'
    },
    
    # ============================================
    # EUROPEAN COMPETITIONS
    # ============================================
    'champions_league': {
        'name': 'UEFA Champions League',
        'country': '🏆',
        'api': 'football-data',
        'code': 'CL',
        'tier': 'free'
    },
    'europa_league': {
        'name': 'UEFA Europa League',
        'country': '🏆',
        'api': 'thesportsdb',
        'code': 'europa',
        'tier': 'free'
    },
    'conference_league': {
        'name': 'UEFA Conference League',
        'country': '🏆',
        'api': 'thesportsdb',
        'code': 'conference',
        'tier': 'free'
    },
    'euro_qualifiers': {
        'name': 'Euro Qualifiers',
        'country': '🏆',
        'api': 'thesportsdb',
        'code': 'euro_qual',
        'tier': 'free'
    },
    'nations_league': {
        'name': 'UEFA Nations League',
        'country': '🏆',
        'api': 'thesportsdb',
        'code': 'nations_league',
        'tier': 'free'
    },
    
    # ============================================
    # SOUTH AMERICA
    # ============================================
    'brasileirao': {
        'name': 'Brasileirão Serie A',
        'country': '🇧🇷',
        'api': 'football-data',
        'code': 'BSA',
        'tier': 'premium'
    },
    'brasileirao_b': {
        'name': 'Brasileirão Serie B',
        'country': '🇧🇷',
        'api': 'thesportsdb',
        'code': 'brasileirao_b',
        'tier': 'free'
    },
    'copa_do_brasil': {
        'name': 'Copa do Brasil',
        'country': '🇧🇷',
        'api': 'thesportsdb',
        'code': 'copa_brasil',
        'tier': 'free'
    },
    'liga_argentina': {
        'name': 'Liga Profesional Argentina',
        'country': '🇦🇷',
        'api': 'thesportsdb',
        'code': 'liga_argentina',
        'tier': 'free'
    },
    'copa_libertadores': {
        'name': 'Copa Libertadores',
        'country': '🌎',
        'api': 'thesportsdb',
        'code': 'libertadores',
        'tier': 'free'
    },
    'copa_sudamericana': {
        'name': 'Copa Sudamericana',
        'country': '🌎',
        'api': 'thesportsdb',
        'code': 'sudamericana',
        'tier': 'free'
    },
    'liga_mx': {
        'name': 'Liga MX',
        'country': '🇲🇽',
        'api': 'thesportsdb',
        'code': 'liga_mx',
        'tier': 'free'
    },
    'mls': {
        'name': 'MLS',
        'country': '🇺🇸',
        'api': 'thesportsdb',
        'code': 'mls',
        'tier': 'free'
    },
    
    # ============================================
    # ASIA
    # ============================================
    'j_league': {
        'name': 'J1 League',
        'country': '🇯🇵',
        'api': 'thesportsdb',
        'code': 'j_league',
        'tier': 'free'
    },
    'k_league': {
        'name': 'K League 1',
        'country': '🇰🇷',
        'api': 'thesportsdb',
        'code': 'k_league',
        'tier': 'free'
    },
    'chinese_super': {
        'name': 'Chinese Super League',
        'country': '🇨🇳',
        'api': 'thesportsdb',
        'code': 'chinese_super',
        'tier': 'free'
    },
    'saudi_pro': {
        'name': 'Saudi Pro League',
        'country': '🇸🇦',
        'api': 'thesportsdb',
        'code': 'saudi_pro',
        'tier': 'free'
    },
    'a_league': {
        'name': 'A-League',
        'country': '🇦🇺',
        'api': 'thesportsdb',
        'code': 'a_league',
        'tier': 'free'
    },
    'indian_super': {
        'name': 'Indian Super League',
        'country': '🇮🇳',
        'api': 'thesportsdb',
        'code': 'indian_super',
        'tier': 'free'
    },
    'afc_champions': {
        'name': 'AFC Champions League',
        'country': '🌏',
        'api': 'thesportsdb',
        'code': 'afc_champions',
        'tier': 'free'
    },
    
    # ============================================
    # AFRICA
    # ============================================
    'egyptian_premier': {
        'name': 'Egyptian Premier League',
        'country': '🇪🇬',
        'api': 'thesportsdb',
        'code': 'egyptian_premier',
        'tier': 'free'
    },
    'south_african_psl': {
        'name': 'SA Premier Soccer League',
        'country': '🇿🇦',
        'api': 'thesportsdb',
        'code': 'sa_psl',
        'tier': 'free'
    },
    'caf_champions': {
        'name': 'CAF Champions League',
        'country': '🌍',
        'api': 'thesportsdb',
        'code': 'caf_champions',
        'tier': 'free'
    },
    'afcon': {
        'name': 'Africa Cup of Nations',
        'country': '🌍',
        'api': 'thesportsdb',
        'code': 'afcon',
        'tier': 'free'
    },
    
    # ============================================
    # NATIONAL TEAMS - MAJOR TOURNAMENTS
    # ============================================
    'world_cup': {
        'name': 'FIFA World Cup',
        'country': '🌍',
        'api': 'football-data',
        'code': 'WC',
        'tier': 'free'
    },
    'world_cup_qualifiers': {
        'name': 'World Cup Qualifiers',
        'country': '🌍',
        'api': 'thesportsdb',
        'code': 'wc_qual',
        'tier': 'free'
    },
    'euros': {
        'name': 'UEFA European Championship',
        'country': '🏆',
        'api': 'football-data',
        'code': 'EC',
        'tier': 'free'
    },
    'copa_america': {
        'name': 'Copa América',
        'country': '🌎',
        'api': 'thesportsdb',
        'code': 'copa_america',
        'tier': 'free'
    },
    'gold_cup': {
        'name': 'CONCACAF Gold Cup',
        'country': '🌎',
        'api': 'thesportsdb',
        'code': 'gold_cup',
        'tier': 'free'
    },
    'asian_cup': {
        'name': 'AFC Asian Cup',
        'country': '🌏',
        'api': 'thesportsdb',
        'code': 'asian_cup',
        'tier': 'free'
    },
    
    # ============================================
    # INTERNATIONAL FRIENDLIES
    # ============================================
    'friendlies': {
        'name': 'International Friendlies',
        'country': '🌍',
        'api': 'thesportsdb',
        'code': 'friendlies',
        'tier': 'free'
    },
}

# National teams for predictions
NATIONAL_TEAMS = {
    # Europe
    'Germany': {'code': 'GER', 'flag': '🇩🇪', 'elo': 1950},
    'France': {'code': 'FRA', 'flag': '🇫🇷', 'elo': 2005},
    'England': {'code': 'ENG', 'flag': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'elo': 1985},
    'Spain': {'code': 'ESP', 'flag': '🇪🇸', 'elo': 1970},
    'Italy': {'code': 'ITA', 'flag': '🇮🇹', 'elo': 1930},
    'Netherlands': {'code': 'NED', 'flag': '🇳🇱', 'elo': 1940},
    'Portugal': {'code': 'POR', 'flag': '🇵🇹', 'elo': 1960},
    'Belgium': {'code': 'BEL', 'flag': '🇧🇪', 'elo': 1925},
    'Croatia': {'code': 'CRO', 'flag': '🇭🇷', 'elo': 1890},
    'Switzerland': {'code': 'SUI', 'flag': '🇨🇭', 'elo': 1850},
    'Denmark': {'code': 'DEN', 'flag': '🇩🇰', 'elo': 1835},
    'Austria': {'code': 'AUT', 'flag': '🇦🇹', 'elo': 1810},
    'Poland': {'code': 'POL', 'flag': '🇵🇱', 'elo': 1780},
    'Turkey': {'code': 'TUR', 'flag': '🇹🇷', 'elo': 1760},
    'Serbia': {'code': 'SRB', 'flag': '🇷🇸', 'elo': 1755},
    'Ukraine': {'code': 'UKR', 'flag': '🇺🇦', 'elo': 1745},
    'Sweden': {'code': 'SWE', 'flag': '🇸🇪', 'elo': 1740},
    'Czech Republic': {'code': 'CZE', 'flag': '🇨🇿', 'elo': 1720},
    'Scotland': {'code': 'SCO', 'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'elo': 1700},
    'Hungary': {'code': 'HUN', 'flag': '🇭🇺', 'elo': 1680},
    'Wales': {'code': 'WAL', 'flag': '🏴󠁧󠁢󠁷󠁬󠁳󠁿', 'elo': 1660},
    'Greece': {'code': 'GRE', 'flag': '🇬🇷', 'elo': 1640},
    'Norway': {'code': 'NOR', 'flag': '🇳🇴', 'elo': 1660},
    'Ireland': {'code': 'IRL', 'flag': '🇮🇪', 'elo': 1620},
    'Romania': {'code': 'ROU', 'flag': '🇷🇴', 'elo': 1650},
    'Slovakia': {'code': 'SVK', 'flag': '🇸🇰', 'elo': 1630},
    'Slovenia': {'code': 'SVN', 'flag': '🇸🇮', 'elo': 1610},
    'Finland': {'code': 'FIN', 'flag': '🇫🇮', 'elo': 1590},
    'Iceland': {'code': 'ISL', 'flag': '🇮🇸', 'elo': 1570},
    'Albania': {'code': 'ALB', 'flag': '🇦🇱', 'elo': 1550},
    'Russia': {'code': 'RUS', 'flag': '🇷🇺', 'elo': 1720},
    
    # South America
    'Argentina': {'code': 'ARG', 'flag': '🇦🇷', 'elo': 2060},
    'Brazil': {'code': 'BRA', 'flag': '🇧🇷', 'elo': 2020},
    'Uruguay': {'code': 'URU', 'flag': '🇺🇾', 'elo': 1870},
    'Colombia': {'code': 'COL', 'flag': '🇨🇴', 'elo': 1830},
    'Chile': {'code': 'CHI', 'flag': '🇨🇱', 'elo': 1760},
    'Ecuador': {'code': 'ECU', 'flag': '🇪🇨', 'elo': 1730},
    'Peru': {'code': 'PER', 'flag': '🇵🇪', 'elo': 1700},
    'Paraguay': {'code': 'PAR', 'flag': '🇵🇾', 'elo': 1650},
    'Venezuela': {'code': 'VEN', 'flag': '🇻🇪', 'elo': 1600},
    'Bolivia': {'code': 'BOL', 'flag': '🇧🇴', 'elo': 1500},
    
    # North/Central America
    'Mexico': {'code': 'MEX', 'flag': '🇲🇽', 'elo': 1820},
    'USA': {'code': 'USA', 'flag': '🇺🇸', 'elo': 1790},
    'Canada': {'code': 'CAN', 'flag': '🇨🇦', 'elo': 1720},
    'Costa Rica': {'code': 'CRC', 'flag': '🇨🇷', 'elo': 1640},
    'Jamaica': {'code': 'JAM', 'flag': '🇯🇲', 'elo': 1580},
    'Panama': {'code': 'PAN', 'flag': '🇵🇦', 'elo': 1560},
    'Honduras': {'code': 'HON', 'flag': '🇭🇳', 'elo': 1520},
    
    # Asia
    'Japan': {'code': 'JPN', 'flag': '🇯🇵', 'elo': 1800},
    'South Korea': {'code': 'KOR', 'flag': '🇰🇷', 'elo': 1780},
    'Australia': {'code': 'AUS', 'flag': '🇦🇺', 'elo': 1720},
    'Iran': {'code': 'IRN', 'flag': '🇮🇷', 'elo': 1740},
    'Saudi Arabia': {'code': 'KSA', 'flag': '🇸🇦', 'elo': 1650},
    'Qatar': {'code': 'QAT', 'flag': '🇶🇦', 'elo': 1580},
    'Iraq': {'code': 'IRQ', 'flag': '🇮🇶', 'elo': 1560},
    'UAE': {'code': 'UAE', 'flag': '🇦🇪', 'elo': 1540},
    'China': {'code': 'CHN', 'flag': '🇨🇳', 'elo': 1500},
    'India': {'code': 'IND', 'flag': '🇮🇳', 'elo': 1350},
    
    # Africa
    'Morocco': {'code': 'MAR', 'flag': '🇲🇦', 'elo': 1850},
    'Senegal': {'code': 'SEN', 'flag': '🇸🇳', 'elo': 1820},
    'Nigeria': {'code': 'NGA', 'flag': '🇳🇬', 'elo': 1750},
    'Egypt': {'code': 'EGY', 'flag': '🇪🇬', 'elo': 1690},
    'Algeria': {'code': 'ALG', 'flag': '🇩🇿', 'elo': 1680},
    'Tunisia': {'code': 'TUN', 'flag': '🇹🇳', 'elo': 1660},
    'Cameroon': {'code': 'CMR', 'flag': '🇨🇲', 'elo': 1700},
    'Ghana': {'code': 'GHA', 'flag': '🇬🇭', 'elo': 1640},
    'Ivory Coast': {'code': 'CIV', 'flag': '🇨🇮', 'elo': 1720},
    'South Africa': {'code': 'RSA', 'flag': '🇿🇦', 'elo': 1560},
    'Mali': {'code': 'MLI', 'flag': '🇲🇱', 'elo': 1600},
    'DR Congo': {'code': 'COD', 'flag': '🇨🇩', 'elo': 1580},
}


def get_all_leagues():
    """Get all available leagues"""
    return LEAGUES


def get_leagues_by_region():
    """Get leagues grouped by region"""
    regions = {
        '🇩🇪 Germany': [],
        '🏴󠁧󠁢󠁥󠁮󠁧󠁿 England': [],
        '🇪🇸 Spain': [],
        '🇮🇹 Italy': [],
        '🇫🇷 France': [],
        '🇳🇱 Netherlands': [],
        '🇵🇹 Portugal': [],
        '🏆 European Cups': [],
        '🌍 International': [],
        '🌎 Americas': [],
        '🌏 Asia & Oceania': [],
        '🌍 Africa': [],
        '🇪🇺 Other Europe': [],
    }
    
    for code, league in LEAGUES.items():
        country = league['country']
        name = league['name']
        
        if country == '🇩🇪':
            regions['🇩🇪 Germany'].append((code, name))
        elif country in ['🏴󠁧󠁢󠁥󠁮󠁧󠁿']:
            regions['🏴󠁧󠁢󠁥󠁮󠁧󠁿 England'].append((code, name))
        elif country == '🇪🇸':
            regions['🇪🇸 Spain'].append((code, name))
        elif country == '🇮🇹':
            regions['🇮🇹 Italy'].append((code, name))
        elif country == '🇫🇷':
            regions['🇫🇷 France'].append((code, name))
        elif country == '🇳🇱':
            regions['🇳🇱 Netherlands'].append((code, name))
        elif country == '🇵🇹':
            regions['🇵🇹 Portugal'].append((code, name))
        elif country == '🏆':
            regions['🏆 European Cups'].append((code, name))
        elif country in ['🌍', '🌎', '🌏'] and 'World' in name or 'Friendly' in name:
            regions['🌍 International'].append((code, name))
        elif country in ['🇧🇷', '🇦🇷', '🇲🇽', '🇺🇸', '🌎']:
            regions['🌎 Americas'].append((code, name))
        elif country in ['🇯🇵', '🇰🇷', '🇨🇳', '🇦🇺', '🇸🇦', '🇮🇳', '🌏']:
            regions['🌏 Asia & Oceania'].append((code, name))
        elif country in ['🇪🇬', '🇿🇦'] or 'Africa' in name or 'CAF' in name:
            regions['🌍 Africa'].append((code, name))
        else:
            regions['🇪🇺 Other Europe'].append((code, name))
    
    return regions


def get_national_teams():
    """Get all national teams"""
    return NATIONAL_TEAMS


def get_team_elo(team_name: str) -> int:
    """Get ELO rating for a national team"""
    team = NATIONAL_TEAMS.get(team_name)
    return team['elo'] if team else 1500


def get_league_count():
    """Get total number of leagues"""
    return len(LEAGUES)


def get_team_count():
    """Get total number of national teams"""
    return len(NATIONAL_TEAMS)
