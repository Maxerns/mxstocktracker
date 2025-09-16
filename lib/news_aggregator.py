import feedparser
import requests
from bs4 import BeautifulSoup
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import time
from textblob import TextBlob
import yfinance as yf

@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    source: str
    published: datetime
    symbols_mentioned: List[str]
    sentiment: str  # 'positive', 'negative', 'neutral'
    relevance_score: float

class NewsAggregator:
    def __init__(self):
        self.news_sources = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'benzinga': 'https://feeds.benzinga.com/benzinga',
            'seeking_alpha': 'https://seekingalpha.com/market_currents.xml',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
            'bloomberg': 'https://feeds.bloomberg.com/markets/news.rss'
        }
        
        # Load company name mappings for better symbol detection
        self.company_mappings = {}
        self.load_company_mappings()
    
    def load_company_mappings(self):
        """Load company names for tracked stocks to improve news detection"""
        try:
            with open("resources/tracker_list.json", "r") as f:
                symbols = json.load(f)
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    if 'longName' in info:
                        self.company_mappings[symbol] = [
                            info['longName'].lower(),
                            symbol.lower()
                        ]
                        # Add common variations
                        if 'shortName' in info:
                            self.company_mappings[symbol].append(info['shortName'].lower())
                except:
                    # Fallback to just the symbol
                    self.company_mappings[symbol] = [symbol.lower()]
        except:
            pass

    def fetch_rss_news(self, url: str, source_name: str) -> List[NewsItem]:
        """Fetch news from RSS feeds"""
        try:
            feed = feedparser.parse(url)
            news_items = []
            
            for entry in feed.entries[:10]:  # Limit to latest 10 items
                try:
                    # Parse publication date
                    published = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6])
                    
                    # Skip old news (older than 24 hours)
                    if published < datetime.now() - timedelta(hours=24):
                        continue
                    
                    title = entry.title if hasattr(entry, 'title') else ''
                    summary = entry.summary if hasattr(entry, 'summary') else ''
                    url = entry.link if hasattr(entry, 'link') else ''
                    
                    # Find mentioned symbols
                    symbols_mentioned = self.extract_symbols(title + ' ' + summary)
                    
                    if symbols_mentioned:  # Only process if relevant symbols found
                        sentiment = self.analyze_sentiment(title + ' ' + summary)
                        relevance_score = self.calculate_relevance_score(title, summary, symbols_mentioned)
                        
                        news_items.append(NewsItem(
                            title=title,
                            summary=summary,
                            url=url,
                            source=source_name,
                            published=published,
                            symbols_mentioned=symbols_mentioned,
                            sentiment=sentiment,
                            relevance_score=relevance_score
                        ))
                except Exception as e:
                    print(f"Error processing RSS entry from {source_name}: {e}")
                    continue
            
            return news_items
        except Exception as e:
            print(f"Error fetching RSS from {source_name}: {e}")
            return []

    def fetch_web_news(self) -> List[NewsItem]:
        """Fetch news from web scraping"""
        news_items = []
        
        # Yahoo Finance specific scraping
        try:
            with open("resources/tracker_list.json", "r") as f:
                symbols = json.load(f)
            
            for symbol in symbols:
                url = f"https://finance.yahoo.com/quote/{symbol}/news"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find news articles
                    articles = soup.find_all('h3', class_=re.compile('.*headline.*|.*title.*'))
                    
                    for article in articles[:5]:  # Limit to 5 per symbol
                        try:
                            title = article.get_text().strip()
                            link_elem = article.find('a')
                            url = link_elem['href'] if link_elem else ''
                            
                            if url and not url.startswith('http'):
                                url = 'https://finance.yahoo.com' + url
                            
                            if title and len(title) > 10:
                                sentiment = self.analyze_sentiment(title)
                                relevance_score = self.calculate_relevance_score(title, '', [symbol])
                                
                                news_items.append(NewsItem(
                                    title=title,
                                    summary='',
                                    url=url,
                                    source='yahoo_finance_web',
                                    published=datetime.now(),
                                    symbols_mentioned=[symbol],
                                    sentiment=sentiment,
                                    relevance_score=relevance_score
                                ))
                        except Exception as e:
                            continue
                    
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print(f"Error scraping Yahoo Finance for {symbol}: {e}")
                    continue
        except:
            pass
        
        return news_items

    def extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text"""
        symbols_found = []
        text_lower = text.lower()
        
        try:
            with open("resources/tracker_list.json", "r") as f:
                tracked_symbols = json.load(f)
        except:
            return []
        
        for symbol in tracked_symbols:
            # Direct symbol match
            if symbol.lower() in text_lower:
                symbols_found.append(symbol)
                continue
            
            # Company name match
            if symbol in self.company_mappings:
                for company_name in self.company_mappings[symbol]:
                    if company_name in text_lower:
                        symbols_found.append(symbol)
                        break
        
        return list(set(symbols_found))  # Remove duplicates

    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of news text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        except:
            return 'neutral'

    def calculate_relevance_score(self, title: str, summary: str, symbols: List[str]) -> float:
        """Calculate how relevant the news is (0-1 scale)"""
        score = 0.5  # Base score
        
        text = (title + ' ' + summary).lower()
        
        # Keywords that increase relevance
        high_impact_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'merger', 'acquisition',
            'partnership', 'contract', 'lawsuit', 'fda approval', 'recall',
            'bankruptcy', 'dividend', 'split', 'ipo', 'delisting', 'upgrade',
            'downgrade', 'target price', 'analyst', 'forecast', 'guidance'
        ]
        
        medium_impact_keywords = [
            'quarterly', 'annual', 'report', 'results', 'announcement',
            'ceo', 'executive', 'management', 'board', 'director'
        ]
        
        for keyword in high_impact_keywords:
            if keyword in text:
                score += 0.2
        
        for keyword in medium_impact_keywords:
            if keyword in text:
                score += 0.1
        
        # Symbol prominence in title
        for symbol in symbols:
            if symbol.lower() in title.lower():
                score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0

    def get_latest_news(self) -> List[NewsItem]:
        """Get latest news from all sources"""
        all_news = []
        
        # Fetch from RSS sources
        for source_name, url in self.news_sources.items():
            news_items = self.fetch_rss_news(url, source_name)
            all_news.extend(news_items)
            time.sleep(0.5)  # Rate limiting
        
        # Fetch from web scraping
        web_news = self.fetch_web_news()
        all_news.extend(web_news)
        
        # Remove duplicates and sort by relevance
        unique_news = self.deduplicate_news(all_news)
        
        # Filter by relevance score (only high relevance news)
        filtered_news = [news for news in unique_news if news.relevance_score >= 0.5]
        
        # Sort by relevance score and recency
        filtered_news.sort(key=lambda x: (x.relevance_score, x.published), reverse=True)
        
        return filtered_news[:10]  # Return top 10 most relevant

    def deduplicate_news(self, news_items: List[NewsItem]) -> List[NewsItem]:
        """Remove duplicate news items based on title similarity"""
        unique_news = []
        seen_titles = set()
        
        for item in news_items:
            # Create a normalized title for comparison
            normalized_title = re.sub(r'[^\w\s]', '', item.title.lower()).strip()
            
            # Check if we've seen a very similar title
            is_duplicate = False
            for seen_title in seen_titles:
                if self.calculate_title_similarity(normalized_title, seen_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(normalized_title)
                unique_news.append(item)
        
        return unique_news

    def calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

    def format_news_for_sms(self, news_item: NewsItem) -> str:
        """Format news item for SMS (160 char limit)"""
        symbols = ', '.join(news_item.symbols_mentioned)
        sentiment_emoji = {
            'positive': '📈',
            'negative': '📉',
            'neutral': '📰'
        }
        
        emoji = sentiment_emoji.get(news_item.sentiment, '📰')
        
        # Create concise message
        message = f"{emoji} {symbols}: {news_item.title}"
        
        # Truncate if too long, leaving space for source
        if len(message) > 140:
            message = message[:137] + "..."
        
        message += f" ({news_item.source})"
        
        return message