import json
import os
from datetime import datetime, timedelta
from typing import List, Dict
from lib.news_aggregator import NewsAggregator, NewsItem
from lib.sms import send_sms

class NewsTracker:
    def __init__(self):
        self.news_aggregator = NewsAggregator()
        self.news_history_file = "resources/news_history.json"
        self.ensure_news_history_file()
    
    def ensure_news_history_file(self):
        """Ensure news history file exists"""
        if not os.path.exists(self.news_history_file):
            with open(self.news_history_file, "w") as f:
                json.dump({}, f)
    
    def load_news_history(self) -> Dict:
        """Load news history from file"""
        try:
            with open(self.news_history_file, "r") as f:
                return json.load(f)
        except:
            return {}
    
    def save_news_history(self, history: Dict):
        """Save news history to file"""
        with open(self.news_history_file, "w") as f:
            json.dump(history, f)
    
    def create_news_key(self, news_item: NewsItem) -> str:
        """Create a unique key for news item to prevent duplicates"""
        # Use title + source + date to create unique key
        date_str = news_item.published.strftime("%Y-%m-%d")
        return f"{news_item.source}_{date_str}_{hash(news_item.title[:50])}"
    
    def has_news_been_sent(self, news_item: NewsItem) -> bool:
        """Check if news has already been sent"""
        history = self.load_news_history()
        news_key = self.create_news_key(news_item)
        
        # Check if this exact news was sent in the last 24 hours
        for symbol in news_item.symbols_mentioned:
            if symbol in history:
                for sent_key in history[symbol]:
                    if news_key == sent_key:
                        return True
        
        return False
    
    def mark_news_as_sent(self, news_item: NewsItem):
        """Mark news as sent to prevent duplicates"""
        history = self.load_news_history()
        news_key = self.create_news_key(news_item)
        
        for symbol in news_item.symbols_mentioned:
            if symbol not in history:
                history[symbol] = []
            
            if news_key not in history[symbol]:
                history[symbol].append(news_key)
        
        # Clean up old entries (older than 7 days)
        self.cleanup_old_news_history(history)
        self.save_news_history(history)
    
    def cleanup_old_news_history(self, history: Dict):
        """Remove news history older than 7 days"""
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for symbol in list(history.keys()):
            cleaned_entries = []
            for entry in history[symbol]:
                try:
                    # Extract date from entry key
                    parts = entry.split('_')
                    if len(parts) >= 3:
                        date_str = parts[1]  # Format: YYYY-MM-DD
                        entry_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if entry_date >= cutoff_date:
                            cleaned_entries.append(entry)
                except:
                    # Keep entry if we can't parse date (safer)
                    cleaned_entries.append(entry)
            
            history[symbol] = cleaned_entries
    
    def check_and_send_news_alerts(self):
        """Check for news and send SMS alerts"""
        print("Checking for news alerts...")
        
        try:
            # Get latest news
            latest_news = self.news_aggregator.get_latest_news()
            
            if not latest_news:
                print("No relevant news found")
                return
            
            news_sent_count = 0
            
            for news_item in latest_news:
                # Check if we've already sent this news
                if self.has_news_been_sent(news_item):
                    continue
                
                # Format and send SMS
                sms_message = self.news_aggregator.format_news_for_sms(news_item)
                
                try:
                    send_sms(sms_message)
                    self.mark_news_as_sent(news_item)
                    news_sent_count += 1
                    
                    print(f"Sent news alert: {sms_message}")
                    
                    # Limit to 3 news alerts per check to avoid spam
                    if news_sent_count >= 3:
                        break
                        
                except Exception as e:
                    print(f"Error sending news SMS: {e}")
            
            if news_sent_count == 0:
                print("No new news alerts to send")
            else:
                print(f"Sent {news_sent_count} news alerts")
                
        except Exception as e:
            print(f"Error in news checking: {e}")
    
    def get_news_summary_for_symbol(self, symbol: str) -> str:
        """Get a summary of recent news for a specific symbol"""
        try:
            # Reload company mappings for this symbol
            self.news_aggregator.load_company_mappings()
            
            # Get all news and filter for this symbol
            all_news = self.news_aggregator.get_latest_news()
            symbol_news = [news for news in all_news if symbol.upper() in [s.upper() for s in news.symbols_mentioned]]
            
            if not symbol_news:
                return f"No recent news found for {symbol.upper()}"
            
            # Format summary
            summary = f"Recent news for {symbol.upper()}:\n"
            for i, news in enumerate(symbol_news[:3], 1):  # Top 3 news items
                sentiment_emoji = {'positive': '📈', 'negative': '📉', 'neutral': '📰'}
                emoji = sentiment_emoji.get(news.sentiment, '📰')
                summary += f"{i}. {emoji} {news.title} ({news.source})\n"
            
            return summary.strip()
            
        except Exception as e:
            return f"Error getting news for {symbol}: {str(e)}"

def track_news():
    """Main function to be called by scheduler"""
    news_tracker = NewsTracker()
    news_tracker.check_and_send_news_alerts()