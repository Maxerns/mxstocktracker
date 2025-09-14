import json
import os
from lib.agent import run_research_pipeline
import asyncio
from datetime import date

from lib.stock_checker import get_stock_price
from lib.technical_analysis import analyze_stock_technical
from lib.sms import send_sms
from lib.news_tracker import track_news

def track_stocks():
  print("tracking stocks started...")

  with open("resources/tracker_list.json", "r") as f:
    tracker_list = json.load(f)

  print("Tracking stocks: ", tracker_list)

  for symbol in tracker_list:
    stock_info = get_stock_price(symbol)

    # Check for technical analysis alerts first
    technical_alerts = analyze_stock_technical(symbol)
    
    # Filter for high priority alerts (priority 4-5) to avoid spam
    high_priority_alerts = [alert for alert in technical_alerts if alert.priority >= 4]
    
    # Check alert history for technical alerts
    with open("resources/alert_history.json", "r") as f:
      alert_history = json.load(f)

    if symbol not in alert_history:
      alert_history[symbol] = []

    # Send technical analysis alerts
    for alert in high_priority_alerts:
      alert_key = f"{alert.alert_type.value}_{str(date.today())}"
      
      # Check if we've already sent this type of alert today
      if alert_key not in alert_history[symbol]:
        alert_history[symbol].append(alert_key)
        
        # Create SMS message for technical alert
        sms_message = f"{symbol} TECH ALERT: {alert.message} | Price: ${alert.current_price:.2f}"
        send_sms(sms_message)
        print(f"Sent technical alert for {symbol}: {alert.alert_type.value}")

    # If the stock price is 5% + or - from the previous close, run the research pipeline
    if (stock_info.current_price > stock_info.previous_close * 1.01) or (stock_info.current_price < stock_info.previous_close * 0.99):
      # Check if we have already alerted the user today for price movement

      price_alert_key = f"PRICE_MOVE_{str(date.today())}"
      if price_alert_key not in alert_history[symbol]:
        alert_history[symbol].append(price_alert_key)
        
        # Save updated alert history
        with open("resources/alert_history.json", "w") as f:
          json.dump(alert_history, f)

        asyncio.run(run_research_pipeline(symbol, stock_info.current_price, stock_info.previous_close))
      else:
        print(f"Already alerted user about {symbol} price movement today.")
    
    # Save updated alert history (for technical alerts)
    with open("resources/alert_history.json", "w") as f:
      json.dump(alert_history, f)

  # Check for news alerts after processing all stocks
  print("Checking for news alerts...")
  track_news()