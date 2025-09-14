from phi.tools import Toolkit
from typing import List
import yfinance as yf
from pydantic import BaseModel
import json
from lib.stock_checker import StockPriceResponse, get_stock_price
from lib.technical_analysis import analyze_stock_technical

class StockTrackerTools(Toolkit):
    def __init__(self):
        super().__init__(name="stock_tracker_tools")

        self.register(self.add_stock_to_tracker)
        self.register(self.remove_stock_from_tracker)
        self.register(self.get_stock_tracker_list)
        self.register(self.get_stock_price_info)
        self.register(self.get_technical_analysis)

    def add_stock_to_tracker(self, symbol: str) -> str:
        """Add a stock symbol to the tracker list"""
        with open("resources/tracker_list.json", "r") as f:
            tracker_list = json.load(f)

        if symbol not in tracker_list:
            tracker_list.append(symbol.upper())

            with open("resources/tracker_list.json", "w") as f:
                json.dump(tracker_list, f)
            
            return f"Added {symbol.upper()} to tracker list"
        else:
            return f"{symbol.upper()} is already in tracker list"

    def remove_stock_from_tracker(self, symbol: str) -> str:
        """Remove a stock symbol from the tracker list"""
        with open("resources/tracker_list.json", "r") as f:
            tracker_list = json.load(f)

        if symbol.upper() in tracker_list:
            tracker_list.remove(symbol.upper())

            with open("resources/tracker_list.json", "w") as f:
                json.dump(tracker_list, f)
            
            return f"Removed {symbol.upper()} from tracker list"
        else:
            return f"{symbol.upper()} is not in tracker list"

    def get_stock_tracker_list(self) -> str:
        """Get the current list of tracked stocks"""
        with open("resources/tracker_list.json", "r") as f:
            print('getting tracker list')
            tracker_list = json.load(f)
            print(tracker_list)
        return f"Currently tracking: {', '.join(tracker_list)}"

    def get_stock_price_info(self, symbol: str) -> str:
        """Get stock price information for a symbol"""
        try:
            price_info = get_stock_price(symbol)
            change_percent = ((price_info.current_price / price_info.previous_close) - 1) * 100
            
            return f"""Stock: {symbol.upper()}
Current Price: ${price_info.current_price:.2f}
Previous Close: ${price_info.previous_close:.2f}
Change: {change_percent:+.2f}%"""
        except Exception as e:
            return f"Error getting stock price for {symbol}: {str(e)}"

    def get_technical_analysis(self, symbol: str) -> str:
        """Get technical analysis alerts for a stock symbol"""
        try:
            alerts = analyze_stock_technical(symbol)
            
            if not alerts:
                return f"No technical alerts found for {symbol.upper()}"
            
            # Sort by priority (highest first)
            alerts.sort(key=lambda x: x.priority, reverse=True)
            
            result = f"Technical Analysis for {symbol.upper()}:\n"
            for alert in alerts[:5]:  # Limit to top 5 alerts
                result += f"• {alert.message} (Priority: {alert.priority})\n"
            
            return result
            
        except Exception as e:
            return f"Error getting technical analysis for {symbol}: {str(e)}"

# Create instances of the tools for use in agents
stock_tools = StockTrackerTools()
get_stock_price_info = stock_tools.get_stock_price_info
add_stock_to_tracker = stock_tools.add_stock_to_tracker
remove_stock_from_tracker = stock_tools.remove_stock_from_tracker
get_stock_tracker_list = stock_tools.get_stock_tracker_list