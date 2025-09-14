import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
from enum import Enum

class AlertType(Enum):
    RSI_OVERSOLD = "RSI_OVERSOLD"
    RSI_OVERBOUGHT = "RSI_OVERBOUGHT"
    MACD_BULLISH_CROSS = "MACD_BULLISH_CROSS"
    MACD_BEARISH_CROSS = "MACD_BEARISH_CROSS"
    MA_GOLDEN_CROSS = "MA_GOLDEN_CROSS"
    MA_DEATH_CROSS = "MA_DEATH_CROSS"
    BREAKOUT_RESISTANCE = "BREAKOUT_RESISTANCE"
    BREAKDOWN_SUPPORT = "BREAKDOWN_SUPPORT"
    VOLUME_SPIKE = "VOLUME_SPIKE"
    BOLLINGER_SQUEEZE = "BOLLINGER_SQUEEZE"
    BOLLINGER_BREAKOUT = "BOLLINGER_BREAKOUT"
    HAMMER_REVERSAL = "HAMMER_REVERSAL"
    DOJI_INDECISION = "DOJI_INDECISION"
    GAP_UP = "GAP_UP"
    GAP_DOWN = "GAP_DOWN"

class TechnicalAlert(BaseModel):
    symbol: str
    alert_type: AlertType
    message: str
    priority: int  # 1-5, with 5 being highest priority
    current_price: float
    indicator_value: Optional[float] = None

class TechnicalAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.data = None
        self.load_data()

    def load_data(self, period="3mo"):
        """Load stock data for analysis"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=period)
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
        except Exception as e:
            print(f"Error loading data for {self.symbol}: {e}")
            self.data = None

    def calculate_rsi(self, period=14):
        """Calculate RSI indicator"""
        if self.data is None or len(self.data) < period:
            return None
        
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        if self.data is None or len(self.data) < slow:
            return None, None, None
        
        exp1 = self.data['Close'].ewm(span=fast).mean()
        exp2 = self.data['Close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    def calculate_moving_averages(self):
        """Calculate various moving averages"""
        if self.data is None:
            return None
        
        ma_dict = {}
        for period in [20, 50, 200]:
            if len(self.data) >= period:
                ma_dict[f'MA{period}'] = self.data['Close'].rolling(window=period).mean()
        return ma_dict

    def calculate_bollinger_bands(self, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        if self.data is None or len(self.data) < period:
            return None, None, None
        
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    def detect_candlestick_patterns(self):
        """Detect basic candlestick patterns"""
        if self.data is None or len(self.data) < 2:
            return []
        
        patterns = []
        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2]
        
        # Hammer pattern
        body = abs(latest['Close'] - latest['Open'])
        upper_shadow = latest['High'] - max(latest['Close'], latest['Open'])
        lower_shadow = min(latest['Close'], latest['Open']) - latest['Low']
        
        if (lower_shadow > 2 * body and upper_shadow < body * 0.1 and 
            latest['Close'] > previous['Close']):
            patterns.append('HAMMER')
        
        # Doji pattern
        if body < (latest['High'] - latest['Low']) * 0.1:
            patterns.append('DOJI')
        
        return patterns

    def check_volume_spike(self, multiplier=2.0):
        """Check for volume spikes"""
        if self.data is None or len(self.data) < 20:
            return False, 0
        
        avg_volume = self.data['Volume'].rolling(window=20).mean().iloc[-2]
        current_volume = self.data['Volume'].iloc[-1]
        
        if current_volume > avg_volume * multiplier:
            return True, current_volume / avg_volume
        return False, current_volume / avg_volume if avg_volume > 0 else 0

    def check_price_gaps(self):
        """Check for price gaps"""
        if self.data is None or len(self.data) < 2:
            return None
        
        current = self.data.iloc[-1]
        previous = self.data.iloc[-2]
        
        gap_up = current['Open'] > previous['High'] * 1.02  # 2% gap up
        gap_down = current['Open'] < previous['Low'] * 0.98  # 2% gap down
        
        if gap_up:
            gap_percent = ((current['Open'] - previous['High']) / previous['High']) * 100
            return 'GAP_UP', gap_percent
        elif gap_down:
            gap_percent = ((previous['Low'] - current['Open']) / previous['Low']) * 100
            return 'GAP_DOWN', gap_percent
        
        return None

    def get_support_resistance_levels(self, window=20):
        """Calculate dynamic support and resistance levels"""
        if self.data is None or len(self.data) < window:
            return None, None
        
        # Simple approach: use recent highs and lows
        recent_data = self.data.tail(window)
        resistance = recent_data['High'].max()
        support = recent_data['Low'].min()
        
        return support, resistance

    def analyze_all(self) -> List[TechnicalAlert]:
        """Run comprehensive technical analysis and return alerts"""
        alerts = []
        
        if self.data is None:
            return alerts
        
        current_price = self.data['Close'].iloc[-1]
        
        # RSI Analysis
        rsi = self.calculate_rsi()
        if rsi is not None:
            current_rsi = rsi.iloc[-1]
            if current_rsi < 30:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.RSI_OVERSOLD,
                    message=f"RSI oversold at {current_rsi:.1f} - potential buy signal",
                    priority=4,
                    current_price=current_price,
                    indicator_value=current_rsi
                ))
            elif current_rsi > 70:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.RSI_OVERBOUGHT,
                    message=f"RSI overbought at {current_rsi:.1f} - potential sell signal",
                    priority=4,
                    current_price=current_price,
                    indicator_value=current_rsi
                ))

        # MACD Analysis
        macd, signal, histogram = self.calculate_macd()
        if macd is not None and len(macd) > 1:
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.MACD_BULLISH_CROSS,
                    message=f"MACD bullish crossover - momentum turning positive",
                    priority=4,
                    current_price=current_price,
                    indicator_value=macd.iloc[-1]
                ))
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.MACD_BEARISH_CROSS,
                    message=f"MACD bearish crossover - momentum turning negative",
                    priority=4,
                    current_price=current_price,
                    indicator_value=macd.iloc[-1]
                ))

        # Moving Average Analysis
        mas = self.calculate_moving_averages()
        if mas and 'MA50' in mas and 'MA200' in mas and len(mas['MA50']) > 1:
            ma50_current = mas['MA50'].iloc[-1]
            ma50_previous = mas['MA50'].iloc[-2]
            ma200_current = mas['MA200'].iloc[-1]
            ma200_previous = mas['MA200'].iloc[-2]
            
            # Golden Cross
            if ma50_current > ma200_current and ma50_previous <= ma200_previous:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.MA_GOLDEN_CROSS,
                    message=f"Golden Cross detected - 50MA crossed above 200MA",
                    priority=5,
                    current_price=current_price
                ))
            
            # Death Cross
            elif ma50_current < ma200_current and ma50_previous >= ma200_previous:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.MA_DEATH_CROSS,
                    message=f"Death Cross detected - 50MA crossed below 200MA",
                    priority=5,
                    current_price=current_price
                ))

        # Support/Resistance Breakouts
        support, resistance = self.get_support_resistance_levels()
        if support and resistance:
            if current_price > resistance * 1.005:  # 0.5% buffer
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.BREAKOUT_RESISTANCE,
                    message=f"Breakout above resistance at ${resistance:.2f}",
                    priority=4,
                    current_price=current_price,
                    indicator_value=resistance
                ))
            elif current_price < support * 0.995:  # 0.5% buffer
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.BREAKDOWN_SUPPORT,
                    message=f"Breakdown below support at ${support:.2f}",
                    priority=4,
                    current_price=current_price,
                    indicator_value=support
                ))

        # Volume Analysis
        volume_spike, volume_ratio = self.check_volume_spike()
        if volume_spike:
            alerts.append(TechnicalAlert(
                symbol=self.symbol,
                alert_type=AlertType.VOLUME_SPIKE,
                message=f"Volume spike: {volume_ratio:.1f}x average volume",
                priority=3,
                current_price=current_price,
                indicator_value=volume_ratio
            ))

        # Bollinger Bands Analysis
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands()
        if upper_bb is not None:
            bb_squeeze = (upper_bb.iloc[-1] - lower_bb.iloc[-1]) / middle_bb.iloc[-1] < 0.1
            if bb_squeeze:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.BOLLINGER_SQUEEZE,
                    message=f"Bollinger Band squeeze - volatility breakout expected",
                    priority=3,
                    current_price=current_price
                ))
            elif current_price > upper_bb.iloc[-1]:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.BOLLINGER_BREAKOUT,
                    message=f"Price broke above upper Bollinger Band",
                    priority=3,
                    current_price=current_price
                ))

        # Candlestick Patterns
        patterns = self.detect_candlestick_patterns()
        for pattern in patterns:
            if pattern == 'HAMMER':
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.HAMMER_REVERSAL,
                    message=f"Hammer reversal pattern detected",
                    priority=3,
                    current_price=current_price
                ))
            elif pattern == 'DOJI':
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.DOJI_INDECISION,
                    message=f"Doji indecision pattern - trend reversal possible",
                    priority=2,
                    current_price=current_price
                ))

        # Price Gaps
        gap_info = self.check_price_gaps()
        if gap_info:
            gap_type, gap_percent = gap_info
            if gap_type == 'GAP_UP':
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.GAP_UP,
                    message=f"Gap up {gap_percent:.1f}% - strong bullish sentiment",
                    priority=4,
                    current_price=current_price,
                    indicator_value=gap_percent
                ))
            elif gap_type == 'GAP_DOWN':
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.GAP_DOWN,
                    message=f"Gap down {gap_percent:.1f}% - strong bearish sentiment",
                    priority=4,
                    current_price=current_price,
                    indicator_value=gap_percent
                ))

        return alerts

def analyze_stock_technical(symbol: str) -> List[TechnicalAlert]:
    """Main function to analyze a stock and return technical alerts"""
    analyzer = TechnicalAnalyzer(symbol)
    return analyzer.analyze_all()