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
    # Price Action Patterns
    HIGHER_HIGH = "HIGHER_HIGH"
    LOWER_LOW = "LOWER_LOW"
    DOUBLE_TOP = "DOUBLE_TOP"
    DOUBLE_BOTTOM = "DOUBLE_BOTTOM"
    HEAD_SHOULDERS = "HEAD_SHOULDERS"
    INVERSE_HEAD_SHOULDERS = "INVERSE_HEAD_SHOULDERS"
    ASCENDING_TRIANGLE = "ASCENDING_TRIANGLE"
    DESCENDING_TRIANGLE = "DESCENDING_TRIANGLE"
    SYMMETRICAL_TRIANGLE = "SYMMETRICAL_TRIANGLE"
    BULLISH_FLAG = "BULLISH_FLAG"
    BEARISH_FLAG = "BEARISH_FLAG"
    WEDGE_RISING = "WEDGE_RISING"
    WEDGE_FALLING = "WEDGE_FALLING"
    ENGULFING_BULLISH = "ENGULFING_BULLISH"
    ENGULFING_BEARISH = "ENGULFING_BEARISH"
    INSIDE_BAR = "INSIDE_BAR"
    OUTSIDE_BAR = "OUTSIDE_BAR"
    PINBAR_BULLISH = "PINBAR_BULLISH"
    PINBAR_BEARISH = "PINBAR_BEARISH"
    BREAKOUT_CONSOLIDATION = "BREAKOUT_CONSOLIDATION"
    FALSE_BREAKOUT = "FALSE_BREAKOUT"
    TREND_REVERSAL = "TREND_REVERSAL"

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

    def find_swing_highs_lows(self, window=5):
        """Find swing highs and lows in price data"""
        if self.data is None or len(self.data) < window * 2 + 1:
            return [], []
        
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(highs) - window):
            # Check for swing high
            is_swing_high = all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i)
            if is_swing_high:
                swing_highs.append((i, highs[i]))
            
            # Check for swing low
            is_swing_low = all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i)
            if is_swing_low:
                swing_lows.append((i, lows[i]))
        
        return swing_highs, swing_lows

    def analyze_trend_structure(self):
        """Analyze trend structure using swing highs and lows"""
        swing_highs, swing_lows = self.find_swing_highs_lows()
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "INSUFFICIENT_DATA"
        
        # Check for higher highs and higher lows (uptrend)
        recent_highs = swing_highs[-2:]
        recent_lows = swing_lows[-2:]
        
        higher_high = recent_highs[-1][1] > recent_highs[-2][1]
        higher_low = recent_lows[-1][1] > recent_lows[-2][1]
        
        lower_high = recent_highs[-1][1] < recent_highs[-2][1]
        lower_low = recent_lows[-1][1] < recent_lows[-2][1]
        
        if higher_high and higher_low:
            return "UPTREND"
        elif lower_high and lower_low:
            return "DOWNTREND"
        elif higher_high and lower_low:
            return "RANGING_BULLISH"
        elif lower_high and higher_low:
            return "RANGING_BEARISH"
        else:
            return "SIDEWAYS"

    def detect_double_top_bottom(self, tolerance=0.02):
        """Detect double top and double bottom patterns"""
        swing_highs, swing_lows = self.find_swing_highs_lows()
        
        patterns = []
        
        # Double Top
        if len(swing_highs) >= 2:
            recent_highs = swing_highs[-2:]
            price_diff = abs(recent_highs[0][1] - recent_highs[1][1]) / recent_highs[0][1]
            
            if price_diff <= tolerance:
                patterns.append("DOUBLE_TOP")
        
        # Double Bottom
        if len(swing_lows) >= 2:
            recent_lows = swing_lows[-2:]
            price_diff = abs(recent_lows[0][1] - recent_lows[1][1]) / recent_lows[0][1]
            
            if price_diff <= tolerance:
                patterns.append("DOUBLE_BOTTOM")
        
        return patterns

    def detect_head_shoulders(self):
        """Detect head and shoulders pattern"""
        swing_highs, swing_lows = self.find_swing_highs_lows()
        
        if len(swing_highs) < 3:
            return []
        
        patterns = []
        recent_highs = swing_highs[-3:]
        
        # Head and Shoulders: Left shoulder < Head > Right shoulder
        left_shoulder = recent_highs[0][1]
        head = recent_highs[1][1]
        right_shoulder = recent_highs[2][1]
        
        if (head > left_shoulder and head > right_shoulder and 
            abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
            patterns.append("HEAD_SHOULDERS")
        
        # Inverse Head and Shoulders
        if len(swing_lows) >= 3:
            recent_lows = swing_lows[-3:]
            left_shoulder = recent_lows[0][1]
            head = recent_lows[1][1]
            right_shoulder = recent_lows[2][1]
            
            if (head < left_shoulder and head < right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                patterns.append("INVERSE_HEAD_SHOULDERS")
        
        return patterns

    def detect_triangles(self, min_touches=3):
        """Detect triangle patterns"""
        swing_highs, swing_lows = self.find_swing_highs_lows()
        
        if len(swing_highs) < min_touches or len(swing_lows) < min_touches:
            return []
        
        patterns = []
        recent_highs = swing_highs[-min_touches:]
        recent_lows = swing_lows[-min_touches:]
        
        # Calculate trendlines
        high_slope = (recent_highs[-1][1] - recent_highs[0][1]) / (recent_highs[-1][0] - recent_highs[0][0])
        low_slope = (recent_lows[-1][1] - recent_lows[0][1]) / (recent_lows[-1][0] - recent_lows[0][0])
        
        # Ascending Triangle (horizontal resistance, rising support)
        if abs(high_slope) < 0.001 and low_slope > 0:
            patterns.append("ASCENDING_TRIANGLE")
        
        # Descending Triangle (falling resistance, horizontal support)
        elif high_slope < 0 and abs(low_slope) < 0.001:
            patterns.append("DESCENDING_TRIANGLE")
        
        # Symmetrical Triangle (converging trendlines)
        elif high_slope < 0 and low_slope > 0:
            patterns.append("SYMMETRICAL_TRIANGLE")
        
        return patterns

    def detect_flags_pennants(self, trend_period=20):
        """Detect flag and pennant patterns"""
        if len(self.data) < trend_period + 10:
            return []
        
        patterns = []
        
        # Get price trend before pattern
        trend_data = self.data.iloc[-(trend_period + 10):-10]
        consolidation_data = self.data.iloc[-10:]
        
        trend_change = (trend_data['Close'].iloc[-1] - trend_data['Close'].iloc[0]) / trend_data['Close'].iloc[0]
        
        # Check for consolidation after strong move
        consolidation_range = (consolidation_data['High'].max() - consolidation_data['Low'].min()) / consolidation_data['Close'].mean()
        
        if abs(trend_change) > 0.05 and consolidation_range < 0.03:  # 5% trend move, 3% consolidation
            if trend_change > 0:
                patterns.append("BULLISH_FLAG")
            else:
                patterns.append("BEARISH_FLAG")
        
        return patterns

    def detect_wedges(self, min_touches=4):
        """Detect rising and falling wedge patterns"""
        swing_highs, swing_lows = self.find_swing_highs_lows()
        
        if len(swing_highs) < min_touches or len(swing_lows) < min_touches:
            return []
        
        patterns = []
        recent_highs = swing_highs[-min_touches:]
        recent_lows = swing_lows[-min_touches:]
        
        # Calculate slopes
        high_slope = (recent_highs[-1][1] - recent_highs[0][1]) / (recent_highs[-1][0] - recent_highs[0][0])
        low_slope = (recent_lows[-1][1] - recent_lows[0][1]) / (recent_lows[-1][0] - recent_lows[0][0])
        
        # Rising Wedge (both slopes positive, converging)
        if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
            patterns.append("WEDGE_RISING")
        
        # Falling Wedge (both slopes negative, converging)
        elif high_slope < 0 and low_slope < 0 and high_slope > low_slope:
            patterns.append("WEDGE_FALLING")
        
        return patterns

    def detect_engulfing_patterns(self):
        """Detect bullish and bearish engulfing patterns"""
        if len(self.data) < 2:
            return []
        
        patterns = []
        current = self.data.iloc[-1]
        previous = self.data.iloc[-2]
        
        current_body = abs(current['Close'] - current['Open'])
        previous_body = abs(previous['Close'] - previous['Open'])
        
        # Bullish Engulfing
        if (previous['Close'] < previous['Open'] and  # Previous bearish
            current['Close'] > current['Open'] and   # Current bullish
            current['Open'] < previous['Close'] and  # Opens below previous close
            current['Close'] > previous['Open'] and  # Closes above previous open
            current_body > previous_body * 1.2):    # Significantly larger body
            patterns.append("ENGULFING_BULLISH")
        
        # Bearish Engulfing
        if (previous['Close'] > previous['Open'] and  # Previous bullish
            current['Close'] < current['Open'] and   # Current bearish
            current['Open'] > previous['Close'] and  # Opens above previous close
            current['Close'] < previous['Open'] and  # Closes below previous open
            current_body > previous_body * 1.2):    # Significantly larger body
            patterns.append("ENGULFING_BEARISH")
        
        return patterns

    def detect_inside_outside_bars(self):
        """Detect inside and outside bar patterns"""
        if len(self.data) < 2:
            return []
        
        patterns = []
        current = self.data.iloc[-1]
        previous = self.data.iloc[-2]
        
        # Inside Bar (current bar contained within previous bar)
        if (current['High'] <= previous['High'] and 
            current['Low'] >= previous['Low']):
            patterns.append("INSIDE_BAR")
        
        # Outside Bar (current bar engulfs previous bar)
        if (current['High'] > previous['High'] and 
            current['Low'] < previous['Low']):
            patterns.append("OUTSIDE_BAR")
        
        return patterns

    def detect_pin_bars(self, body_ratio=0.3, wick_ratio=2.0):
        """Detect pin bar (hammer/shooting star) patterns"""
        if len(self.data) < 1:
            return []
        
        patterns = []
        current = self.data.iloc[-1]
        
        high = current['High']
        low = current['Low']
        open_price = current['Open']
        close = current['Close']
        
        body = abs(close - open_price)
        total_range = high - low
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        
        if total_range == 0:
            return patterns
        
        # Bullish Pin Bar (long lower wick)
        if (body / total_range <= body_ratio and 
            lower_wick / body >= wick_ratio and 
            upper_wick / total_range <= 0.1):
            patterns.append("PINBAR_BULLISH")
        
        # Bearish Pin Bar (long upper wick)
        if (body / total_range <= body_ratio and 
            upper_wick / body >= wick_ratio and 
            lower_wick / total_range <= 0.1):
            patterns.append("PINBAR_BEARISH")
        
        return patterns

    def detect_breakout_patterns(self, consolidation_period=20, breakout_threshold=0.02):
        """Detect breakout from consolidation patterns"""
        if len(self.data) < consolidation_period + 5:
            return []
        
        patterns = []
        
        # Analyze consolidation period
        consolidation_data = self.data.iloc[-(consolidation_period + 5):-5]
        recent_data = self.data.iloc[-5:]
        
        # Calculate consolidation range
        consolidation_high = consolidation_data['High'].max()
        consolidation_low = consolidation_data['Low'].min()
        consolidation_range = (consolidation_high - consolidation_low) / consolidation_data['Close'].mean()
        
        current_price = self.data['Close'].iloc[-1]
        
        # Check if price was in tight consolidation
        if consolidation_range < 0.05:  # Less than 5% range
            # Check for breakout
            if current_price > consolidation_high * (1 + breakout_threshold):
                patterns.append("BREAKOUT_CONSOLIDATION")
            elif current_price < consolidation_low * (1 - breakout_threshold):
                patterns.append("BREAKDOWN_CONSOLIDATION")
        
        return patterns

    def detect_false_breakouts(self, lookback_period=10):
        """Detect false breakout patterns"""
        if len(self.data) < lookback_period + 5:
            return []
        
        patterns = []
        
        # Look for recent breakout that failed
        recent_data = self.data.iloc[-lookback_period:]
        current_price = self.data['Close'].iloc[-1]
        
        recent_high = recent_data['High'].max()
        recent_low = recent_data['Low'].min()
        
        # Check if price broke above recent high but then fell back
        if (recent_data['High'].iloc[-5:].max() > recent_high * 1.02 and  # Broke above
            current_price < recent_high * 0.98):  # Fell back below
            patterns.append("FALSE_BREAKOUT")
        
        return patterns

    def detect_trend_reversals(self, ma_period=20):
        """Detect potential trend reversal signals"""
        if len(self.data) < ma_period + 10:
            return []
        
        patterns = []
        ma = self.data['Close'].rolling(window=ma_period).mean()
        current_price = self.data['Close'].iloc[-1]
        previous_ma = ma.iloc[-2]
        current_ma = ma.iloc[-1]
        
        # Price crossing moving average
        if (self.data['Close'].iloc[-2] < previous_ma and current_price > current_ma):
            patterns.append("TREND_REVERSAL_BULLISH")
        elif (self.data['Close'].iloc[-2] > previous_ma and current_price < current_ma):
            patterns.append("TREND_REVERSAL_BEARISH")
        
        return patterns

    def analyze_price_action(self) -> List[TechnicalAlert]:
        """Comprehensive price action analysis"""
        alerts = []
        
        if self.data is None:
            return alerts
        
        current_price = self.data['Close'].iloc[-1]
        
        # Trend Structure Analysis
        trend = self.analyze_trend_structure()
        if trend == "UPTREND":
            swing_highs, swing_lows = self.find_swing_highs_lows()
            if len(swing_highs) >= 2:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.HIGHER_HIGH,
                    message=f"Higher high confirmed - uptrend intact",
                    priority=3,
                    current_price=current_price
               ))
        elif trend == "DOWNTREND":
            swing_highs, swing_lows = self.find_swing_highs_lows()
            if len(swing_lows) >= 2:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.LOWER_LOW,
                    message=f"Lower low confirmed - downtrend intact",
                    priority=3,
                    current_price=current_price
                ))
        
        # Chart Pattern Detection
        double_patterns = self.detect_double_top_bottom()
        for pattern in double_patterns:
            if pattern == "DOUBLE_TOP":
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.DOUBLE_TOP,
                    message=f"Double top pattern detected - bearish reversal signal",
                    priority=4,
                    current_price=current_price
                ))
            elif pattern == "DOUBLE_BOTTOM":
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.DOUBLE_BOTTOM,
                    message=f"Double bottom pattern detected - bullish reversal signal",
                    priority=4,
                    current_price=current_price
                ))
        
        # Head and Shoulders
        hs_patterns = self.detect_head_shoulders()
        for pattern in hs_patterns:
            if pattern == "HEAD_SHOULDERS":
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.HEAD_SHOULDERS,
                    message=f"Head and shoulders pattern - bearish reversal",
                    priority=5,
                    current_price=current_price
                ))
            elif pattern == "INVERSE_HEAD_SHOULDERS":
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.INVERSE_HEAD_SHOULDERS,
                    message=f"Inverse head and shoulders - bullish reversal",
                    priority=5,
                    current_price=current_price
                ))
        
        # Triangle Patterns
        triangle_patterns = self.detect_triangles()
        for pattern in triangle_patterns:
            alert_type = getattr(AlertType, pattern, None)
            if alert_type:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=alert_type,
                    message=f"{pattern.replace('_', ' ').title()} pattern forming",
                    priority=3,
                    current_price=current_price
                ))
        
        # Flag and Pennant Patterns
        flag_patterns = self.detect_flags_pennants()
        for pattern in flag_patterns:
            alert_type = getattr(AlertType, pattern, None)
            if alert_type:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=alert_type,
                    message=f"{pattern.replace('_', ' ').title()} pattern - continuation signal",
                    priority=4,
                    current_price=current_price
                ))
        
        # Wedge Patterns
        wedge_patterns = self.detect_wedges()
        for pattern in wedge_patterns:
            alert_type = getattr(AlertType, pattern, None)
            if alert_type:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=alert_type,
                    message=f"{pattern.replace('_', ' ').title()} pattern detected",
                    priority=4,
                    current_price=current_price
                ))
        
        # Engulfing Patterns
        engulfing_patterns = self.detect_engulfing_patterns()
        for pattern in engulfing_patterns:
            alert_type = getattr(AlertType, pattern, None)
            if alert_type:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=alert_type,
                    message=f"{pattern.replace('_', ' ').title()} pattern - strong reversal signal",
                    priority=4,
                    current_price=current_price
                ))
        
        # Inside/Outside Bars
        bar_patterns = self.detect_inside_outside_bars()
        for pattern in bar_patterns:
            alert_type = getattr(AlertType, pattern, None)
            if alert_type:
                priority = 3 if pattern == "INSIDE_BAR" else 4
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=alert_type,
                    message=f"{pattern.replace('_', ' ').title()} pattern detected",
                    priority=priority,
                    current_price=current_price
                ))
        
        # Pin Bars
        pin_patterns = self.detect_pin_bars()
        for pattern in pin_patterns:
            alert_type = getattr(AlertType, pattern, None)
            if alert_type:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=alert_type,
                    message=f"{pattern.replace('_', ' ').title()} - reversal signal",
                    priority=4,
                    current_price=current_price
                ))
        
        # Breakout Patterns
        breakout_patterns = self.detect_breakout_patterns()
        for pattern in breakout_patterns:
            if "BREAKOUT" in pattern:
                alerts.append(TechnicalAlert(
                    symbol=self.symbol,
                    alert_type=AlertType.BREAKOUT_CONSOLIDATION,
                    message=f"Breakout from consolidation detected",
                    priority=4,
                    current_price=current_price
                ))
        
        # False Breakouts
        false_breakouts = self.detect_false_breakouts()
        if false_breakouts:
            alerts.append(TechnicalAlert(
                symbol=self.symbol,
                alert_type=AlertType.FALSE_BREAKOUT,
                message=f"False breakout detected - potential reversal",
                priority=4,
                current_price=current_price
            ))
        
        # Trend Reversals
        reversal_patterns = self.detect_trend_reversals()
        if reversal_patterns:
            alerts.append(TechnicalAlert(
                symbol=self.symbol,
                alert_type=AlertType.TREND_REVERSAL,
                message=f"Trend reversal signal detected",
                priority=4,
                current_price=current_price
            ))
        
        return alerts

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

        # Price Action Analysis
        price_action_alerts = self.analyze_price_action()
        alerts.extend(price_action_alerts)

        return alerts

def analyze_stock_technical(symbol: str) -> List[TechnicalAlert]:
    """Main function to analyze a stock and return technical alerts"""
    analyzer = TechnicalAnalyzer(symbol)
    return analyzer.analyze_all()