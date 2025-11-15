"""
🧠 NeuroFlux Advanced Market Analysis Agent
Sophisticated market analysis with technical indicators, sentiment correlation, and predictive analytics.

Built with love by Nyros Veil 🚀

Features:
- Multi-timeframe technical analysis
- Cross-asset correlation analysis
- Market regime detection
- Risk-adjusted return calculations
- Advanced charting and visualization
- Predictive market signals
"""

import os
import sys
import time
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from termcolor import cprint
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.base_agent import BaseAgent, AgentStatus
from models.model_factory import ModelFactory

# Load environment variables
load_dotenv()

@dataclass
class MarketRegime:
    """Market regime classification."""
    name: str
    volatility: float
    trend_strength: float
    liquidity: float
    risk_level: str  # 'low', 'medium', 'high'

@dataclass
class TechnicalSignals:
    """Technical analysis signals."""
    rsi: float
    macd_signal: str  # 'bullish', 'bearish', 'neutral'
    moving_avg_trend: str  # 'uptrend', 'downtrend', 'sideways'
    support_resistance: Dict[str, float]
    volume_analysis: str

class AdvancedMarketAnalysisAgent(BaseAgent):
    """
    Advanced market analysis agent with sophisticated technical and fundamental analysis.
    """

    def __init__(self, agent_id: str = None, name: str = "Advanced Market Analysis Agent"):
        super().__init__(agent_id, name)
        self.analysis_history = []
        self.market_regime = None
        self.technical_indicators = {}
        self.correlation_matrix = {}
        self.risk_metrics = {}

        # Analysis parameters
        self.lookback_periods = [20, 50, 200]  # SMA periods
        self.rsi_period = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9

    async def initialize(self) -> bool:
        """Initialize the advanced market analysis agent."""
        try:
            cprint(f"🧠 Initializing {self.name}...", "cyan")

            # Initialize model for advanced analysis
            self.model = ModelFactory.create_model("claude-3-haiku-20240307")

            # Set up analysis directories
            self.output_dir = os.path.join("src", "data", "market_analysis_agent")
            os.makedirs(self.output_dir, exist_ok=True)

            cprint(f"✅ {self.name} initialized successfully", "green")
            return True

        except Exception as e:
            cprint(f"❌ Failed to initialize {self.name}: {e}", "red")
            return False

    async def analyze_market_conditions(self, symbols: List[str]) -> Dict[str, Any]:
        """Perform comprehensive market analysis."""
        try:
            cprint(f"📊 Starting advanced market analysis for {len(symbols)} symbols", "blue")

            analysis_results = {}

            for symbol in symbols:
                # Get market data
                market_data = await self._fetch_market_data(symbol)

                # Perform technical analysis
                technical_analysis = self._calculate_technical_indicators(market_data)

                # Detect market regime
                regime = self._detect_market_regime(market_data, technical_analysis)

                # Calculate risk metrics
                risk_metrics = self._calculate_risk_metrics(market_data)

                # Generate trading signals
                signals = self._generate_trading_signals(technical_analysis, regime, risk_metrics)

                analysis_results[symbol] = {
                    'timestamp': datetime.now().isoformat(),
                    'technical_analysis': technical_analysis,
                    'market_regime': regime.__dict__ if regime else None,
                    'risk_metrics': risk_metrics,
                    'trading_signals': signals,
                    'confidence_score': self._calculate_confidence_score(signals)
                }

            # Perform cross-asset correlation analysis
            if len(symbols) > 1:
                correlation_analysis = self._analyze_correlations(symbols)
                analysis_results['cross_asset_analysis'] = correlation_analysis

            # Save analysis results
            self._save_analysis_results(analysis_results)

            cprint(f"✅ Market analysis completed for {len(symbols)} symbols", "green")
            return analysis_results

        except Exception as e:
            cprint(f"❌ Market analysis failed: {e}", "red")
            return {}

    async def _fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch historical market data for analysis."""
        try:
            # This would integrate with exchange APIs
            # For now, return mock data structure
            dates = pd.date_range(end=datetime.now(), periods=200, freq='1H')
            np.random.seed(42)  # For reproducible results

            data = {
                'timestamp': dates,
                'open': 100 + np.random.randn(200).cumsum(),
                'high': 105 + np.random.randn(200).cumsum(),
                'low': 95 + np.random.randn(200).cumsum(),
                'close': 100 + np.random.randn(200).cumsum(),
                'volume': np.random.randint(1000, 10000, 200)
            }

            df = pd.DataFrame(data)
            df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4  # Realistic close
            return df

        except Exception as e:
            cprint(f"❌ Failed to fetch market data for {symbol}: {e}", "red")
            return pd.DataFrame()

    def _calculate_technical_indicators(self, data: pd.DataFrame) -> TechnicalSignals:
        """Calculate comprehensive technical indicators."""
        try:
            if data.empty:
                return TechnicalSignals(0, 'neutral', 'sideways', {}, 'insufficient_data')

            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            volume = data['volume'].values

            # RSI Calculation
            rsi = self._calculate_rsi(close_prices, self.rsi_period)

            # MACD Calculation
            macd_line, signal_line, histogram = self._calculate_macd(close_prices)
            macd_signal = self._interpret_macd(macd_line[-1], signal_line[-1], histogram[-1])

            # Moving Average Trend
            ma_trend = self._calculate_ma_trend(close_prices)

            # Support/Resistance Levels
            support_resistance = self._find_support_resistance(high_prices, low_prices, close_prices)

            # Volume Analysis
            volume_analysis = self._analyze_volume(volume, close_prices)

            return TechnicalSignals(
                rsi=rsi[-1] if len(rsi) > 0 else 50,
                macd_signal=macd_signal,
                moving_avg_trend=ma_trend,
                support_resistance=support_resistance,
                volume_analysis=volume_analysis
            )

        except Exception as e:
            cprint(f"❌ Technical analysis failed: {e}", "red")
            return TechnicalSignals(50, 'neutral', 'sideways', {}, 'error')

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return np.array([50.0])  # Neutral RSI

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')

        rs = avg_gains / np.where(avg_losses == 0, 0.001, avg_losses)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD indicator."""
        if len(prices) < self.macd_slow:
            return np.array([0]), np.array([0]), np.array([0])

        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, self.macd_fast)
        ema_slow = self._calculate_ema(prices, self.macd_slow)

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = self._calculate_ema(macd_line, self.macd_signal)

        # Histogram
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return np.full(len(data), np.mean(data))

        multiplier = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[period-1] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1 - multiplier))

        return ema

    def _interpret_macd(self, macd: float, signal: float, histogram: float) -> str:
        """Interpret MACD signals."""
        if macd > signal and histogram > 0:
            return 'bullish'
        elif macd < signal and histogram < 0:
            return 'bearish'
        else:
            return 'neutral'

    def _calculate_ma_trend(self, prices: np.ndarray) -> str:
        """Determine moving average trend."""
        if len(prices) < 200:
            return 'insufficient_data'

        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        sma_200 = np.mean(prices[-200:])

        # Trend analysis
        if sma_20 > sma_50 > sma_200:
            return 'strong_uptrend'
        elif sma_20 > sma_50 and sma_50 > sma_200:
            return 'uptrend'
        elif sma_20 < sma_50 < sma_200:
            return 'strong_downtrend'
        elif sma_20 < sma_50 and sma_50 < sma_200:
            return 'downtrend'
        else:
            return 'sideways'

    def _find_support_resistance(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, float]:
        """Find support and resistance levels."""
        try:
            # Simple pivot point analysis
            recent_high = np.max(highs[-20:])
            recent_low = np.min(lows[-20:])
            current_price = closes[-1]

            return {
                'resistance': recent_high,
                'support': recent_low,
                'current_price': current_price,
                'pivot_point': (recent_high + recent_low + current_price) / 3
            }
        except:
            return {'resistance': 0, 'support': 0, 'current_price': 0, 'pivot_point': 0}

    def _analyze_volume(self, volume: np.ndarray, prices: np.ndarray) -> str:
        """Analyze volume patterns."""
        try:
            if len(volume) < 20:
                return 'insufficient_data'

            avg_volume = np.mean(volume[-20:])
            recent_volume = np.mean(volume[-5:])

            price_change = (prices[-1] - prices[-10]) / prices[-10]

            if recent_volume > avg_volume * 1.5 and price_change > 0.02:
                return 'high_volume_breakout'
            elif recent_volume > avg_volume * 1.2:
                return 'above_average_volume'
            elif recent_volume < avg_volume * 0.8:
                return 'low_volume'
            else:
                return 'normal_volume'
        except:
            return 'analysis_error'

    def _detect_market_regime(self, data: pd.DataFrame, technical: TechnicalSignals) -> Optional[MarketRegime]:
        """Detect current market regime."""
        try:
            if data.empty:
                return None

            # Calculate volatility (standard deviation of returns)
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Trend strength based on moving averages
            trend_strength = 0.5  # Neutral default
            if technical.moving_avg_trend in ['strong_uptrend', 'strong_downtrend']:
                trend_strength = 0.8
            elif technical.moving_avg_trend in ['uptrend', 'downtrend']:
                trend_strength = 0.6

            # Liquidity proxy (volume analysis)
            liquidity = 0.5
            if technical.volume_analysis == 'high_volume_breakout':
                liquidity = 0.8
            elif technical.volume_analysis == 'low_volume':
                liquidity = 0.3

            # Risk level determination
            if volatility > 0.8 or trend_strength < 0.4:
                risk_level = 'high'
            elif volatility > 0.5 or trend_strength > 0.7:
                risk_level = 'medium'
            else:
                risk_level = 'low'

            # Regime naming
            if volatility < 0.3 and trend_strength > 0.7:
                regime_name = 'strong_trend'
            elif volatility > 0.7:
                regime_name = 'high_volatility'
            elif liquidity < 0.4:
                regime_name = 'low_liquidity'
            else:
                regime_name = 'normal'

            return MarketRegime(
                name=regime_name,
                volatility=float(volatility),
                trend_strength=trend_strength,
                liquidity=liquidity,
                risk_level=risk_level
            )

        except Exception as e:
            cprint(f"❌ Market regime detection failed: {e}", "red")
            return None

    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics."""
        try:
            if data.empty:
                return {}

            returns = data['close'].pct_change().dropna()

            # Sharpe ratio (assuming 0% risk-free rate)
            if len(returns) > 1:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Value at Risk (95% confidence)
            if len(returns) > 1:
                var_95 = np.percentile(returns, 5)
            else:
                var_95 = 0

            # Volatility
            volatility = returns.std() * np.sqrt(252)

            return {
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'value_at_risk_95': float(var_95),
                'volatility': float(volatility),
                'total_return': float((data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100)
            }

        except Exception as e:
            cprint(f"❌ Risk metrics calculation failed: {e}", "red")
            return {}

    def _generate_trading_signals(self, technical: TechnicalSignals, regime: Optional[MarketRegime],
                                risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive trading signals."""
        try:
            signals = {
                'overall_signal': 'hold',
                'confidence': 0.5,
                'entry_price': None,
                'stop_loss': None,
                'take_profit': None,
                'risk_reward_ratio': 1.0,
                'timeframe': '1h'
            }

            # Technical signal scoring
            score = 0

            # RSI signals
            if technical.rsi < 30:
                score += 0.3  # Oversold
            elif technical.rsi > 70:
                score -= 0.3  # Overbought

            # MACD signals
            if technical.macd_signal == 'bullish':
                score += 0.25
            elif technical.macd_signal == 'bearish':
                score -= 0.25

            # Trend signals
            if technical.moving_avg_trend in ['strong_uptrend', 'uptrend']:
                score += 0.2
            elif technical.moving_avg_trend in ['strong_downtrend', 'downtrend']:
                score -= 0.2

            # Volume confirmation
            if technical.volume_analysis == 'high_volume_breakout':
                score += 0.15

            # Adjust for market regime
            if regime:
                if regime.risk_level == 'high':
                    score *= 0.7  # Reduce confidence in high-risk regimes
                elif regime.trend_strength > 0.7:
                    score *= 1.2  # Increase confidence in strong trends

            # Adjust for risk metrics
            if risk_metrics.get('sharpe_ratio', 0) > 1.0:
                score += 0.1
            if risk_metrics.get('max_drawdown', 0) > -0.2:  # Too much drawdown
                score -= 0.2

            # Determine overall signal
            if score > 0.4:
                signals['overall_signal'] = 'buy'
            elif score < -0.4:
                signals['overall_signal'] = 'sell'
            else:
                signals['overall_signal'] = 'hold'

            signals['confidence'] = min(abs(score), 1.0)

            # Calculate entry/exit levels if signal is strong enough
            if signals['confidence'] > 0.6 and technical.support_resistance:
                if signals['overall_signal'] == 'buy':
                    signals['entry_price'] = technical.support_resistance.get('support', 0)
                    signals['stop_loss'] = signals['entry_price'] * 0.95
                    signals['take_profit'] = signals['entry_price'] * 1.05
                    signals['risk_reward_ratio'] = 1.05 / 0.95 - 1
                elif signals['overall_signal'] == 'sell':
                    signals['entry_price'] = technical.support_resistance.get('resistance', 0)
                    signals['stop_loss'] = signals['entry_price'] * 1.05
                    signals['take_profit'] = signals['entry_price'] * 0.95
                    signals['risk_reward_ratio'] = 1.05 / 0.95 - 1

            return signals

        except Exception as e:
            cprint(f"❌ Signal generation failed: {e}", "red")
            return {'overall_signal': 'hold', 'confidence': 0.0, 'error': str(e)}

    def _calculate_confidence_score(self, signals: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the analysis."""
        try:
            base_confidence = signals.get('confidence', 0.5)

            # Adjust based on signal strength and risk/reward
            rr_ratio = signals.get('risk_reward_ratio', 1.0)
            if rr_ratio > 2.0:
                base_confidence *= 1.2
            elif rr_ratio < 1.0:
                base_confidence *= 0.8

            return min(base_confidence, 1.0)

        except:
            return 0.5

    def _analyze_correlations(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze correlations between different assets."""
        try:
            correlation_data = {}

            # This would analyze correlations between symbols
            # For now, return placeholder structure
            correlation_data['correlation_matrix'] = {}
            correlation_data['highly_correlated_pairs'] = []
            correlation_data['market_regime_impact'] = 'neutral'

            return correlation_data

        except Exception as e:
            cprint(f"❌ Correlation analysis failed: {e}", "red")
            return {}

    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_analysis_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            cprint(f"💾 Analysis results saved to {filepath}", "green")

        except Exception as e:
            cprint(f"❌ Failed to save analysis results: {e}", "red")

    async def get_market_insights(self, symbol: str) -> Dict[str, Any]:
        """Get AI-powered market insights for a symbol."""
        try:
            # Use LLM to provide additional market insights
            prompt = f"""
            Analyze the current market conditions for {symbol} and provide insights on:
            1. Short-term price direction (next 1-3 days)
            2. Key support/resistance levels
            3. Market sentiment factors
            4. Risk factors to consider
            5. Trading strategy recommendations

            Provide a concise but comprehensive analysis.
            """

            if self.model:
                insights = await self.model.generate_response(prompt, max_tokens=500)
                return {
                    'symbol': symbol,
                    'insights': insights,
                    'generated_at': datetime.now().isoformat()
                }
            else:
                return {
                    'symbol': symbol,
                    'insights': 'AI model not available for insights',
                    'generated_at': datetime.now().isoformat()
                }

        except Exception as e:
            cprint(f"❌ Failed to get market insights: {e}", "red")
            return {'error': str(e)}

    async def run_cycle(self) -> Dict[str, Any]:
        """Run a complete market analysis cycle."""
        try:
            # Default symbols to analyze
            symbols = ['BTC', 'ETH', 'SOL']

            # Perform comprehensive analysis
            analysis_results = await self.analyze_market_conditions(symbols)

            # Generate AI insights for top symbol
            if symbols:
                insights = await self.get_market_insights(symbols[0])
                analysis_results['ai_insights'] = insights

            return analysis_results

        except Exception as e:
            cprint(f"❌ Analysis cycle failed: {e}", "red")
            return {'error': str(e)}

    async def shutdown(self) -> None:
        """Shutdown the market analysis agent."""
        cprint(f"🛑 Shutting down {self.name}", "yellow")
        # Cleanup resources if needed
        cprint(f"✅ {self.name} shutdown complete", "green")