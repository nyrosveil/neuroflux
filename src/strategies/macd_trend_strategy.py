"""
🧠 NeuroFlux MACD Trend Strategy
Example strategy demonstrating neuro-flux adaptive trend following.

Built with love by Nyros Veil 🚀

This strategy uses MACD with flux-adaptive parameters for trend trading.
"""

class MACDTrendStrategy:
    """
    MACD-based trend strategy with neuro-flux adaptation.

    Buys on bullish crossovers, sells on bearish crossovers.
    Adjusts sensitivity based on market flux levels.
    """

    def __init__(self, config):
        """Initialize strategy with configuration."""
        self.config = config
        self.name = "MACD Trend"
        self.description = "Flux-adaptive MACD trend-following strategy"

        # Base parameters
        self.macd_signal_threshold = 0.001
        self.min_confidence = 0.6

    def analyze(self, market_data):
        """
        Analyze market data and return trading signal.

        Args:
            market_data (dict): Market data including price, volume, indicators

        Returns:
            dict: Analysis results with signal, confidence, reasoning
        """
        macd = market_data.get('macd', 0.0)
        macd_signal = market_data.get('macd_signal', 0.0)
        macd_hist = market_data.get('macd_hist', 0.0)
        flux_level = market_data.get('flux_level', 0.0)

        # Neuro-flux adaptation: adjust threshold based on flux
        flux_adjustment = flux_level * self.config.FLUX_SENSITIVITY
        signal_threshold = self.macd_signal_threshold * (1 + flux_adjustment)

        analysis = {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reasoning': '',
            'flux_adjusted': flux_adjustment > 0.1
        }

        macd_diff = macd - macd_signal

        if macd_diff > signal_threshold and macd_hist > 0:
            analysis['signal'] = 'BUY'
            analysis['confidence'] = min(0.9, 0.5 + abs(macd_diff) / signal_threshold)
            analysis['reasoning'] = f"MACD bullish crossover: {macd:.4f} > {macd_signal:.4f} by {macd_diff:.4f}"
        elif macd_diff < -signal_threshold and macd_hist < 0:
            analysis['signal'] = 'SELL'
            analysis['confidence'] = min(0.9, 0.5 + abs(macd_diff) / signal_threshold)
            analysis['reasoning'] = f"MACD bearish crossover: {macd:.4f} < {macd_signal:.4f} by {abs(macd_diff):.4f}"
        else:
            analysis['reasoning'] = f"MACD neutral: diff {macd_diff:.4f}, threshold {signal_threshold:.4f}"

        # Reduce confidence in high flux
        if flux_level > self.config.FLUX_SENSITIVITY:
            analysis['confidence'] *= (1 - flux_level)
            analysis['reasoning'] += f" | Confidence reduced by {flux_level:.1f} flux level"

        return analysis

    def should_buy(self, market_data):
        """Check if strategy signals a buy."""
        analysis = self.analyze(market_data)
        return analysis['signal'] == 'BUY' and analysis['confidence'] >= self.min_confidence

    def should_sell(self, market_data):
        """Check if strategy signals a sell."""
        analysis = self.analyze(market_data)
        return analysis['signal'] == 'SELL' and analysis['confidence'] >= self.min_confidence

    def get_position_size(self, market_data, available_balance):
        """
        Calculate position size based on strategy and risk management.

        Args:
            market_data (dict): Current market data
            available_balance (float): Available balance in USD

        Returns:
            float: Position size in USD
        """
        base_size = min(self.config.usd_size, self.config.max_usd_order_size)

        # Adjust size based on confidence and flux
        analysis = self.analyze(market_data)
        confidence_multiplier = analysis['confidence']

        # Reduce size in high flux
        flux_multiplier = 1.0
        if market_data.get('flux_level', 0) > self.config.FLUX_SENSITIVITY:
            flux_multiplier = 0.5

        position_size = base_size * confidence_multiplier * flux_multiplier
        return min(position_size, available_balance * 0.1)  # Max 10% of balance