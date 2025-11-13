"""
🧠 NeuroFlux RSI Momentum Strategy
Example strategy demonstrating neuro-flux adaptive trading.

Built with love by Nyros Veil 🚀

This strategy uses RSI with flux-adaptive parameters for momentum trading.
"""

class RSIMomentumStrategy:
    """
    RSI-based momentum strategy with neuro-flux adaptation.

    Buys on oversold conditions, sells on overbought.
    Adjusts RSI thresholds based on market flux levels.
    """

    def __init__(self, config):
        """Initialize strategy with configuration."""
        self.config = config
        self.name = "RSI Momentum"
        self.description = "Flux-adaptive RSI momentum strategy"

        # Base parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.min_confidence = 0.6

    def analyze(self, market_data):
        """
        Analyze market data and return trading signal.

        Args:
            market_data (dict): Market data including price, volume, indicators

        Returns:
            dict: Analysis results with signal, confidence, reasoning
        """
        rsi = market_data.get('rsi', 50)
        flux_level = market_data.get('flux_level', 0.0)

        # Neuro-flux adaptation: adjust thresholds based on flux
        flux_adjustment = flux_level * self.config.FLUX_SENSITIVITY
        oversold_threshold = self.rsi_oversold + (flux_adjustment * 10)
        overbought_threshold = self.rsi_overbought - (flux_adjustment * 10)

        analysis = {
            'signal': 'HOLD',
            'confidence': 0.5,
            'reasoning': '',
            'flux_adjusted': flux_adjustment > 0.1
        }

        if rsi <= oversold_threshold:
            analysis['signal'] = 'BUY'
            analysis['confidence'] = min(0.9, 0.5 + (oversold_threshold - rsi) / 20)
            analysis['reasoning'] = f"RSI {rsi:.1f} below flux-adjusted oversold threshold {oversold_threshold:.1f}"
        elif rsi >= overbought_threshold:
            analysis['signal'] = 'SELL'
            analysis['confidence'] = min(0.9, 0.5 + (rsi - overbought_threshold) / 20)
            analysis['reasoning'] = f"RSI {rsi:.1f} above flux-adjusted overbought threshold {overbought_threshold:.1f}"
        else:
            analysis['reasoning'] = f"RSI {rsi:.1f} in neutral zone ({oversold_threshold:.1f}-{overbought_threshold:.1f})"

        # Reduce confidence in high flux conditions
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