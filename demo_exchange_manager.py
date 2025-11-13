"""
🧠 NeuroFlux Exchange Manager Demo
Demonstration of the unified multi-exchange trading infrastructure.

Built with love by Nyros Veil 🚀

This script demonstrates the Exchange Manager capabilities:
- Multi-exchange connectivity
- Unified trading operations
- Portfolio management
- Real-time data streaming
"""

import asyncio
import time
import sys
import os
from typing import Dict, Any
from termcolor import colored, cprint

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import NeuroFlux components
from exchanges import ExchangeManager, Order, OrderSide, OrderType


async def demo_exchange_manager():
    """Demonstrate Exchange Manager functionality"""
    cprint("🚀 NeuroFlux Exchange Manager Demo", "cyan", attrs=["bold"])
    cprint("=" * 50, "cyan")

    # Initialize Exchange Manager with demo configuration
    config = {
        'flux_sensitivity': 0.8,
        # Note: In real usage, these would be loaded from environment variables
        # 'hyperliquid_api_key': os.getenv('HYPERLIQUID_API_KEY'),
        # 'hyperliquid_secret_key': os.getenv('HYPERLIQUID_SECRET_KEY'),
        # 'hyperliquid_wallet_address': os.getenv('HYPERLIQUID_WALLET_ADDRESS'),
    }

    exchange_manager = ExchangeManager(config)

    # Start the exchange manager
    cprint("\n📡 Starting Exchange Manager...", "yellow")
    started = await exchange_manager.start()

    if not started:
        cprint("❌ Failed to start Exchange Manager", "red")
        return

    cprint("✅ Exchange Manager started successfully", "green")

    # Display exchange status
    cprint("\n📊 Exchange Status:", "blue")
    status = exchange_manager.get_exchange_status()
    for name, stat in status.items():
        connected = "🟢 Connected" if stat.connected else "🔴 Disconnected"
        cprint(f"  {name}: {connected} | Positions: {stat.open_positions} | Orders: {stat.open_orders}", "white")

    # Get portfolio balance
    cprint("\n💰 Portfolio Balance:", "blue")
    try:
        balance = await exchange_manager.get_portfolio_balance()
        if balance:
            for asset, amount in balance.items():
                cprint(f"  {asset}: {amount:.4f}", "white")
        else:
            cprint("  No balance data available (likely no API credentials)", "yellow")
    except Exception as e:
        cprint(f"  Error getting balance: {str(e)}", "red")

    # Get market data
    cprint("\n📈 Market Data (BTC):", "blue")
    try:
        market_data = await exchange_manager.get_market_data("BTC")
        if market_data:
            cprint(f"  Price: ${market_data.price:.2f}", "white")
            cprint(f"  Bid/Ask: ${market_data.bid:.2f} / ${market_data.ask:.2f}", "white")
            cprint(f"  24h Volume: {market_data.volume_24h:.2f}", "white")
            cprint(f"  24h Change: {market_data.change_24h:.2f}%", "white")
        else:
            cprint("  No market data available", "yellow")
    except Exception as e:
        cprint(f"  Error getting market data: {str(e)}", "red")

    # Demonstrate order creation (simulation only)
    cprint("\n📝 Order Creation Demo:", "blue")
    demo_order = Order(
        id="demo_order_001",
        symbol="BTC",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        amount=0.001,  # Small amount for demo
        price=50000.0,  # Demo price
        status="pending"
    )

    cprint(f"  Demo Order: {demo_order.side.value.upper()} {demo_order.amount} {demo_order.symbol}", "white")
    cprint(f"    Type: {demo_order.type.value} | Price: ${demo_order.price}", "white")

    # Try to place order (will fail without real credentials)
    try:
        order_id = await exchange_manager.place_order(demo_order)
        if order_id:
            cprint(f"  ✅ Order placed successfully: {order_id}", "green")
        else:
            cprint("  ❌ Order placement failed (expected without real credentials)", "yellow")
    except Exception as e:
        cprint(f"  ❌ Order placement error: {str(e)}", "red")

    # Get positions
    cprint("\n📊 Current Positions:", "blue")
    try:
        positions = await exchange_manager.get_all_positions()
        if positions:
            for pos in positions:
                pnl_color = "green" if pos.pnl >= 0 else "red"
                cprint(f"  {pos.symbol} {pos.side.value.upper()}: {pos.amount:.4f} @ ${pos.entry_price:.2f}", "white")
                cprint(f"    Current: ${pos.current_price:.2f} | P&L: {pos.pnl:.2f} ({pos.pnl_percentage:.2f}%)", pnl_color)
        else:
            cprint("  No open positions", "yellow")
    except Exception as e:
        cprint(f"  Error getting positions: {str(e)}", "red")

    # Demonstrate flux level adjustment
    cprint("\n🧠 Neuro-Flux Adaptation:", "blue")
    original_flux = exchange_manager.flux_level
    cprint(f"  Current flux level: {original_flux:.2f}", "white")

    # Simulate market volatility increase
    exchange_manager.update_flux_level(0.7)
    cprint(f"  Updated flux level (high volatility): {exchange_manager.flux_level:.2f}", "yellow")

    # Reset to normal
    exchange_manager.update_flux_level(0.3)
    cprint(f"  Updated flux level (low volatility): {exchange_manager.flux_level:.2f}", "green")

    # Stop the exchange manager
    cprint("\n🛑 Stopping Exchange Manager...", "yellow")
    await exchange_manager.stop()
    cprint("✅ Exchange Manager stopped", "green")

    cprint("\n🎯 Demo completed successfully!", "cyan", attrs=["bold"])


async def demo_real_time_data():
    """Demonstrate real-time data streaming capabilities"""
    cprint("\n📡 Real-Time Data Streaming Demo", "cyan", attrs=["bold"])
    cprint("=" * 40, "cyan")

    exchange_manager = ExchangeManager()

    # Define a callback for market data
    async def market_data_callback(data):
        cprint(f"📊 {data.symbol}: ${data.price:.2f} (Bid: ${data.bid:.2f}, Ask: ${data.ask:.2f})", "green")

    # Subscribe to real-time data (simulation)
    symbols = ["BTC", "ETH"]
    cprint(f"📡 Subscribing to real-time data for: {', '.join(symbols)}", "yellow")

    success = await exchange_manager.subscribe_market_data(symbols, market_data_callback)
    if success:
        cprint("✅ Real-time data subscription successful", "green")

        # Simulate some market data updates
        cprint("📡 Simulating market data updates...", "blue")
        for i in range(3):
            # In real implementation, this would come from WebSocket streams
            cprint(f"  Update {i+1}: Market data would stream here", "white")
            await asyncio.sleep(1)

        cprint("✅ Real-time data demo completed", "green")
    else:
        cprint("⚠️  Real-time data subscription not available (expected for demo)", "yellow")


async def main():
    """Main demo function"""
    cprint("🧠 Welcome to NeuroFlux Exchange Manager Demo", "cyan", attrs=["bold"])
    cprint("Built with love by Nyros Veil 🚀\n", "white")

    try:
        # Run main demo
        await demo_exchange_manager()

        # Run real-time data demo
        await demo_real_time_data()

    except KeyboardInterrupt:
        cprint("\n⏹️  Demo interrupted by user", "yellow")
    except Exception as e:
        cprint(f"\n❌ Demo failed with error: {str(e)}", "red")
        import traceback
        traceback.print_exc()
    finally:
        cprint("\n👋 Thank you for trying NeuroFlux!", "cyan")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())