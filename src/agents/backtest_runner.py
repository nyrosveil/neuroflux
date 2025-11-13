"""
🧠 NeuroFlux Backtest Runner Agent
Executes backtests programmatically with neuro-optimization.

Built with love by Nyros Veil 🚀

Runs backtests using the enhanced neuro-flux framework.
Integrates with RBI agent for automated strategy testing.
Provides performance analytics and optimization reports.
"""

import os
import time
import json
import subprocess
import sys
from datetime import datetime
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import enhanced backtesting framework
from backtesting.neuro_backtest import run_neuro_backtest, load_rbi_strategy

# Output directory for backtest results
OUTPUT_DIR = "src/data/backtest_runner/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_latest_rbi_result():
    """
    Find the most recent RBI agent result.
    """
    try:
        rbi_data_dir = "src/data/rbi_agent/"
        if not os.path.exists(rbi_data_dir):
            return None

        # Find all RBI result directories
        rbi_dirs = [d for d in os.listdir(rbi_data_dir)
                   if d.startswith("rbi_result_") and os.path.isdir(os.path.join(rbi_data_dir, d))]

        if not rbi_dirs:
            return None

        # Get the latest one
        latest_dir = max(rbi_dirs)
        return os.path.join(rbi_data_dir, latest_dir)

    except Exception as e:
        cprint(f"❌ Error finding RBI results: {str(e)}", "red")
        return None

def execute_rbi_backtest(rbi_result_path):
    """
    Execute backtest from RBI-generated code.
    """
    try:
        backtest_code_path = os.path.join(rbi_result_path, "backtest_code.py")

        if not os.path.exists(backtest_code_path):
            cprint(f"❌ RBI backtest code not found: {backtest_code_path}", "red")
            return None

        cprint(f"🚀 Executing RBI backtest: {backtest_code_path}", "blue")

        # Execute the backtest code
        result = subprocess.run(
            [sys.executable, backtest_code_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(backtest_code_path)
        )

        if result.returncode == 0:
            cprint("✅ RBI backtest executed successfully", "green")

            # Try to load results if they exist
            results_file = os.path.join(rbi_result_path, "backtest_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    backtest_results = json.load(f)
                return backtest_results
            else:
                return {"status": "success", "output": result.stdout}
        else:
            cprint(f"❌ RBI backtest failed: {result.stderr}", "red")
            return {"status": "failed", "error": result.stderr}

    except Exception as e:
        cprint(f"❌ Error executing RBI backtest: {str(e)}", "red")
        return None

def run_enhanced_backtest(strategy_name="NeuroFluxStrategy", data_path=None):
    """
    Run enhanced neuro-flux backtest.
    """
    try:
        if data_path is None:
            # Try to find sample data
            sample_data_paths = [
                "src/data/rbi/BTC-USD-1H.csv",
                "src/data/sample_data.csv"
            ]

            for path in sample_data_paths:
                if os.path.exists(path):
                    data_path = path
                    break

            if data_path is None:
                cprint("❌ No data file found for backtesting", "red")
                return None

        cprint(f"🧠 Running enhanced backtest with data: {data_path}", "cyan")

        # Import strategy dynamically (placeholder for now)
        # In full implementation, this would load different strategies
        from backtesting.neuro_backtest import NeuroFluxStrategy

        # Run neuro-flux enhanced backtest
        results = run_neuro_backtest(NeuroFluxStrategy, data_path, rbi_integration=True)

        if results:
            cprint("✅ Enhanced backtest completed successfully", "green")
            return results
        else:
            cprint("❌ Enhanced backtest failed", "red")
            return None

    except Exception as e:
        cprint(f"❌ Error in enhanced backtest: {str(e)}", "red")
        return None

def analyze_backtest_results(results):
    """
    Analyze backtest results and generate insights.
    """
    if not results:
        return None

    analysis = {
        'timestamp': datetime.now().isoformat(),
        'performance_summary': {},
        'risk_metrics': {},
        'optimization_insights': {},
        'recommendations': []
    }

    try:
        # Extract key metrics
        if 'backtest_stats' in results:
            stats = results['backtest_stats']
            analysis['performance_summary'] = {
                'total_return': stats.get('Return [%]', 0),
                'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                'max_drawdown': stats.get('Max. Drawdown [%]', 0),
                'win_rate': stats.get('Win Rate [%]', 0),
                'profit_factor': stats.get('Profit Factor', 0)
            }

        # Flux-adjusted metrics
        if 'flux_adjusted_metrics' in results:
            flux_metrics = results['flux_adjusted_metrics']
            analysis['risk_metrics'] = {
                'flux_adjusted_sharpe': flux_metrics.get('sharpe_ratio', 0),
                'flux_adjusted_return': flux_metrics.get('return_pct', 0),
                'flux_penalty': flux_metrics.get('flux_penalty', 0)
            }

        # Optimization insights
        if 'optimization' in results:
            opt = results['optimization']
            analysis['optimization_insights'] = {
                'best_score': opt.get('best_score', 0),
                'iterations': len(opt.get('history', [])),
                'parameters': results.get('parameters', {})
            }

        # Generate recommendations
        total_return = analysis['performance_summary'].get('total_return', 0)
        sharpe = analysis['performance_summary'].get('sharpe_ratio', 0)
        max_dd = analysis['performance_summary'].get('max_drawdown', 0)

        if total_return > 50 and sharpe > 1.5 and max_dd < 20:
            analysis['recommendations'].append("Excellent strategy performance - consider live deployment")
        elif total_return > 20 and sharpe > 1.0:
            analysis['recommendations'].append("Good performance - monitor closely for live testing")
        elif max_dd > 30:
            analysis['recommendations'].append("High drawdown detected - consider risk management improvements")
        else:
            analysis['recommendations'].append("Strategy needs optimization - review parameters")

        return analysis

    except Exception as e:
        cprint(f"❌ Error analyzing results: {str(e)}", "red")
        return None

def save_backtest_report(results, analysis):
    """
    Save comprehensive backtest report.
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'backtest_results': results,
        'analysis': analysis,
        'metadata': {
            'framework_version': 'NeuroFlux v1.0',
            'optimization_enabled': True,
            'flux_adaptation': True
        }
    }

    # Save latest report
    with open(f"{OUTPUT_DIR}/latest_backtest_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Append to history
    with open(f"{OUTPUT_DIR}/backtest_reports.jsonl", 'a') as f:
        f.write(json.dumps(report, default=str) + '\n')

def main():
    """Main backtest runner loop."""
    cprint("🧠 NeuroFlux Backtest Runner Agent starting...", "cyan")
    cprint("Automated backtesting with neuro-optimization", "yellow")

    while True:
        try:
            cprint("🔍 Checking for new RBI results...", "blue")

            # Find latest RBI result
            rbi_result_path = find_latest_rbi_result()

            if rbi_result_path:
                cprint(f"📁 Found RBI result: {os.path.basename(rbi_result_path)}", "green")

                # Execute RBI backtest
                rbi_results = execute_rbi_backtest(rbi_result_path)

                if rbi_results:
                    cprint("📊 RBI backtest completed", "green")

                    # Run enhanced neuro-flux backtest
                    enhanced_results = run_enhanced_backtest()

                    if enhanced_results:
                        # Analyze results
                        analysis = analyze_backtest_results(enhanced_results)

                        if analysis:
                            # Save comprehensive report
                            save_backtest_report(enhanced_results, analysis)

                            cprint("✅ Backtest analysis complete", "green")
                            cprint(f"💰 Return: {analysis['performance_summary'].get('total_return', 0):.2f}%", "yellow")
                            cprint(f"📈 Sharpe: {analysis['performance_summary'].get('sharpe_ratio', 0):.2f}", "yellow")
                        else:
                            cprint("⚠️  Results analysis failed", "yellow")
                    else:
                        cprint("⚠️  Enhanced backtest failed", "yellow")
                else:
                    cprint("⚠️  RBI backtest execution failed", "yellow")
            else:
                cprint("😴 No new RBI results found", "blue")

            cprint(f"⏳ Sleeping {config.SLEEP_BETWEEN_RUNS_MINUTES} minutes before next check", "white")
            time.sleep(config.SLEEP_BETWEEN_RUNS_MINUTES * 60)

        except KeyboardInterrupt:
            cprint("\n🛑 Backtest Runner Agent stopped by user", "yellow")
            break
        except Exception as e:
            cprint(f"❌ Backtest Runner Agent error: {str(e)}", "red")
            time.sleep(60)  # Brief pause on error

if __name__ == "__main__":
    main()