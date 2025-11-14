"""
🧠 NeuroFlux Data Aggregator
Collects and aggregates data from all agents for analytics processing.

Built with love by Nyros Veil 🚀

Features:
- Unified data collection from all agent directories
- Time-series data processing and normalization
- Agent data validation and error handling
- Historical data aggregation and caching
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import glob
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import NeuroFlux config
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class DataAggregator:
    """Centralized data aggregator for all NeuroFlux agents"""

    def __init__(self, data_dir: str = "src/data"):
        self.data_dir = Path(data_dir)
        self.agent_directories = self._discover_agent_directories()
        self.cache = {}
        self.last_update = {}

        cprint(f"📊 DataAggregator initialized with {len(self.agent_directories)} agent directories", "cyan")

    def _discover_agent_directories(self) -> Dict[str, Path]:
        """Discover all agent data directories"""
        agent_dirs = {}

        if not self.data_dir.exists():
            cprint(f"⚠️  Data directory {self.data_dir} does not exist", "yellow")
            return agent_dirs

        # Look for agent directories
        for item in self.data_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                agent_name = item.name.replace('_agent', '').replace('_', '')
                agent_dirs[agent_name] = item

        return agent_dirs

    def get_latest_reports(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest report for a specific agent"""
        if agent_name not in self.agent_directories:
            return None

        agent_dir = self.agent_directories[agent_name]
        latest_file = agent_dir / "latest_report.json"

        if not latest_file.exists():
            # Try alternative naming patterns
            alternatives = [
                agent_dir / f"latest_{agent_name}.json",
                agent_dir / f"{agent_name}_latest.json",
                agent_dir / "latest.json"
            ]

            for alt_file in alternatives:
                if alt_file.exists():
                    latest_file = alt_file
                    break

        if not latest_file.exists():
            return None

        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                # Add metadata
                data['_metadata'] = {
                    'agent': agent_name,
                    'source': 'latest_report',
                    'timestamp': datetime.now().isoformat(),
                    'file_path': str(latest_file)
                }
                return data
        except Exception as e:
            cprint(f"❌ Error reading latest report for {agent_name}: {e}", "red")
            return None

    def get_historical_data(self, agent_name: str, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a specific agent"""
        if agent_name not in self.agent_directories:
            return []

        agent_dir = self.agent_directories[agent_name]
        history_file = agent_dir / f"{agent_name}_history.jsonl"

        if not history_file.exists():
            # Try alternative patterns
            alternatives = [
                agent_dir / "history.jsonl",
                agent_dir / f"{agent_name.replace('_agent', '')}_history.jsonl"
            ]

            for alt_file in alternatives:
                if alt_file.exists():
                    history_file = alt_file
                    break

        if not history_file.exists():
            return []

        data = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        try:
            with open(history_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())

                            # Check timestamp if available
                            if 'timestamp' in record:
                                record_time = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                                if record_time < cutoff_time:
                                    continue

                            # Add metadata
                            record['_metadata'] = {
                                'agent': agent_name,
                                'source': 'history',
                                'timestamp': datetime.now().isoformat(),
                                'file_path': str(history_file)
                            }

                            data.append(record)
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            cprint(f"❌ Error reading historical data for {agent_name}: {e}", "red")

        return data

    def get_all_agents_latest(self) -> Dict[str, Dict[str, Any]]:
        """Get latest reports from all agents"""
        results = {}

        for agent_name in self.agent_directories.keys():
            latest = self.get_latest_reports(agent_name)
            if latest:
                results[agent_name] = latest

        cprint(f"📊 Collected latest reports from {len(results)} agents", "blue")
        return results

    def get_all_agents_history(self, hours_back: int = 24) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical data from all agents"""
        results = {}

        for agent_name in self.agent_directories.keys():
            history = self.get_historical_data(agent_name, hours_back)
            if history:
                results[agent_name] = history

        cprint(f"📊 Collected historical data from {len(results)} agents ({hours_back}h window)", "blue")
        return results

    def get_agent_summary(self, agent_name: str) -> Dict[str, Any]:
        """Get summary statistics for a specific agent"""
        summary = {
            'agent_name': agent_name,
            'has_latest_report': False,
            'history_records': 0,
            'last_update': None,
            'data_files': []
        }

        if agent_name not in self.agent_directories:
            return summary

        agent_dir = self.agent_directories[agent_name]

        # Check for latest report
        latest_file = agent_dir / "latest_report.json"
        if latest_file.exists():
            summary['has_latest_report'] = True
            summary['last_update'] = datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()

        # Count history records
        history_file = agent_dir / f"{agent_name}_history.jsonl"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    summary['history_records'] = sum(1 for line in f if line.strip())
            except:
                pass

        # List all data files
        summary['data_files'] = [f.name for f in agent_dir.glob("*") if f.is_file()]

        return summary

    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall system data summary"""
        summary = {
            'total_agents': len(self.agent_directories),
            'agents_with_data': 0,
            'total_history_records': 0,
            'last_system_update': datetime.now().isoformat(),
            'agent_summaries': {}
        }

        for agent_name in self.agent_directories.keys():
            agent_summary = self.get_agent_summary(agent_name)
            summary['agent_summaries'][agent_name] = agent_summary

            if agent_summary['has_latest_report'] or agent_summary['history_records'] > 0:
                summary['agents_with_data'] += 1

            summary['total_history_records'] += agent_summary['history_records']

        return summary

    def refresh_cache(self):
        """Refresh all cached data"""
        self.cache = {}
        self.last_update = {}
        cprint("🔄 Data cache refreshed", "green")

    def get_cached_data(self, key: str, fetch_func, ttl_seconds: int = 300):
        """Get cached data with TTL"""
        now = time.time()

        if key in self.cache and key in self.last_update:
            if now - self.last_update[key] < ttl_seconds:
                return self.cache[key]

        # Fetch fresh data
        data = fetch_func()
        self.cache[key] = data
        self.last_update[key] = now

        return data