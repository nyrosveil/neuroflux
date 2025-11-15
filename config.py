"""
NeuroFlux Configuration Management
Smart environment-based configuration loading
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

def parse_size(size_str: str) -> int:
    """Parse size string like '100MB', '50KB', '1GB'"""
    size_str = size_str.upper()
    if size_str.endswith('GB'):
        return int(size_str.replace('GB', '')) * 1024 * 1024 * 1024
    elif size_str.endswith('MB'):
        return int(size_str.replace('MB', '')) * 1024 * 1024
    elif size_str.endswith('KB'):
        return int(size_str.replace('KB', '')) * 1024
    else:
        return int(size_str)  # Assume bytes if no unit

# Determine environment
ENV = os.getenv('FLASK_ENV', 'development')
ENV_FILE = PROJECT_ROOT / f'.env.{ENV}'

# Load base configuration
base_env = PROJECT_ROOT / '.env'
if base_env.exists():
    load_dotenv(base_env)

# Load environment-specific configuration
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

class Config:
    """NeuroFlux Configuration Class"""

    # Environment
    ENV = ENV
    DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    TESTING = os.getenv('FLASK_TESTING', 'false').lower() == 'true'

    # Server Configuration
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', '5001'))
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

    # Flask Configuration
    FLASK_ENV = ENV
    FLASK_DEBUG = DEBUG
    FLASK_TESTING = TESTING

    # Session Configuration
    SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', '3600'))
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_REQUEST_SIZE', '10MB').replace('MB', '')) * 1024 * 1024

    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    if CORS_ORIGINS == '*':
        CORS_ORIGINS = ['*']
    else:
        CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS.split(',')]

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE')
    LOG_MAX_SIZE = parse_size(os.getenv('LOG_MAX_SIZE', '100MB'))
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

    # Performance Configuration
    GUNICORN_WORKERS = int(os.getenv('GUNICORN_WORKERS', '4'))
    GUNICORN_THREADS = int(os.getenv('GUNICORN_THREADS', '2'))
    MEMORY_LIMIT = os.getenv('MEMORY_LIMIT', '2G')
    CPU_QUOTA = os.getenv('CPU_QUOTA', '200%')

    # AI Model APIs
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    ANTHROPIC_KEY = os.getenv('ANTHROPIC_KEY')
    DEEPSEEK_KEY = os.getenv('DEEPSEEK_KEY')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')

    # Market Data APIs
    COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
    BIRDEYE_API_KEY = os.getenv('BIRDEYE_API_KEY')

    # Blockchain Configuration
    SOLANA_PRIVATE_KEY = os.getenv('SOLANA_PRIVATE_KEY')
    SOLANA_RPC_ENDPOINT = os.getenv('SOLANA_RPC_ENDPOINT', 'https://api.mainnet-beta.solana.com')

    # Exchange APIs
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    HYPER_LIQUID_ETH_PRIVATE_KEY = os.getenv('HYPER_LIQUID_ETH_PRIVATE_KEY')

    # X10 Exchange
    X10_API_KEY = os.getenv('X10_API_KEY')
    X10_PRIVATE_KEY = os.getenv('X10_PRIVATE_KEY')
    X10_PUBLIC_KEY = os.getenv('X10_PUBLIC_KEY')
    X10_VAULT_ID = os.getenv('X10_VAULT_ID')

    # Coinbase API Keys (required even for public data)
    COINBASE_API_KEY = os.getenv('COINBASE_API_KEY')
    COINBASE_API_SECRET = os.getenv('COINBASE_API_SECRET')

    # Bybit API Keys
    BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
    BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET')

    # KuCoin API Keys
    KUCOIN_API_KEY = os.getenv('KUCOIN_API_KEY')
    KUCOIN_API_SECRET = os.getenv('KUCOIN_API_SECRET')

    # NeuroFlux Core Configuration
    FLUX_SENSITIVITY = float(os.getenv('FLUX_SENSITIVITY', '0.8'))
    ADAPTIVE_LEARNING_RATE = float(os.getenv('ADAPTIVE_LEARNING_RATE', '0.1'))
    NEURAL_NETWORK_LAYERS = eval(os.getenv('NEURAL_NETWORK_LAYERS', '[64, 32, 16]'))
    SWARM_SIZE = int(os.getenv('SWARM_SIZE', '6'))

    # ML Configuration
    ML_ENABLED = os.getenv('ML_ENABLED', 'true').lower() == 'true'
    ML_MODEL_CACHE_DIR = os.getenv('ML_MODEL_CACHE_DIR', str(PROJECT_ROOT / 'models'))

    # Monitoring Configuration
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
    METRICS_ENABLED = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    PROMETHEUS_PORT = int(os.getenv('PROMETHEUS_PORT', '9090'))

    # Backup Configuration
    BACKUP_ENABLED = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
    BACKUP_SCHEDULE = os.getenv('BACKUP_SCHEDULE', 'daily')
    BACKUP_RETENTION_DAYS = int(os.getenv('BACKUP_RETENTION_DAYS', '30'))
    BACKUP_DIR = os.getenv('BACKUP_DIR', str(PROJECT_ROOT / 'backups'))

    # System Configuration
    TIMEZONE = os.getenv('TIMEZONE', 'UTC')
    LOCALE = os.getenv('LOCALE', 'en_US.UTF-8')

    # Environment Detection
    IS_CONDA = bool(os.getenv('CONDA_DEFAULT_ENV'))
    IS_VENV = bool(os.getenv('VIRTUAL_ENV'))
    CONDA_ENV_NAME = os.getenv('CONDA_DEFAULT_ENV')
    VENV_PATH = os.getenv('VIRTUAL_ENV')

    # Python Path Configuration
    PYTHONPATH = [
        str(PROJECT_ROOT / 'src'),
        str(PROJECT_ROOT)
    ]

    # Exchange Configuration
    DEFAULT_EXCHANGE = os.getenv('DEFAULT_EXCHANGE', 'binance')
    EXCHANGE_PRIORITY = ['binance', 'bybit', 'kucoin', 'coinbase']

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary for easy access"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and not callable(getattr(cls, key))
        }

    @classmethod
    def validate(cls) -> list:
        """Validate configuration and return list of issues"""
        issues = []

        # Check required configurations
        if not cls.SECRET_KEY or cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            issues.append("SECRET_KEY not set or using default (security risk)")

        if cls.ENV == 'production' and cls.DEBUG:
            issues.append("DEBUG should be False in production")

        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_KEY:
            issues.append("No AI API keys configured (limited functionality)")

        # Check environment setup
        if not cls.IS_CONDA and not cls.IS_VENV:
            issues.append("Not running in conda or virtual environment")

        # Check critical paths
        if cls.LOG_FILE and not os.path.exists(os.path.dirname(cls.LOG_FILE or '')):
            issues.append(f"Log directory does not exist: {os.path.dirname(cls.LOG_FILE)}")

        return issues

    @classmethod
    def print_info(cls):
        """Print configuration information"""
        print("NeuroFlux Configuration")
        print("=======================")

        print(f"Environment: {cls.ENV}")
        print(f"Debug: {cls.DEBUG}")
        print(f"Host: {cls.HOST}:{cls.PORT}")

        print(f"Conda Environment: {cls.CONDA_ENV_NAME or 'None'}")
        print(f"Virtual Environment: {cls.VENV_PATH or 'None'}")

        # API Keys status
        api_keys = []
        if cls.OPENAI_API_KEY: api_keys.append("OpenAI")
        if cls.ANTHROPIC_KEY: api_keys.append("Anthropic")
        if cls.COINGECKO_API_KEY: api_keys.append("CoinGecko")
        print(f"Configured APIs: {', '.join(api_keys) if api_keys else 'None'}")

        # Validation issues
        issues = cls.validate()
        if issues:
            print(f"Configuration Issues: {len(issues)}")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration: Valid")


# Create global config instance
config = Config()

# Set Python path
for path in config.PYTHONPATH:
    if path not in sys.path:
        sys.path.insert(0, path)