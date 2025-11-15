#!/usr/bin/env python3
"""
Test version detection utilities
"""

import sys
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from version_utils import (
    get_package_version, get_multiple_versions,
    get_flask_version, get_numpy_version,
    get_pandas_version, get_ccxt_version,
    format_version_output
)

def test_version_detection():
    """Test all version detection methods"""
    print("🧪 Testing Version Detection")
    print("=" * 40)

    # Test individual functions
    print(f"Flask: {get_flask_version()}")
    print(f"NumPy: {get_numpy_version()}")
    print(f"Pandas: {get_pandas_version()}")
    print(f"CCXT: {get_ccxt_version()}")

    print("\nTesting generic function:")
    packages = ['flask', 'numpy', 'pandas', 'ccxt', 'requests']
    for pkg in packages:
        version = get_package_version(pkg)
        print(f"  {pkg}: {version}")

    print("\nTesting multiple versions:")
    try:
        import numpy as np
        import pandas as pd
        import flask
        import ccxt

        modules = {
            'numpy': np,
            'pandas': pd,
            'flask': flask,
            'ccxt': ccxt
        }

        versions = get_multiple_versions(modules)
        for pkg, ver in versions.items():
            print(f"  {pkg}: {ver}")
    except ImportError as e:
        print(f"  Import error: {e}")

    print("\nTesting formatted output:")
    print(format_version_output("Flask", get_flask_version()))
    print(format_version_output("NumPy", get_numpy_version()))

    print("\n✅ Version detection test complete")

if __name__ == '__main__':
    test_version_detection()