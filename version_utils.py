"""
NeuroFlux Version Utilities
Safe package version detection with fallbacks for deprecated APIs
"""

import warnings
from typing import Optional


def get_package_version(package_name: str, fallback_module=None) -> str:
    """
    Get package version safely, handling deprecation warnings and API changes.

    Args:
        package_name: Name of the package (e.g., 'flask', 'numpy')
        fallback_module: Optional module object to try __version__ on

    Returns:
        Version string or 'unknown' if unable to determine
    """
    # Method 1: importlib.metadata (recommended for Python 3.8+)
    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        pass

    # Method 2: importlib_metadata (backport for older Python)
    try:
        import importlib_metadata
        return importlib_metadata.version(package_name)
    except Exception:
        pass

    # Method 3: pkg_resources (fallback)
    try:
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        pass

    # Method 4: Module __version__ attribute (with warning suppression)
    if fallback_module:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return fallback_module.__version__
        except Exception:
            pass

    return "unknown"


def get_multiple_versions(packages: dict) -> dict:
    """
    Get versions for multiple packages at once.

    Args:
        packages: Dict of {package_name: module_object}

    Returns:
        Dict of {package_name: version_string}
    """
    versions = {}
    for name, module in packages.items():
        versions[name] = get_package_version(name, module)
    return versions


def format_version_output(package_name: str, version: str, status: str = "✅") -> str:
    """
    Format version output for console display.

    Args:
        package_name: Name of the package
        version: Version string
        status: Status emoji (✅, ❌, ⚠️)

    Returns:
        Formatted string for console output
    """
    return f"{status} {package_name}: {version}"


# Convenience functions for common packages
def get_flask_version() -> str:
    """Get Flask version safely"""
    try:
        import flask
        return get_package_version('flask', flask)
    except ImportError:
        return "not installed"


def get_numpy_version() -> str:
    """Get NumPy version safely"""
    try:
        import numpy as np
        return get_package_version('numpy', np)
    except ImportError:
        return "not installed"


def get_pandas_version() -> str:
    """Get Pandas version safely"""
    try:
        import pandas as pd
        return get_package_version('pandas', pd)
    except ImportError:
        return "not installed"


def get_ccxt_version() -> str:
    """Get CCXT version safely"""
    try:
        import ccxt
        return get_package_version('ccxt', ccxt)
    except ImportError:
        return "not installed"