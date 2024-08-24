"""NYC Home Cost Calculator.

A package for simulating and analyzing home ownership costs in New York City.

This package provides tools to calculate and visualize the long-term financial
implications of purchasing a home in NYC, taking into account factors such as
property appreciation, income changes, and market volatility.

Modules:
    calculator: Contains the main NYCHomeCostCalculator class.

Usage:
    from nyc_home_cost_calculator.calculator import NYCHomeCostCalculator

    calculator = NYCHomeCostCalculator(...)
    results = calculator.get_cost_statistics()
"""

from __future__ import annotations

__all__ = ["__version__"]

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0"
