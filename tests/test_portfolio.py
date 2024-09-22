from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

mpl.use("Agg")

from nyc_home_cost_calculator.portfolio import Portfolio

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def mock_yf_download(monkeypatch: pytest.MonkeyPatch) -> None:
    def _mock_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D", tz="UTC")
        price_data = pd.DataFrame(
            {
                "QQQ": rng.standard_normal(100).cumsum() + 100.0,
                "SPY": rng.standard_normal(100).cumsum() + 100.0,
            },
            index=dates,
        )
        price_data.columns = pd.MultiIndex.from_product([["Adj Close"], price_data.columns])

        if kwargs.get("actions", False):
            dividend_data = pd.DataFrame(
                {
                    "QQQ": [0.0] * 100,
                    "SPY": [0.0] * 100,
                },
                index=dates,
            )
            # Add a dividend for QQQ on day 50
            dividend_data.loc[dates[50], "QQQ"] = 100.0
            # Add dividend for SPY on day 75
            dividend_data.loc[dates[75], "SPY"] = 100.0
            dividend_data.columns = pd.MultiIndex.from_product([["Dividends"], dividend_data.columns])
            return pd.concat([price_data, dividend_data], axis=1)
        return price_data

    monkeypatch.setattr("yfinance.download", _mock_download)


def test_portfolio_initialization(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], [0.6, 0.4], market_ticker=None)
    assert isinstance(portfolio, Portfolio)
    assert portfolio.tickers == ["SPY", "QQQ"]
    np.testing.assert_array_almost_equal(portfolio.weights, np.array([0.6, 0.4]))


def test_data_fetching(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], [0.5, 0.5], market_ticker=None)
    assert isinstance(portfolio.price_values, pd.DataFrame)
    assert len(portfolio.price_values) == 100
    assert set(portfolio.price_values.columns) == {"SPY", "QQQ"}


def test_returns_calculation(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], [0.5, 0.5], market_ticker=None)
    assert isinstance(portfolio.returns, pd.DataFrame)
    assert len(portfolio.returns) == 99  # One less than data due to pct_change
    assert set(portfolio.returns.columns) == {"SPY", "QQQ"}


def test_metrics_calculation(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], [0.5, 0.5], market_ticker=None)
    metrics = portfolio.metrics
    assert isinstance(metrics, dict)
    expected_metrics = [
        "cagr",
        "volatility",
        "sharpe_ratio",
        "max_drawdown",
        "sortino_ratio",
        "calmar_ratio",
    ]
    assert all(metric in metrics for metric in expected_metrics)


def test_portfolio_comparison(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio1 = Portfolio(["SPY", "QQQ"], [0.6, 0.4], market_ticker=None)
    portfolio2 = Portfolio(["SPY", "QQQ"], [0.4, 0.6], market_ticker=None)
    comparison = portfolio1.compare(portfolio2)
    assert isinstance(comparison, dict)
    comparison_str = Portfolio.comparison_to_str(comparison)
    assert isinstance(comparison_str, str)
    assert "return:" in comparison_str
    assert "VaR 95:" in comparison_str
    assert "Sharpe ratio:" in comparison_str


@pytest.mark.parametrize(
    "weights",
    [
        [0.5, 0.5],
        [0.7, 0.3],
        [1.0, 0.0],
    ],
)
def test_portfolio_with_different_weights(mock_yf_download: pytest.MonkeyPatch, weights: list[float]) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], weights, market_ticker=None)
    assert isinstance(portfolio, Portfolio)
    np.testing.assert_array_almost_equal(portfolio.weights, np.array(weights))


def test_invalid_weights(mock_yf_download: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="Weights must sum to 1.0"):
        Portfolio(["SPY", "QQQ"], [0.5, 0.6], market_ticker=None)  # Weights sum to > 1


def test_plot_returns(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], [0.5, 0.5], market_ticker=None)
    portfolio.plot_returns()


def test_portfolio_with_dividend_reinvestment(mock_yf_download: pytest.MonkeyPatch) -> None:
    pf_with_dist = Portfolio(["SPY", "QQQ"], [0.5, 0.5], reinvest_distributions=True, market_ticker=None)
    pf_wo_dist = Portfolio(["SPY", "QQQ"], [0.5, 0.5], reinvest_distributions=False, market_ticker=None)

    # The portfolio with reinvested dividends should have a higher final value
    assert pf_with_dist.dollar_returns.iloc[-1] > pf_wo_dist.dollar_returns.iloc[-1]

    # Check if the return on the dividend payment dates is higher for the portfolio with reinvested dividends
    assert pf_with_dist.returns.iloc[49].loc["QQQ"] > pf_wo_dist.returns.iloc[49].loc["QQQ"]
    assert pf_with_dist.returns.iloc[74].loc["SPY"] > pf_wo_dist.returns.iloc[74].loc["SPY"]

    # Check that the weights are correctly applied
    assert len(pf_with_dist.weighted_returns) == len(pf_with_dist.returns)
    assert len(pf_wo_dist.weighted_returns) == len(pf_wo_dist.returns)
