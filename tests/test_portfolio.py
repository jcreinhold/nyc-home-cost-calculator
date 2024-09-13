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
    rng = np.random.default_rng(42)

    def _mock_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D", tz="UTC")
        data = pd.DataFrame(
            {
                "QQQ": rng.standard_normal(100).cumsum() + 100.0,
                "SPY": rng.standard_normal(100).cumsum() + 100.0,
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_product([["Adj Close"], data.columns])
        return data

    monkeypatch.setattr("yfinance.download", _mock_download)


def test_portfolio_initialization(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], [0.6, 0.4], market_ticker=None)
    assert isinstance(portfolio, Portfolio)
    assert portfolio.tickers == ["SPY", "QQQ"]
    np.testing.assert_array_almost_equal(portfolio.weights, np.array([0.6, 0.4]))


def test_data_fetching(mock_yf_download: pytest.MonkeyPatch) -> None:
    portfolio = Portfolio(["SPY", "QQQ"], [0.5, 0.5], market_ticker=None)
    assert isinstance(portfolio.data, pd.DataFrame)
    assert len(portfolio.data) == 100
    assert set(portfolio.data.columns) == {"SPY", "QQQ"}


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
        "geometric_mean",
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
