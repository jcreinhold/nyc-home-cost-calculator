"""A portfolio of assets."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from nyc_home_cost_calculator.measures import (
    arithmetic_mean,
    beta,
    expected_shortfall,
    geometric_mean,
    maximum_drawdown,
    money_weighted_return,
    sharpe_ratio,
    sortino_ratio,
    standard_deviation,
    time_weighted_return,
    value_at_risk,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


TRADING_DAYS_PER_YEAR = 252


class Portfolio:
    """Represents a portfolio of assets."""

    def __init__(
        self,
        tickers: Sequence[str],
        weights: Sequence[float] | None = None,
        initial_investments: Sequence[float] | None = None,
        *,
        period: str = "5y",
        price: str = "Adj Close",
    ) -> None:
        """Initialize a Portfolio object.

        Args:
            tickers: List of ticker symbols for the assets in the portfolio.
            weights: List of weights corresponding to the assets in the portfolio.
            initial_investments: List of initial investments corresponding to the assets in the portfolio.
            period: The time period for fetching the data (default: "5y").
            price: The price metric to use for the data (default: "Adj Close").
        """
        self.tickers = tickers
        if weights is None and initial_investments is None:
            msg = "Either weights or initial_investments must be provided"
            raise ValueError(msg)

        if weights is not None and initial_investments is not None:
            msg = "Provide either weights or initial_investments, not both"
            raise ValueError(msg)

        if initial_investments is not None:
            self.initial_investments = np.asarray(initial_investments)
            total_investment = self.initial_investments.sum()
            self.weights = self.initial_investments / total_investment
        else:
            self.weights = np.asarray(weights)
            if not np.isclose(self.weights.sum(), 1.0, rtol=1e-5):
                msg = "Weights must sum to 1.0"
                raise ValueError(msg)
            self.initial_investments = self.weights * 100_000

        self.period = period
        self.price = price
        self.full_data = self._fetch_data()
        self.data = self.full_data[self.price].dropna()
        self.returns = self._calculate_returns()
        self.weighted_returns = self._calculate_weighted_returns()
        self.dollar_returns = self._calculate_dollar_returns()
        self.metrics = self._calculate_metrics()

    def _fetch_data(self) -> pd.DataFrame:
        return yf.download(self.tickers, period=self.period)

    def _calculate_returns(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.data.pct_change().dropna())

    def _calculate_weighted_returns(self) -> pd.Series:
        return (self.returns * self.weights).sum(axis=1)

    def _calculate_dollar_returns(self) -> pd.Series:
        cumulative_returns = (1 + self.returns).cumprod()
        dollar_returns = cumulative_returns * self.initial_investments
        return dollar_returns.sum(axis=1) - self.initial_investments.sum()

    def _calculate_metrics(self) -> dict[str, float]:
        returns = self.weighted_returns
        values = (1 + returns).cumprod()

        annualized_return = geometric_mean(returns) * TRADING_DAYS_PER_YEAR
        max_dd = maximum_drawdown(values)
        final_dollar_return = self.dollar_returns.iloc[-1]

        metrics = {
            "arithmetic_mean": float(arithmetic_mean(returns) * TRADING_DAYS_PER_YEAR),
            "geometric_mean": float(annualized_return),
            "twr": float(time_weighted_return(returns)),
            "irr": float(money_weighted_return(values)),
            "sortino_ratio": float(sortino_ratio(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "var_95": float(value_at_risk(returns)),
            "es_95": float(expected_shortfall(returns)),
            "volatility": float(standard_deviation(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "sharpe_ratio": float(sharpe_ratio(returns)),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(annualized_return / abs(max_dd)) if max_dd != 0 else np.inf,
            "avg_correlation": float(self._calculate_avg_correlation()),
            "final_dollar_return": float(final_dollar_return),
        }

        # Calculate beta if a market index is provided
        if hasattr(self, "market_returns"):
            metrics["beta"] = float(beta(returns, self.market_returns.to_numpy()))

        return metrics

    def set_market_index(self, market_ticker: str) -> None:
        """Set a market index for beta calculation."""
        market_data = yf.download(market_ticker, period=self.period)[self.price]
        self.market_returns = market_data.pct_change().dropna()
        self.metrics = self._calculate_metrics()  # Recalculate metrics to include beta

    def _calculate_avg_correlation(self) -> float:
        """Calculate the average pairwise correlation between ticker returns."""
        corr_matrix = self.returns.corr()
        # Extract upper triangle of correlation matrix, excluding diagonal
        upper_triangle = np.triu(corr_matrix.values, k=1)
        # Calculate mean of non-zero elements
        return np.mean(upper_triangle[upper_triangle != 0])

    def plot_returns(self, *, figsize: tuple[int, int] = (10, 6)) -> None:
        """Plot the cumulative returns of the portfolio."""
        cum_returns = (1.0 + self.weighted_returns).cumprod()
        plt.figure(figsize=figsize)
        plt.plot(cum_returns.index, cum_returns.to_numpy())
        plt.title("Portfolio Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.show()

    def compare(self, other: Portfolio, *, significance_level: float = 0.05) -> dict[str, TestResult]:
        """Compare the metrics of this portfolio with another portfolio.

        Args:
            other: The other portfolio to compare with.
            significance_level: The significance level for the statistical test. Defaults to 0.05.

        Returns:
            The comparison results.
        """
        comparison: dict[str, TestResult] = {}

        for metric, value in self.metrics.items():
            other_value = other.metrics[metric]
            diff = float(value - other_value)

            if metric in {"arithmetic_mean", "geometric_mean"}:
                # Use t-test for return metrics
                t_stat, p_value = map(float, stats.ttest_ind(self.weighted_returns, other.weighted_returns))
                is_significant = p_value < significance_level
                comparison[metric] = TestResult(is_significant, t_stat, p_value, diff, significance_level)
            elif metric == "volatility":
                # Use F-test for volatility
                f_stat = np.var(self.weighted_returns) / np.var(other.weighted_returns)
                df1 = df2 = len(self.weighted_returns) - 1
                p_value = 2.0 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
                is_significant = p_value < significance_level
                comparison[metric] = TestResult(is_significant, f_stat, p_value, diff, significance_level)
            else:
                # For other metrics, just compute the difference
                comparison[metric] = TestResult(None, None, None, diff, significance_level)

        return comparison

    @staticmethod
    def comparison_to_str(comparison_dict: dict[str, TestResult]) -> str:
        """Print the comparison results."""
        results = []
        for metric, result in comparison_dict.items():
            _metric = metric.replace("_", " ")
            _metric = _metric.capitalize() if " " in _metric and "95" not in _metric else _metric.upper()
            if _metric == "VAR 95":
                _metric = "VaR 95"
            if result.is_significant is not None:
                significance = "Significant" if result.is_significant else "Not significant"
                results.append(f"{_metric}: {result.diff:.4f} ({significance}, p={result.p_value:.4f})")
            else:
                results.append(f"{_metric}: {result.diff:.4f} (Statistical significance not applicable)")
        return "\n".join(results)


class TestResult(NamedTuple):
    """Represents the result of a statistical test."""

    is_significant: bool | None = None
    test_stat: float | None = None
    p_value: float | None = None
    diff: float | None = None
    significance_level: float = 0.05

    def __str__(self) -> str:
        """Return a string representation of the TestResult object."""
        if self.is_significant is None:
            return "Statistical significance not applicable"
        if self.is_significant:
            return f"Significant at {self.significance_level} level (p-value: {self.p_value:.4f})"
        return f"Not significant (p-value: {self.p_value:.4f})"
