"""A portfolio of assets."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, TypeVar, cast

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.transforms import Bbox
from scipy import stats

from nyc_home_cost_calculator.measures import (
    arithmetic_mean,
    beta,
    calmar_ratio,
    expected_shortfall,
    geometric_mean,
    maximum_drawdown,
    modified_var,
    money_weighted_return,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    standard_deviation,
    time_weighted_return,
    value_at_risk,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    T = TypeVar("T", pd.Series, pd.DataFrame)

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
        market_ticker: str | None = "^GSPC",
        tz: str | None = None,
    ) -> None:
        """Initialize a Portfolio object.

        Args:
            tickers: List of ticker symbols for the assets in the portfolio.
            weights: List of weights corresponding to the assets in the portfolio.
            initial_investments: List of initial investments corresponding to the assets in the portfolio.
            period: The time period for fetching the data (default: "5y").
            price: The price metric to use for the data (default: "Adj Close").
            market_ticker: The ticker symbol for the market index (default: "^GSPC").
            tz: The timezone for the data (default: "UTC").
        """
        self.tickers = tickers
        if weights is None and initial_investments is None:
            msg = "Either weights or initial_investments must be provided"
            raise ValueError(msg)

        if weights is not None and initial_investments is not None:
            msg = "Provide either weights or initial_investments, not both"
            raise ValueError(msg)

        if initial_investments is not None:
            self.initial_investments = np.asanyarray(initial_investments)
            total_investment = self.initial_investments.sum()
            self.weights = self.initial_investments / total_investment
        else:
            self.weights = np.asarray(weights)
            if not np.isclose(self.weights.sum(), 1.0, rtol=1e-5):
                msg = "Weights must sum to 1.0"
                raise ValueError(msg)
            self.initial_investments = self.weights * 100_000.0

        self.period = period
        self.price = price
        self.tz = tz
        self.full_data = self._fetch_data()
        self.full_data = self._set_tz(self.full_data)
        self.data = self.full_data[self.price].dropna()
        self.returns = self._calculate_returns()
        self.weighted_returns = self._calculate_weighted_returns()
        self.dollar_returns = self._calculate_dollar_returns()
        if market_ticker is not None:
            self.set_market_index(market_ticker, recalculate=False)
        self.metrics = self._calculate_metrics()

    def _fetch_data(self) -> pd.DataFrame:
        return yf.download(self.tickers, period=self.period)

    def _calculate_returns(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.data.pct_change().dropna())

    def _calculate_weighted_returns(self) -> pd.Series:
        return (self.returns * self.weights).sum(axis=1)

    def _calculate_dollar_returns(self) -> pd.Series:
        cumulative_returns = (1.0 + self.returns).cumprod()
        dollar_returns = cumulative_returns * self.initial_investments
        return dollar_returns.sum(axis=1) - self.initial_investments.sum()

    def _calculate_metrics(self) -> dict[str, float]:
        returns = self.weighted_returns

        values = (1.0 + returns).cumprod()
        final_dollar_return = self.dollar_returns.iloc[-1]

        metrics = {
            "arithmetic_mean": float(arithmetic_mean(returns) * TRADING_DAYS_PER_YEAR),
            "geometric_mean": float(geometric_mean(returns) * TRADING_DAYS_PER_YEAR),
            "twr": float(time_weighted_return(returns)),
            "irr": float(money_weighted_return(values)),
            "sortino_ratio": float(sortino_ratio(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "var_95": float(value_at_risk(returns)),
            "es_95": float(expected_shortfall(returns)),
            "mvar_95": float(modified_var(returns)),
            "omega_ratio": float(omega_ratio(returns)),
            "volatility": float(standard_deviation(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)),
            "sharpe_ratio": float(sharpe_ratio(returns)),
            "max_drawdown": float(maximum_drawdown(values)),
            "calmar_ratio": float(calmar_ratio(returns, period=TRADING_DAYS_PER_YEAR)),
            "avg_correlation": float(self._calculate_avg_correlation()),
            "final_dollar_return": float(final_dollar_return),
        }

        # Calculate beta if a market index is provided
        if hasattr(self, "market_returns"):
            metrics["beta"] = float(beta(returns, self.market_returns))

        return metrics

    def _set_tz(self, series: T) -> T:
        return series.tz_localize(self.tz) if series.index.tzinfo is None else series.tz_convert(self.tz)  # type: ignore[attr-defined]

    def set_market_index(self, market_ticker: str, *, recalculate: bool = True) -> None:
        """Set a market index for beta calculation."""
        market_data = yf.download(market_ticker, period=self.period)[self.price]
        market_data = self._set_tz(market_data)
        self.market_ticker = market_ticker
        self.market_returns = market_data.pct_change().dropna()
        if recalculate:
            self.metrics = self._calculate_metrics()  # Recalculate metrics to include beta

    def _calculate_avg_correlation(self) -> float:
        """Calculate the average pairwise correlation between ticker returns."""
        corr_matrix = self.returns.corr()
        # Extract upper triangle of correlation matrix, excluding diagonal
        upper_triangle = np.triu(corr_matrix.values, k=1)
        # Calculate mean of non-zero elements
        return np.mean(upper_triangle[upper_triangle != 0])

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
            if _metric == "MVAR 95":
                _metric = "mVaR 95"
            if result.is_significant is not None:
                significance = "Significant" if result.is_significant else "Not significant"
                results.append(f"{_metric}: {result.diff:.4f} ({significance}, p={result.p_value:.4f})")
            else:
                results.append(f"{_metric}: {result.diff:.4f} (Statistical significance not applicable)")
        return "\n".join(results)

    def plot_returns(self, *, figsize: tuple[int, int] = (10, 6)) -> tuple[plt.Figure, plt.Axes]:
        """Plot the cumulative returns of the portfolio."""
        cum_returns = (1.0 + self.weighted_returns).cumprod()
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(cum_returns.index, cum_returns.to_numpy())
        ax.set_title("Portfolio Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        return fig, ax

    def plot_portfolio(
        self, metrics: list[str] | None = None, *, figsize: tuple[int, int] = (15, 10)
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create a comprehensive plot of the portfolio's performance and metrics.

        Args:
            metrics: List of metrics to plot. If None, plots all available metrics.
            figsize: Size of the figure (width, height) in inches.

        Returns:
            A tuple containing the Figure and ndarray of Axes.
        """
        if metrics is None:
            metrics = list(self.metrics.keys())

        fig, axs = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle(f"Portfolio Analysis: {', '.join(self.tickers)}", fontsize=16)

        # Plot cumulative returns
        cum_returns = (1.0 + self.weighted_returns).cumprod()
        axs[0, 0].plot(cum_returns.index, cum_returns)
        axs[0, 0].set_title("Cumulative Returns")
        axs[0, 0].set_xlabel("Date")
        axs[0, 0].set_ylabel("Cumulative Return")

        # Plot rolling volatility
        rolling_vol = self.weighted_returns.rolling(window=30).std() * np.sqrt(252.0)
        axs[0, 1].plot(rolling_vol.index, rolling_vol)
        axs[0, 1].set_title("30-Day Rolling Volatility (Annualized)")
        axs[0, 1].set_xlabel("Date")
        axs[0, 1].set_ylabel("Volatility")

        # Plot drawdowns
        drawdowns = (cum_returns / cum_returns.cummax()) - 1.0
        axs[1, 0].fill_between(drawdowns.index, drawdowns, 0, alpha=0.3)
        axs[1, 0].set_title("Drawdowns")
        axs[1, 0].set_xlabel("Date")
        axs[1, 0].set_ylabel("Drawdown")

        # Plot returns distribution
        axs[1, 1].hist(self.weighted_returns, bins=50, density=True, alpha=0.7)
        axs[1, 1].set_title("Returns Distribution")
        axs[1, 1].set_xlabel("Return")
        axs[1, 1].set_ylabel("Frequency")

        # Plot asset allocation
        _ = _plot_treemap(self.weights, self.tickers, ax=axs[2, 0])
        axs[2, 0].set_title("Asset Allocation")

        # Plot key metrics as a table
        _ = _plot_table(self.metrics, ax=axs[2, 1])

        fig.tight_layout()
        return fig, axs


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


def _clean_metric_name(name: str) -> str:
    if name == "var_95":
        return "VaR 95"
    if name == "mvar_95":
        return "mVaR 95"
    if name == "es_95":
        return "ES 95"
    if name in {"twr", "irr"}:
        return name.upper()
    return name.replace("_", " ").title()


def _format_metric_value(name: str, value: float) -> str:
    percentage_metrics = {
        "arithmetic_mean",
        "geometric_mean",
        "twr",
        "irr",
        "max_drawdown",
        "var_95",
        "es_95",
        "mvar_95",
        "volatility",
    }
    if name in percentage_metrics:
        return f"{value:.2%}"
    dollar_metrics = {"final_dollar_return"}
    if name in dollar_metrics:
        return f"${value:,.2f}"
    return f"{value:.4f}"


def _squarify(sizes: Sequence[float], x: float, y: float, width: float, height: float) -> list[dict[str, float]]:
    if len(sizes) == 0:
        return []

    total = sum(sizes)
    sizes = [size / total * width * height for size in sizes]

    rectangles = []
    while sizes:
        rect = {"x": x, "y": y}
        if width < height:
            rect["dx"] = width
            rect["dy"] = sizes[0] / width
            y += rect["dy"]
            height -= rect["dy"]
        else:
            rect["dy"] = height
            rect["dx"] = sizes[0] / height
            x += rect["dx"]
            width -= rect["dx"]
        rectangles.append(rect)
        sizes.pop(0)
        if len(sizes) > 0:
            sizes = [size * (width * height) / sum(sizes) for size in sizes]

    return rectangles


def _plot_treemap(sizes: Sequence[float], labels: Sequence[str], ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()

    rectangles = _squarify(sizes, 0, 0, 100, 100)
    viridis = mpl.colormaps["viridis"].resampled(8)
    colors = viridis(np.linspace(0, 1, len(sizes)))

    for rect, label, color in zip(rectangles, labels, colors, strict=True):
        ax.add_patch(plt.Rectangle((rect["x"], rect["y"]), rect["dx"], rect["dy"], facecolor=color))

        # Calculate font size based on rectangle size
        font_size = min(rect["dx"], rect["dy"]) / 4

        # Text with outline
        text = f"{label}\n{sizes[labels.index(label)]:.2%}"
        x, y = rect["x"] + rect["dx"] / 2, rect["y"] + rect["dy"] / 2

        # White outline
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            wrap=True,
            color="white",
            fontsize=font_size + 1,
            fontweight="bold",
            path_effects=[path_effects.withStroke(linewidth=3, foreground="black")],
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()

    return ax


def _plot_table(metrics: dict[str, float], ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    metric_names = [_clean_metric_name(m) for m in metrics]
    metric_values = [_format_metric_value(m, v) for m, v in metrics.items()]

    # Split metrics into two rows if there are more than 5
    if len(metrics) > 5:
        mid = len(metrics) // 2
        metric_names_rows = [metric_names[:mid], metric_names[mid:]]
        metric_values_rows = [metric_values[:mid], metric_values[mid:]]
    else:
        metric_names_rows = [metric_names]
        metric_values_rows = [metric_values]

    ax.axis("tight")
    ax.axis("off")

    table_height = 0.4 if len(metrics) > 5 else 0.2
    y_offset = 0.1

    for i, (names, values) in enumerate(zip(metric_names_rows, metric_values_rows, strict=True)):
        table = ax.table(
            cellText=[values],
            colLabels=names,
            cellLoc="center",
            loc="center",
            bbox=Bbox.from_bounds(0, 1 - (i + 1) * table_height - y_offset, 1, table_height),
        )

        table.auto_set_font_size(value=False)
        table.set_fontsize(9)
        table.auto_set_column_width(col=list(range(len(names))))

        # Apply color and style
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#4472C4")
            else:
                cell.set_facecolor("#E9EDF4")

    ax.set_title("Key Metrics", fontweight="bold", y=1.1)
    return ax
