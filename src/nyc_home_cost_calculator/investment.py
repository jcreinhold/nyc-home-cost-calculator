"""Calculate investment returns/costs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from nyc_home_cost_calculator.base import AbstractSimulatorBase
from nyc_home_cost_calculator.simulate import SimulationResults

if TYPE_CHECKING:  # pragma: no cover
    from typing import Any

    from nyc_home_cost_calculator.portfolio import Portfolio


class InvestmentCalculator(AbstractSimulatorBase):
    """Calculates investment-related costs for NYC."""

    def __init__(
        self,
        initial_investment: float = 100_000.0,
        monthly_contribution: float = 1_000.0,
        total_years: int = 30,
        mean_return_rate: float = 0.07,
        volatility: float = 0.15,
        degrees_of_freedom: int = 5,
        simulations: int = 5_000,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the InvestmentCalculator object.

        Args:
            initial_investment: The initial investment amount. Defaults to 100_000.0.
            monthly_contribution: The monthly contribution amount. Defaults to 1_000.0.
            total_years: The total number of years for the investment. Defaults to 30.
            mean_return_rate: The mean annual return rate. Defaults to 0.07.
            volatility: The annual volatility. Defaults to 0.15.
            degrees_of_freedom: The degrees of freedom for the Student's t-distribution. Defaults to 5.
            simulations: The number of simulations to run. Defaults to 5_000.
            rng: The random number generator. Defaults to None.
        """
        super().__init__(total_years=total_years, simulations=simulations, rng=rng)
        self.initial_investment = initial_investment
        self.monthly_contribution = monthly_contribution
        self.mean_return_rate = mean_return_rate
        self.volatility = volatility
        self.degrees_of_freedom = degrees_of_freedom

    def _simulate_vectorized(self, months: np.ndarray) -> SimulationResults:
        total_months, _ = shape = months.shape

        # Generate monthly returns using Student's t-distribution
        if self.volatility == 0.0:
            # Handle zero volatility case
            monthly_log_returns = np.full(shape, np.log1p(self.mean_return_rate / 12))
        else:
            monthly_log_returns = stats.t.rvs(
                df=self.degrees_of_freedom,
                loc=np.log1p(self.mean_return_rate / 12.0) - 0.5 * (self.volatility / np.sqrt(12.0)) ** 2,
                scale=self.volatility / np.sqrt(12.0),
                size=shape,
                random_state=self.rng,
            )

        # Calculate cumulative returns
        cumulative_log_returns = np.cumsum(monthly_log_returns, axis=0)

        # Convert log returns to simple returns
        cumulative_returns = np.expm1(cumulative_log_returns)

        # Create a matrix of monthly contributions
        contributions = np.full(shape, self.monthly_contribution)
        contributions[0] += self.initial_investment

        # Calculate portfolio values
        portfolio_values = np.zeros(shape)
        portfolio_values[0] = self.initial_investment * np.exp(monthly_log_returns[0])

        # Vectorized calculation of portfolio values
        for i in range(1, total_months):
            portfolio_values[i] = (portfolio_values[i - 1] + self.monthly_contribution) * np.exp(monthly_log_returns[i])

        # Calculate cumulative contributions
        cumulative_contributions = np.cumsum(contributions, axis=0)

        return SimulationResults(
            monthly_costs=-contributions,
            profit_loss=portfolio_values - cumulative_contributions,
            total_years=self.total_years,
            simulations=self.simulations,
            portfolio_values=portfolio_values,
            investment_returns=cumulative_returns,
            extra={
                "monthly_log_returns": monthly_log_returns,
            },
            _input_parameters=self._get_input_parameters(),
        )

    def _get_input_parameters(self) -> list[tuple[str, str]]:
        return [
            ("Initial Investment", f"${self.initial_investment:,.2f}"),
            ("Monthly Contribution", f"${self.monthly_contribution:,.2f}"),
            ("Total Years", f"{self.total_years}"),
            ("Mean Annual Return Rate", f"{self.mean_return_rate:.2%}"),
            ("Annual Volatility", f"{self.volatility:.2%}"),
            ("Degrees of Freedom", f"{self.degrees_of_freedom}"),
            ("Number of Simulations", f"{self.simulations}"),
        ]

    @classmethod
    def from_portfolio(cls, portfolio: Portfolio, **kwargs: Any) -> InvestmentCalculator:
        """Initialize InvestmentCalculator from a Portfolio instance.

        Args:
            portfolio: A Portfolio instance.
            **kwargs: Additional keyword arguments to override default parameters.

        Returns:
            An instance of InvestmentCalculator initialized with portfolio data.
        """
        # Calculate annualized return and volatility from portfolio data
        returns = portfolio.metrics["cagr"]
        volatility = portfolio.metrics["volatility"]

        # Set default parameters
        params = {
            "initial_investment": portfolio.price_values.iloc[0].sum(),
            "monthly_contribution": 0.0,  # Assume no additional contributions by default
            "total_years": len(portfolio.price_values) // 252,  # Assuming 252 trading days per year
            "mean_return_rate": returns,
            "volatility": volatility,
            "degrees_of_freedom": 5,  # Default value, can be overridden
            "simulations": 5_000,  # Default value, can be overridden
        }

        # Override defaults with any provided kwargs
        params.update(kwargs)

        return cls(**params)
