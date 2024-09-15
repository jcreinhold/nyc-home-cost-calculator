"""Simulate career income in the NYC Home Cost Calculator."""

import numpy as np
from scipy import stats

from nyc_home_cost_calculator.base import AbstractSimulatorBase
from nyc_home_cost_calculator.simulate import SimulationResults


class CareerIncomeSimulator(AbstractSimulatorBase):
    """A class that simulates career income in the NYC Home Cost Calculator."""

    def __init__(
        self,
        initial_income: float = 75_000.0,
        total_years: int = 30,
        mean_income_growth: float = 0.03,
        income_volatility: float = 0.02,
        promotion_probability: float = 0.05,
        promotion_increase: float = 0.15,
        demotion_probability: float = 0.02,
        demotion_decrease: float = 0.1,
        layoff_probability: float = 0.03,
        mean_layoff_duration_months: float = 6.0,
        layoff_income_impact: float = 0.1,
        simulations: int = 5_000,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the CareerIncomeSimulator object.

        Args:
            initial_income: The initial income.
            total_years: The total number of years.
            mean_income_growth: The mean income growth rate.
            income_volatility: The income volatility.
            promotion_probability: The probability of promotion.
            promotion_increase: The increase in income due to promotion.
            demotion_probability: The probability of demotion.
            demotion_decrease: The decrease in income due to demotion.
            layoff_probability: The probability of layoff.
            mean_layoff_duration_months: The average duration of layoff in months.
            layoff_income_impact: The impact of layoff on income.
            simulations: The number of simulations.
            rng: The random number generator.
        """
        super().__init__(total_years=total_years, simulations=simulations, rng=rng)
        self.initial_income = initial_income
        self.mean_income_growth = mean_income_growth
        self.income_volatility = income_volatility
        self.promotion_probability = promotion_probability
        self.promotion_increase = promotion_increase
        self.demotion_probability = demotion_probability
        self.demotion_decrease = demotion_decrease
        self.layoff_probability = layoff_probability
        self.mean_layoff_duration_months = mean_layoff_duration_months
        self.layoff_income_impact = layoff_income_impact

    def _simulate_vectorized(self, months: np.ndarray) -> SimulationResults:  # noqa: PLR0914
        total_months, num_simulations = shape = months.shape

        # Generate monthly income growth rates
        monthly_growth = stats.norm.rvs(
            loc=self.mean_income_growth / 12.0,
            scale=self.income_volatility / np.sqrt(12.0),
            size=shape,
            random_state=self.rng,
        )

        # Generate events (promotions, demotions, layoffs)
        promotions = self.rng.random(shape) < self.promotion_probability / 12.0
        demotions = self.rng.random(shape) < self.demotion_probability / 12.0
        layoffs = self.rng.random(shape) < self.layoff_probability / 12.0

        # Generate random layoff durations
        layoff_durations = self.rng.poisson(lam=self.mean_layoff_duration_months, size=shape)

        # Calculate cumulative growth factors
        cumulative_growth = np.cumprod(1.0 + monthly_growth, axis=0)

        # Apply promotions and demotions
        promotion_factors = np.cumprod(1.0 + promotions * self.promotion_increase, axis=0)
        demotion_factors = np.cumprod(1.0 - demotions * self.demotion_decrease, axis=0)

        # Calculate base income without layoffs
        income = self.initial_income * cumulative_growth * promotion_factors * demotion_factors

        # Apply layoffs
        unemployment_benefit_rate = 0.5  # Unemployment benefit as a fraction of previous income
        max_weekly_benefit = 504.0  # Maximum weekly unemployment benefit in NYS as of 2023
        max_monthly_benefit = max_weekly_benefit * 52.0 / 12.0  # Convert to monthly

        layoff_mask = np.zeros(shape, dtype=np.bool_)
        for i in range(total_months):
            new_layoffs = layoffs[i] & ~layoff_mask[i]
            layoff_end = np.minimum(i + layoff_durations[i], total_months - 1)
            for j in range(num_simulations):
                if new_layoffs[j]:
                    layoff_mask[i : layoff_end[j], j] = True

        prev_income = np.roll(income, 1, axis=0)
        prev_income[0] = self.initial_income

        unemployment_benefit = np.minimum(prev_income * unemployment_benefit_rate, max_monthly_benefit)
        income = np.where(layoff_mask, unemployment_benefit, income)

        # Apply post-layoff income impact
        post_layoff_mask = np.roll(layoff_mask, 1, axis=0) & ~layoff_mask
        post_layoff_mask[0] = False
        income[post_layoff_mask] *= 1.0 - self.layoff_income_impact

        # Calculate monthly income
        monthly_income = income / 12.0

        return SimulationResults(
            monthly_costs=np.zeros_like(income),  # No costs in this simulation
            profit_loss=np.cumsum(income, axis=0),  # Treating income as "profit"
            total_years=self.total_years,
            simulations=self.simulations,
            personal_income=income,
            monthly_income=monthly_income,
            promotions=promotions,
            demotions=demotions,
            layoffs=layoffs,
            extra={
                "layoff_durations": layoff_durations,
                "layoff_mask": layoff_mask,
            },
        )

    def _get_input_parameters(self) -> list[tuple[str, str]]:
        return [
            ("Initial Annual Income", f"${self.initial_income:,.2f}"),
            ("Total Years", f"{self.total_years}"),
            ("Mean Annual Income Growth", f"{self.mean_income_growth:.2%}"),
            ("Income Volatility", f"{self.income_volatility:.2%}"),
            ("Annual Promotion Probability", f"{self.promotion_probability:.2%}"),
            ("Promotion Increase", f"{self.promotion_increase:.2%}"),
            ("Annual Demotion Probability", f"{self.demotion_probability:.2%}"),
            ("Demotion Decrease", f"{self.demotion_decrease:.2%}"),
            ("Annual Layoff Probability", f"{self.layoff_probability:.2%}"),
            ("Mean Layoff Duration (Months)", f"{self.mean_layoff_duration_months:.1f}"),
            ("Layoff Income Impact", f"{self.layoff_income_impact:.2%}"),
            ("Number of Simulations", f"{self.simulations}"),
        ]
