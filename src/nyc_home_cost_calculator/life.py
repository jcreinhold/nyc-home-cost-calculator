"""Simulate financial life events."""

import numpy as np

from nyc_home_cost_calculator.base import AbstractSimulatorBase
from nyc_home_cost_calculator.income import CareerIncomeSimulator
from nyc_home_cost_calculator.simulate import SimulationResults
from nyc_home_cost_calculator.tax import FilingStatus


class FinancialLifeSimulator(AbstractSimulatorBase):
    """Simulate financial life events."""

    def __init__(
        self,
        career_simulator: CareerIncomeSimulator,
        initial_age: int = 25,
        marriage_probability: float = 0.05,
        divorce_probability: float = 0.02,
        partner_income_ratio: float = 1.0,
        divorce_cost: float = 50_000.0,
        initial_marriage_status: FilingStatus = FilingStatus.SINGLE,
        simulations: int = 5_000,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the FinancialLifeSimulator.

        Args:
            career_simulator: The career income simulator.
            initial_age: The initial age. Defaults to 25.
            marriage_probability: The annual marriage probability. Defaults to 0.05.
            divorce_probability: The annual divorce probability. Defaults to 0.02.
            partner_income_ratio: The partner income ratio. Defaults to 1.0.
            divorce_cost: The divorce cost. Defaults to $50,000.
            initial_marriage_status: The initial marriage status. Defaults to FilingStatus.SINGLE.
            simulations: The number of simulations. Defaults to 5000.
            rng: The random number generator. Defaults to None.
        """
        super().__init__(total_years=career_simulator.total_years, simulations=simulations, rng=rng)
        self.career_simulator = career_simulator
        self.initial_age = initial_age
        self.marriage_probability = marriage_probability
        self.divorce_probability = divorce_probability
        self.partner_income_ratio = partner_income_ratio
        self.divorce_cost = divorce_cost
        self.initial_marriage_status = initial_marriage_status

    def _simulate_vectorized(self, months: np.ndarray) -> SimulationResults:
        total_months, num_simulations = shape = months.shape

        # Simulate career income
        career_results = self.career_simulator._simulate_vectorized(months)  # noqa: SLF001
        if career_results.personal_income is None:
            msg = "Career simulator must return personal_income in results."
            raise ValueError(msg)
        personal_income = career_results.personal_income

        # Initialize marital status
        marital_status = np.full(
            shape,
            self.initial_marriage_status in {FilingStatus.MARRIED_JOINT, FilingStatus.MARRIED_SEPARATE},
            dtype=np.int64,
        )

        # Generate marriage and divorce events
        marriages = self.rng.random(shape) < self.marriage_probability / 12.0
        divorces = self.rng.random(shape) < self.divorce_probability / 12.0

        # Update marital status
        for i in range(1, total_months):
            # Only single people can get married
            new_marriages = marriages[i] & (marital_status[i - 1] == 0)
            # Only married people can get divorced
            new_divorces = divorces[i] & (marital_status[i - 1] == 1)
            marital_status[i] = marital_status[i - 1] + new_marriages - new_divorces

        # Calculate partner income
        partner_income = np.zeros_like(personal_income)
        partner_income[marital_status == 1] = personal_income[marital_status == 1] * self.partner_income_ratio

        # Calculate divorce costs
        divorce_costs = np.where(marital_status[1:] < marital_status[:-1], self.divorce_cost, 0.0)
        divorce_costs = np.insert(divorce_costs, 0, 0.0, axis=0)  # Add a row of zeros at the beginning

        # Calculate total household income
        household_income = personal_income + partner_income

        return SimulationResults(
            monthly_costs=household_income / 12.0,
            profit_loss=np.cumsum(household_income / 12.0, axis=0),
            total_years=self.total_years,
            simulations=self.career_simulator.simulations,
            monthly_income=household_income / 12.0,
            personal_income=personal_income,
            partner_income=partner_income,
            household_income=household_income,
            marital_status=marital_status,
            promotions=career_results.promotions,
            demotions=career_results.demotions,
            layoffs=career_results.layoffs,
            extra={
                "layoff_durations": career_results.extra["layoff_durations"],
                "layoff_mask": career_results.extra["layoff_mask"],
                "divorce_costs": divorce_costs,
            },
        )

    def _get_input_parameters(self) -> list[tuple[str, str]]:
        career_params = self.career_simulator._get_input_parameters()  # noqa: SLF001
        return [
            *career_params,
            ("Initial Age", f"{self.initial_age}"),
            ("Annual Marriage Probability", f"{self.marriage_probability:.2%}"),
            ("Annual Divorce Probability", f"{self.divorce_probability:.2%}"),
            ("Partner Income Ratio", f"{self.partner_income_ratio:.2f}"),
            ("Divorce Cost", f"${self.divorce_cost:,.2f}"),
            ("Number of Simulations", f"{self.simulations}"),
        ]
