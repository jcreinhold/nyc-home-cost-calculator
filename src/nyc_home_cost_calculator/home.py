"""Module for calculating and simulating the long-term costs and potential profits/losses of home ownership in NYC."""

from __future__ import annotations

import numpy as np

from nyc_home_cost_calculator.base import AbstractNYCCostCalculator
from nyc_home_cost_calculator.income import CareerIncomeSimulator
from nyc_home_cost_calculator.life import FinancialLifeSimulator
from nyc_home_cost_calculator.simulate import SimulationResults
from nyc_home_cost_calculator.tax import FilingStatus, TaxCalculator
from nyc_home_cost_calculator.utils import NormalRV, StudentTRV


class NYCHomeCostCalculator(AbstractNYCCostCalculator):
    """A class to calculate and simulate the long-term costs and potential profits/losses of home ownership in NYC.

    This calculator takes into account various factors such as property appreciation,
    income changes, market volatility, taxes, and other associated costs of homeownership.
    It uses Monte Carlo simulation to project potential financial outcomes over time.
    """

    def __init__(
        self,
        home_price: float = 1_000_000.0,
        down_payment: float = 200_000.0,
        mortgage_rate: float = 0.05,
        loan_term: int = 30,
        initial_income: float = 150_000.0,
        initial_age: int = 30,
        hoa_fee: float = 500.0,
        insurance_rate: float = 0.005,
        maintenance_rate: float = 0.01,
        property_tax_rate: float = 0.01,
        mean_appreciation_rate: float = 0.03,
        appreciation_volatility: float = 0.05,
        mean_inflation_rate: float = 0.02,
        inflation_volatility: float = 0.01,
        mean_income_change_rate: float = 0.03,
        income_change_volatility: float = 0.03,
        retirement_contribution_rate: float = 0.15,
        filing_status: FilingStatus = FilingStatus.SINGLE,
        extra_payment: float = 0.0,
        extra_payment_start_year: int = 1,
        extra_payment_end_year: int = 30,
        purchase_closing_cost_rate: float = 0.03,
        sale_closing_cost_rate: float = 0.07,
        marriage_probability: float = 0.05,
        divorce_probability: float = 0.02,
        partner_income_ratio: float = 0.8,
        divorce_cost: float = 50_000.0,
        degrees_of_freedom: int = 5,
        simulations: int = 5_000,
        rng: np.random.Generator | None = None,
    ):
        """Initialize the NYCHomeCostCalculator with home purchase and financial parameters.

        Args:
            home_price: The purchase price of the home. Defaults to $1,000,000.
            down_payment: The initial down payment amount. Defaults to $200,000.
            mortgage_rate: Annual mortgage interest rate (as a decimal). Defaults to 5%.
            loan_term: Length of the mortgage in years. Defaults to 30 (a 30-year mortgage).
            initial_income: Initial annual income of the homeowner. Defaults to $150,000.
            initial_age: Initial age of the homeowner. Defaults to 30.
            hoa_fee: Monthly homeowners association fee. Defaults to $500.
            insurance_rate: Annual insurance rate as a fraction of home value. Defaults to 0.5% of home value.
            maintenance_rate: Annual maintenance cost as a fraction of home value. Defaults to 1% of home value.
            property_tax_rate: Annual property tax rate. Defaults to 1% of home value.
            mean_appreciation_rate: Expected annual home appreciation rate. Defaults to 3% per year.
            appreciation_volatility: Volatility of the home appreciation rate. Defaults to 5%.
            mean_inflation_rate: Expected annual inflation rate. Defaults to 2% per year.
            inflation_volatility: Volatility of the inflation rate. Defaults to 1%.
            mean_income_change_rate: Expected annual rate of income change. Defaults to 3% per year.
            income_change_volatility: Volatility of annual income changes. Defaults to 3%.
            retirement_contribution_rate: Percentage of income contributed to retirement. Defaults to 15%.
            filing_status: Tax filing status category for U.S. federal income tax. Defaults to 'single'.
            extra_payment: An extra payment added toward principal. Defaults to $0.
            extra_payment_start_year: Year into loan term that extra payments are started. Defaults to the first year.
            extra_payment_end_year: Year into loan term that extra payments are ended. Defaults to year 30.
            purchase_closing_cost_rate: Closing costs for purchase as a percentage of home price. Defaults to 4%.
            sale_closing_cost_rate: Closing costs for sale as a percentage of sale price. Defaults to 7%.
            marriage_probability: Probability of getting married. Defaults to 5%.
            divorce_probability: Probability of getting divorced. Defaults to 2%.
            partner_income_ratio: Ratio of partner's income to the homeowner's income. Defaults to 0.8.
            divorce_cost: Cost of divorce. Defaults to $50,000.
            degrees_of_freedom: The degrees of freedom for the Student's t (for appreciation). Defaults to 5.
            simulations: Number of Monte Carlo simulations to run. Defaults to 5,000.
            rng: Custom random number generator. If None, use default numpy RNG. Defaults to None.
        """
        # Store all input parameters as instance variables
        super().__init__(
            initial_cost=home_price,
            total_years=loan_term,
            mean_inflation_rate=mean_inflation_rate,
            inflation_volatility=inflation_volatility,
            simulations=simulations,
            rng=rng,
        )
        self.home_price = self.initial_cost
        self.loan_term = self.total_years
        self.down_payment = down_payment
        self.mortgage_rate = mortgage_rate
        self.initial_income = initial_income
        self.initial_age = initial_age
        self.hoa_fee = hoa_fee
        self.insurance_rate = insurance_rate
        self.maintenance_rate = maintenance_rate
        self.property_tax_rate = property_tax_rate
        self.mean_appreciation_rate = mean_appreciation_rate
        self.appreciation_volatility = appreciation_volatility
        self.mean_income_change_rate = mean_income_change_rate
        self.income_change_volatility = income_change_volatility
        self.retirement_contribution_rate = retirement_contribution_rate
        self.filing_status = filing_status
        self.extra_payment = extra_payment
        self.extra_payment_start_year = extra_payment_start_year
        self.extra_payment_end_year = extra_payment_end_year
        self.purchase_closing_cost_rate = purchase_closing_cost_rate
        self.sale_closing_cost_rate = sale_closing_cost_rate

        self.tax_calculator = TaxCalculator()

        self.purchase_closing_costs = self.home_price * self.purchase_closing_cost_rate
        self.loan_amount = self.home_price - self.down_payment + self.purchase_closing_costs
        self.monthly_payment = self.calculate_monthly_payment(self.loan_amount, self.mortgage_rate, self.loan_term * 12)

        self.career_simulator = CareerIncomeSimulator(
            initial_income=initial_income,
            total_years=loan_term,
            mean_income_growth=mean_income_change_rate,
            income_volatility=income_change_volatility,
            simulations=simulations,
            rng=rng,
        )

        self.life_simulator = FinancialLifeSimulator(
            career_simulator=self.career_simulator,
            initial_age=initial_age,
            marriage_probability=marriage_probability,
            divorce_probability=divorce_probability,
            partner_income_ratio=partner_income_ratio,
            divorce_cost=divorce_cost,
            initial_marriage_status=filing_status,
            simulations=simulations,
            rng=rng,
        )

        self.appreciation_rate = StudentTRV(
            self.mean_appreciation_rate,
            self.appreciation_volatility,
            shape=degrees_of_freedom,
            rng=self.rng,
        )
        self.inflation_rate = NormalRV(self.mean_inflation_rate, self.inflation_volatility, rng=self.rng)
        self.income_change_rate = NormalRV(self.mean_income_change_rate, self.income_change_volatility, rng=self.rng)
        self.federal_adjustment = NormalRV(0.0, 0.01, rng=self.rng)
        self.state_adjustment = NormalRV(0.0, 0.005, rng=self.rng)
        self.local_adjustment = NormalRV(0.0, 0.002, rng=self.rng)

    def calculate_monthly_payment(self, principal: float, annual_rate: float, months: int) -> float:
        """Calculate the monthly mortgage payment.

        Args:
            principal: The loan principal amount.
            annual_rate: Annual interest rate (as a decimal).
            months: The total number of monthly payments.

        Returns:
            The calculated monthly payment amount.
        """
        monthly_rate = annual_rate / 12.0
        return principal * (monthly_rate * (1.0 + monthly_rate) ** months) / ((1.0 + monthly_rate) ** months - 1.0)

    def _calculate_current_age(self, year: np.ndarray) -> np.ndarray:
        """Calculate the current age based on the simulation year."""
        return self.initial_age + year

    def _simulate_vectorized(self, months: np.ndarray) -> SimulationResults:
        total_months, num_simulations = shape = months.shape
        years = months // 12

        # Initialize arrays
        home_values = np.full(shape, self.home_price)
        incomes = np.full(shape, self.initial_income)
        hoa_fees = np.full(shape, self.hoa_fee)
        remaining_balances = np.full(shape, self.loan_amount)

        # Simulate life events and income
        life_results = self.life_simulator._simulate_vectorized(months)  # noqa: SLF001
        if life_results.household_income is None:
            msg = "Life simulator must return household_income in results."
            raise ValueError(msg)
        if life_results.marital_status is None:
            msg = "Life simulator must return marital_status in results."
            raise ValueError(msg)
        incomes = life_results.household_income
        marital_status = life_results.marital_status

        # Calculate cumulative rates
        # Adjust appreciation rate to account for inflation
        inflation_rates = self.inflation_rate(shape)
        appreciation_rates = self.appreciation_rate(shape)
        real_appreciation_rate = (1.0 + appreciation_rates) * (1.0 + inflation_rates) - 1.0
        cumulative_real_appreciation = np.cumprod(1.0 + real_appreciation_rate / 12.0, axis=0)
        cumulative_inflation = np.cumprod(1.0 + inflation_rates / 12.0, axis=0)

        # Update values over time
        home_values = self.home_price * cumulative_real_appreciation
        hoa_fees = self.hoa_fee * cumulative_inflation

        # Calculate mortgage amortization
        monthly_rate = self.mortgage_rate / 12.0
        payment_periods = np.arange(total_months)[:, np.newaxis]
        interest_payments = remaining_balances * monthly_rate
        principal_payments = self.monthly_payment - interest_payments

        # Apply extra payments
        extra_payment_mask = (years >= self.extra_payment_start_year - 1) & (years < self.extra_payment_end_year)
        extra_payments = np.where(extra_payment_mask, self.extra_payment / 12.0, 0.0)
        principal_payments += np.minimum(extra_payments, remaining_balances)

        # Update remaining balances
        remaining_balances = np.maximum(
            self.loan_amount * (1.0 + monthly_rate) ** payment_periods
            - self.monthly_payment * ((1.0 + monthly_rate) ** payment_periods - 1) / monthly_rate
            - np.cumsum(extra_payments, axis=0),
            0.0,
        )

        # Update remaining balances
        remaining_balances -= principal_payments
        remaining_balances = np.maximum(remaining_balances, 0.0)

        # Recalculate actual principal and interest payments
        actual_total_payments = np.diff(remaining_balances, axis=0, prepend=self.loan_amount)
        actual_principal_payments = -actual_total_payments
        actual_interest_payments = np.minimum(remaining_balances * monthly_rate, self.monthly_payment)

        # Calculate monthly costs
        property_taxes = home_values * (self.property_tax_rate / 12.0)
        insurance = home_values * (self.insurance_rate / 12.0)
        maintenance = home_values * (self.maintenance_rate / 12.0)

        # Calculate tax rates
        effective_tax_rates = self.tax_calculator.calculate_effective_tax_rates(
            incomes=incomes,
            filing_status=marital_status,
            federal_adj=self.federal_adjustment(shape),
            state_adj=self.state_adjustment(shape),
            local_adj=self.local_adjustment(shape),
            inflation_rates=inflation_rates,
        )

        # Calculate ages for each month
        ages = np.broadcast_to(self._calculate_current_age(years), shape)

        # Calculate tax deductions
        tax_deductions = self.tax_calculator.calculate_tax_deduction(
            ages=ages,
            mortgage_rates=np.full(shape, self.mortgage_rate),
            mortgage_interests=actual_interest_payments * 12,  # Convert to annual
            property_taxes=property_taxes * 12,  # Convert to annual
            federal_rates=effective_tax_rates.federal_rate,
            state_rates=effective_tax_rates.state_rate,
            local_rates=effective_tax_rates.local_rate,
            retirement_contributions=incomes * self.retirement_contribution_rate,
            annual_incomes=incomes,
            filing_status=marital_status,
        )

        # Convert annual tax deductions to monthly
        monthly_tax_savings = tax_deductions / 12

        monthly_costs = (
            actual_principal_payments
            + actual_interest_payments
            + property_taxes
            + insurance
            + maintenance
            + hoa_fees
            - monthly_tax_savings
        )

        # Calculate cumulative costs
        cumulative_costs = np.cumsum(monthly_costs, axis=0)

        # Add initial costs to cumulative costs
        cumulative_costs += self.purchase_closing_costs + self.down_payment

        # Calculate final sale costs and profit/loss
        sale_closing_costs = home_values * self.sale_closing_cost_rate
        profit_loss = home_values - self.home_price - cumulative_costs - sale_closing_costs

        return SimulationResults(
            monthly_costs=monthly_costs,
            profit_loss=profit_loss,
            simulations=num_simulations,
            total_years=self.total_years,
            home_values=home_values,
            remaining_mortgage_balance=remaining_balances,
            property_taxes=property_taxes,
            insurance_costs=insurance,
            maintenance_costs=maintenance,
            household_income=incomes,
            monthly_income=life_results.monthly_income,
            partner_income=life_results.partner_income,
            personal_income=life_results.personal_income,
            marital_status=marital_status,
            tax_deductions=tax_deductions,
            federal_effective_tax_rate=effective_tax_rates.federal_rate,
            state_effective_tax_rate=effective_tax_rates.state_rate,
            local_effective_tax_rate=effective_tax_rates.local_rate,
            cumulative_costs=cumulative_costs,
            promotions=life_results.promotions,
            demotions=life_results.demotions,
            layoffs=life_results.layoffs,
            extra={
                "hoa_fees": hoa_fees,
                "principal_payments": actual_principal_payments,
                "interest_payments": actual_interest_payments,
                "sale_closing_costs": sale_closing_costs,
                "cumulative_real_appreciation": cumulative_real_appreciation,
                "cumulative_inflation": cumulative_inflation,
                "layoff_durations": life_results.extra["layoff_durations"],
                "layoff_mask": life_results.extra["layoff_mask"],
                "divorce_costs": life_results.extra["divorce_costs"],
            },
        )

    def _get_input_parameters(self) -> list[tuple[str, str]]:
        return [
            ("Home Price", f"${self.home_price:,.2f}"),
            ("Down Payment", f"${self.down_payment:,.2f}"),
            ("Mortgage Rate", f"{self.mortgage_rate:.2%}"),
            ("Loan Term", f"{self.loan_term} years"),
            ("Initial Annual Income", f"${self.initial_income:,.2f}"),
            ("Monthly HOA Fee", f"${self.hoa_fee:,.2f}"),
            ("Insurance Rate", f"{self.insurance_rate:.2%}"),
            ("Maintenance Rate", f"{self.maintenance_rate:.2%}"),
            ("Property Tax Rate", f"{self.property_tax_rate:.2%}"),
            ("Mean Appreciation Rate", f"{self.mean_appreciation_rate:.2%}"),
            ("Appreciation Volatility", f"{self.appreciation_volatility:.2%}"),
            ("Mean Inflation Rate", f"{self.mean_inflation_rate:.2%}"),
            ("Inflation Volatility", f"{self.inflation_volatility:.2%}"),
            ("Mean Income Change Rate", f"{self.mean_income_change_rate:.2%}"),
            ("Income Change Volatility", f"{self.income_change_volatility:.2%}"),
            ("Extra Payment", f"${self.extra_payment:,.2f}"),
            ("Extra Payment Start Year", f"{self.extra_payment_start_year}"),
            ("Extra Payment End Year", f"{self.extra_payment_end_year}"),
            ("Purchase Closing Cost Rate", f"{self.purchase_closing_cost_rate:.2%}"),
            ("Sale Closing Cost Rate", f"{self.sale_closing_cost_rate:.2%}"),
            ("Number of Simulations", f"{self.simulations}"),
        ]
