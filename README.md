# NYC Housing Cost Calculator

A Python package for simulating long-term costs of buying and renting in New York City.

## Features

- Calculates homeownership and rental costs over time
- Uses Monte Carlo simulation for market uncertainty
- Considers taxes, property appreciation, and income changes
- Visualizes cost projections
- Provides statistical summaries

## Install

```bash
pip install nyc-housing-cost-calculator
```

## Usage

### Homeownership Costs

```python
from nyc_housing_cost_calculator import NYCHomeCostCalculator, FilingStatus

calc = NYCHomeCostCalculator(
    home_price=1_000_000.0,
    down_payment=200_000.0,
    mortgage_rate=0.03,
    loan_term=30,
    initial_income=150_000.0,
    hoa_fee=500,
    filing_status=FilingStatus.MARRIED_JOINT,
    simulations=10_000,
)

print(f"Home Ownership Cost Statistics: {calc.get_cost_statistics()}")
calc.plot_costs_over_time()
```

### Rental Costs

```python
from nyc_housing_cost_calculator import NYCRentalCostCalculator

calc = NYCRentalCostCalculator(
    initial_rent=4_000,
    lease_term=1,
    total_years=30,
    initial_income=150_000,
    utility_cost=200,
    renters_insurance=300,
    simulations=10_000,
)

print(f"Rental Cost Statistics: {calc.get_cost_statistics()}")
calc.plot_costs_over_time()
```

This tool provides estimates. Consult financial advisors for important housing decisions.
