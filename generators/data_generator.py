import numpy as np
import pandas as pd

np.random.seed(42)

n_neighborhoods = 200
years = [2022, 2023, 2024]

data = []

for year in years:
    for i in range(n_neighborhoods):
        median_income = np.random.normal(60000, 15000)
        poverty_rate = np.clip(np.random.normal(0.15, 0.07), 0.02, 0.40)
        age_65_plus = np.clip(np.random.normal(0.16, 0.05), 0.05, 0.30)
        unemployment = np.clip(np.random.normal(0.06, 0.03), 0.01, 0.15)
        pm25 = np.random.normal(10, 3)
        food_access = np.clip(np.random.normal(0.6, 0.2), 0.1, 1.0)

        # Create synthetic diabetes prevalence relationship
        diabetes_prev = (
            0.05
            + 0.4 * poverty_rate
            + 0.3 * age_65_plus
            + 0.2 * unemployment
            + 0.01 * pm25
            - 0.000001 * median_income
            - 0.1 * food_access
        )

        diabetes_prev = np.clip(diabetes_prev, 0.03, 0.30)

        data.append([
            f"N{i+1}",
            year,
            diabetes_prev,
            median_income,
            poverty_rate,
            age_65_plus,
            unemployment,
            pm25,
            food_access
        ])

columns = [
    "neighborhood_id",
    "year",
    "diabetes_prev",
    "median_income",
    "poverty_rate",
    "age_65_plus",
    "unemployment_rate",
    "pm25",
    "food_access_index"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("synthetic_diabetes_data.csv", index=False)

print("Synthetic dataset created!")