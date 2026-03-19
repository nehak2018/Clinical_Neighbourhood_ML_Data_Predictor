import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# STEP 1: Load
df = pd.read_csv("synthetic_diabetes_data.csv")

# STEP 2: Filter year
df_current = df[df['year'] == 2022]

# STEP 3: Define features
x = df_current[['median_income',
                'poverty_rate',
                'age_65_plus',
                'unemployment_rate',
                'pm25',
                'food_access_index']]

y = df_current['diabetes_prev']

# STEP 4: Split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# STEP 5: Train
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(x, y)
y_pred = model.predict(x)

results = x.copy()
results['Actual'] = y.values
results['Predicted'] = y_pred

print(results.head())



'''
model.fit(X_train, y_train)

# STEP 6: Evaluate
y_pred = model.predict(X_test)

results = X_test.copy()
results['Actual'] = y_test.values
results['Predicted'] = y_pred

print(results.head())


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R2:", r2)
'''