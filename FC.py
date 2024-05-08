import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Import Data
df = pd.read_csv('FuelConsumption.csv')

# Convert Categorical to Numerical
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))

# Split Data
X = df.drop('FUEL CONSUMPTION', axis=1)
y = df['FUEL CONSUMPTION']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Model
reg = LinearRegression().fit(X_train, y_train)

# Score
print("Score:", reg.score(X_test, y_test))

# Predictions and Plot
y_pred = reg.predict(X_test)
plt.scatter(y_test, y_pred, color="black")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="blue", linewidth=3)
plt.xlabel('Actual Fuel Consumption')
plt.ylabel('Predicted Fuel Consumption')
plt.title('Linear Regression Prediction')
plt.show()
