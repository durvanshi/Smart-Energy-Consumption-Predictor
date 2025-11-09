
# ----------------------------------------
# Simple Smart Energy Consumption Predictor
# ----------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

print("? Smart Energy Consumption Predictor ?")

# ----------------------------------------
# Step 1: Load Example Data
# ----------------------------------------
# Sample dataset of daily energy usage (in kWh)
data = {
    "Day": [1, 2, 3, 4, 5, 6, 7],
    "EnergyUsed": [20, 22, 25, 23, 30, 28, 35]
}

df = pd.DataFrame(data)
print("\nSample Energy Data:\n", df)

# ----------------------------------------
# Step 2: Train a Simple Linear Regression Model
# ----------------------------------------
X = df[["Day"]]          # input (days)
y = df["EnergyUsed"]      # output (energy used)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------
# Step 3: Test the Model
# ----------------------------------------
y_pred = model.predict(X_test)

print("\nPredicted Energy Usage:", y_pred)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# ----------------------------------------
# Step 4: Predict Future Usage
# ----------------------------------------
future_day = int(input("\nEnter day number to predict energy usage: "))
predicted = model.predict([[future_day]])
print(f"?? Predicted Energy Consumption for Day {future_day}: {predicted[0]:.2f} kWh")
