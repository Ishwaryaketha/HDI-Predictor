import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("hdi.csv")

# Keep only required columns
df = df[[
    "Life Expectancy at Birth",
    "Expected Years of Education",
    "Mean Years of Education",
    "Gross National Income (GNI) per Capita",
    "Human Development Index (HDI)"
]]

# Remove commas from GNI column and convert to number
df["Gross National Income (GNI) per Capita"] = (
    df["Gross National Income (GNI) per Capita"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Gross National Income (GNI) per Capita"] = pd.to_numeric(
    df["Gross National Income (GNI) per Capita"],
    errors="coerce"
)

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors="coerce")

# Remove rows with missing values
df = df.dropna()

# Features
X = df[[
    "Life Expectancy at Birth",
    "Expected Years of Education",
    "Mean Years of Education",
    "Gross National Income (GNI) per Capita"
]]

# Target
y = df["Human Development Index (HDI)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
print("Model saved successfully!")