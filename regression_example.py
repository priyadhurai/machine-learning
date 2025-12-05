from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Sample data
data = {
    "Area": [1000, 1500, 2000, 2500, 3000],
    "Price": [50, 75, 100, 130, 165]
}

df = pd.DataFrame(data)

X = df[["Area"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("MSE:", mean_squared_error(y_test, predictions))
