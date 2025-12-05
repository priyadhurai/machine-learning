from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

data = {
    "Age": [25, 32, 47, 51, 62],
    "Salary": [40, 60, 85, 95, 110],
    "Bought": [0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df[["Age", "Salary"]]
y = df["Bought"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Accuracy:", accuracy_score(y_test, predictions))
