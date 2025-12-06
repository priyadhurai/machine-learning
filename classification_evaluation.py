import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# Sample dataset
data = {
    'Age': [25, 32, 47, 51, 62, 23, 44, 36],
    'Income': [50000, 60000, 80000, 120000, 150000, 45000, 76000, 64000],
    'Default': [0, 0, 1, 0, 1, 0, 1, 0]  # Target
}

df = pd.DataFrame(data)

# 1. Split the data
X = df[['Age', 'Income']]
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 2. Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 3. Predict
y_pred = model.predict(X_test)

# 4. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
