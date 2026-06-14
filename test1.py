import pandas as pd

data = pd.read_csv("predictive_maintenance.csv")

X = data[['Air temperature [K]',
          'Process temperature [K]',
          'Rotational speed [rpm]',
          'Torque [Nm]',
          'Tool wear [min]']]

y = data['Target']

print(X.head())
print(y.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("Model trained successfully!")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
import pickle

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")
