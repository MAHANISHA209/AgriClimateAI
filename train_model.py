import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Read the CSV file
X = []
y = []

with open("Crop_recommendation.csv", "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        X.append([
            float(row["N"]),
            float(row["P"]),
            float(row["K"]),
            float(row["temperature"]),
            float(row["humidity"]),
            float(row["ph"]),
            float(row["rainfall"])
        ])

        y.append(row["label"])

# Convert input data into NumPy array
X = np.array(X)

print("Dataset loaded successfully!")
print("Number of rows:", len(X))
print("Number of features:", X.shape[1])

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Test the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)

# 6. Save the trained model
joblib.dump(model, "crop_model.pkl")

print("Model saved successfully as crop_model.pkl")