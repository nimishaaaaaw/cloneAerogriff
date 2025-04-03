import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Define the dataset filename
filename = "drone_crash_data.csv"

# Check if the dataset exists, if not, generate it
if not os.path.exists(filename):
    print(f"{filename} not found. Please generate the dataset first.")
else:
    # Step 1: Load Data from CSV
    df = pd.read_csv(filename)
    print(f"Dataset loaded from {filename}")

    # Step 2: Split data into features (X) and target (y)
    X = df.drop(columns=["Crash"])  # Features
    y = df["Crash"]  # Target variable

    # Step 3: Train-Test Split (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Train the Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 5: Predict on Test Data
    y_pred = model.predict(X_test)

    # Step 6: Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
