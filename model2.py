import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the provided dataset (real data)
df_real = pd.read_csv('fin.csv')

# Feature Engineering
# Creating new features from existing ones
df_real['Altitude_Diff'] = df_real['Altitude(meters)'] - df_real['VpsAltitude(meters)']
df_real['Speed_Combined'] = (df_real['HSpeed(m/s)'] + df_real['GpsSpeed(m/s)']) / 2
df_real['Risky_Altitude'] = np.where(df_real['Altitude(meters)'] < 20, 1, 0)
df_real['Risky_Speed'] = np.where(df_real['Speed_Combined'] > 30, 1, 0)
df_real['Risky_Home_Distance'] = np.where(df_real['HomeDistance(feet)'] > 500, 1, 0)

# Generate a 'Crash' label based on extreme conditions
df_real['Crash'] = np.where(
    (df_real['BatteryPower(%)'] < 20) |    
    (df_real['Altitude_Diff'] > 50) | 
    (df_real['Risky_Speed'] == 1) | 
    (df_real['Risky_Home_Distance'] == 1), 1, 0
)

# Drop non-useful columns
X = df_real.drop(columns=['Time(text)', 'FlightMode', 'GpsSpeed(mph)', 'AppTip', 'AppWarning', 'AppMessage'])
y = df_real['Crash']

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear')
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{name} Accuracy: {accuracy * 100:.2f}%')
