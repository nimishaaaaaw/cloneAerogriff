import pandas as pd
import numpy as np

# Number of rows
num_rows = 10000  

# Generating data with realistic ranges and noise
data = {
    "Altitude": np.random.randint(10, 300, num_rows),  # Meters
    "Speed": np.random.randint(1, 50, num_rows),  # m/s
    "Battery": np.random.randint(1, 100, num_rows),  # Percentage
    "Wind Speed": np.random.randint(0, 25, num_rows),  # m/s
    "GPS Signal": np.random.randint(-100, -40, num_rows),  # Signal strength dBm
    "Obstacle Distance": np.random.randint(1, 50, num_rows),  # Meters
    "Motor Health": np.random.randint(50, 100, num_rows),  # 50-100% health
    "IMU Stability": np.random.uniform(0, 1, num_rows),  # 0 to 1 (higher = better stability)
    "Temperature": np.random.randint(10, 60, num_rows),  # Celsius
}

df = pd.DataFrame(data)

# Simulating "Crash" conditions with noise
df["Crash"] = np.where(
    (df["Battery"] < 20) & (df["Wind Speed"] > 15) & (df["Obstacle Distance"] < 5), 1, 0
)

# Introduce some randomness (5% of data gets a random crash)
random_crash_indices = np.random.choice(df.index, size=int(num_rows * 0.05), replace=False)
df.loc[random_crash_indices, "Crash"] = 1  

# Save to CSV
df.to_csv("drone_crash_data.csv", index=False)
print("Dataset saved as drone_crash_data.csv")
