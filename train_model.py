import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("yellow_tripdata_2019-01.csv", nrows=100_000)

# Clean and preprocess
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 200)]
df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]

df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['hour'] = df['tpep_pickup_datetime'].dt.hour

# Select features and target
features = ['trip_distance', 'passenger_count', 'hour']
X = df[features]
y = df['fare_amount']

# Handle missing or invalid values
X = X.fillna(0)
y = y.fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'fare_model.pkl')

print("Model trained and saved to 'fare_model.pkl'")
