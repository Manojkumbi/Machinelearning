import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Create some sample training data (for demonstration purposes)
train_data =  pd.read_csv('trainer/heart.csv')

# Fit a new StandardScaler
scaler = StandardScaler()

# Fit the scaler with the training data
scaler.fit(train_data)

# Save the fitted scaler to a new file
joblib.dump(scaler, 'scaler.pkl')

print("Scaler has been fitted and saved.")
