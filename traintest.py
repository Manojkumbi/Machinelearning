import joblib
import numpy as np
import pandas as pd

# Load the corrected scaler and model
scaler = joblib.load('model/scaler.pkl')  # Use the correct scaler file
model = joblib.load('model/logistic-model.pkl')  # Load the pre-trained model

# Sample data for prediction
data = [
    [66, 1, 3, 143, 223, 1, 1, 128, 0, 3.3, 0, 0, 1],  # Example row
    # Add other rows if needed...
]

# Convert to DataFrame and scale the features
df = pd.DataFrame(data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
scaled_features = scaler.transform(df)

# Predict with the model
predictions = model.predict(scaled_features)

# Determine risk based on predictions
results = ['High risk of heart attack' if pred == 1 else 'Low risk of heart attack' for pred in predictions]

# Print the results
print(results)
