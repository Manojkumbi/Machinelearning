import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

# Generate data with correlations
def generate_data(num_samples):
    data = []

    for _ in range(num_samples):
        age = random.randint(30, 80)
        sex = random.choice([0, 1])
        cp = random.randint(0, 3)
        trestbps = random.randint(90, 180)
        chol = random.randint(150, 350)

        # Create some correlations
        if age > 60:
            trestbps = random.randint(130, 180)
        if chol > 300:
            cp = random.randint(2, 3)

        fbs = random.choice([0, 1])
        restecg = random.randint(0, 2)

        # Thalach (maximum heart rate) tends to decrease with age
        thalach = max(150 - (age // 3), 100)

        exang = random.choice([0, 1])
        oldpeak = round(random.uniform(0.0, 6.0), 1)
        slope = random.randint(0, 2)
        ca = random.randint(0, 3)
        thal = random.randint(1, 3)

        # Define a simple logic to create a meaningful target
        if (cp == 3 and trestbps > 130) or (age > 60 and thalach < 140):
            target = 1 if random.random() > 0.2 else 0
        else:
            target = 0 if random.random() > 0.2 else 1

        data.append({
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "target": target
        })
    
    return pd.DataFrame(data)

# Generate 10,000 samples
generated_data = generate_data(10000)

# Check feature distributions and correlations
original_data = pd.read_csv("heart.csv")  # Load the original data
generated_data.describe()  # Compare feature distributions
original_data.describe()

# Save the generated data to a CSV file
generated_data.to_csv("generated_heart.csv", index=False)
