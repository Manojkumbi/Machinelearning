import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('heart.csv')


X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

joblib.dump(scaler, 'scaler.pkl')
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)

joblib.dump(model, 'logistic_model.pkl')
