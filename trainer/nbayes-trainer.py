import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib

data = pd.read_csv('heart.csv')

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train, y_train)

accuracy = gaussian_nb.score(X_test, y_test)
print("Test Set Accuracy:", accuracy)

joblib.dump(gaussian_nb, 'gaussian-nb-model.pkl')

print("Gaussian Naive Bayes model saved to 'gaussian-nb.pkl'")
