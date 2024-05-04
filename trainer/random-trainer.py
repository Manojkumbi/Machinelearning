import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv('heart.csv')

X = data.drop(columns=['target']) 
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42) 
random_forest.fit(X_train, y_train)

accuracy = random_forest.score(X_test, y_test)  # This calculates accuracy on the test set
print("Accuracy:", accuracy)

joblib.dump(random_forest, 'random-forest.pkl')
