import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv('heart.csv')

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

joblib.dump(decision_tree, 'decision-tree-model.pkl')

print("Decision Tree model saved to 'decision-tree.pkl'")
