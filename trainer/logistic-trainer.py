import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib

data = pd.read_csv('heart.csv')

# Check class distribution
print("Class distribution:\n", data['target'].value_counts())

X = data.drop(columns=['target'])
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
joblib.dump(scaler, 'scaler.pkl')

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(solver='liblinear', C=1.0)
model.fit(X_train_scaled, y_train)

# Evaluate the model
accuracy = model.score(X_test_scaled, y_test)
confusion = confusion_matrix(y_test, model.predict(X_test_scaled))
report = classification_report(y_test, model.predict(X_test_scaled))

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)

# Save the model
joblib.dump(model, 'logistic-model.pkl')
