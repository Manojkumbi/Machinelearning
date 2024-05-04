from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        model_name = request.form['model']
        model_path = f'model/{model_name}-model.pkl'
        model = joblib.load(model_path)
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_features)

        if prediction[0] == 1:
            result = 'High risk of heart attack'
        else:
            result = 'Low risk of heart attack'

        return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
