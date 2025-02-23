from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('your_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    
    # Convert gender to numerical value
    gender = request.form['Gender']
    gender_numeric = 1 if gender.lower() == 'male' else 0
    age = float(request.form['Age'])
    
    height = float(request.form['Height'])
    weight = float(request.form['Weight'])
    duration = float(request.form['Duration'])
    heart_rate = float(request.form['Heart_Rate'])
    body_temp = float(request.form['Body_Temp'])

    # Prepare input features
    features = np.array([[gender_numeric, age, height, weight, duration, heart_rate, body_temp]])

    # Make prediction
    prediction = model.predict(features)
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

