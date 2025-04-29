# router.py
from flask import Flask, render_template, request, jsonify
from utils import predict_pressure

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    user_data = {
        'Level_of_Hemoglobin': request.form['hemoglobin'],
        'Age': request.form['age'],
        'BMI': request.form['bmi'],
        'Gender': request.form['gender'],
        'Smoking': request.form['smoking'],
        'Physical_activity': request.form['activity'],
        'salt_content_in_the_diet': request.form['salt'],
        'Chronic_kidney_disease': request.form['kidney'],
        'Adrenal_and_thyroid_disorders': request.form['disorders']
    }

    try:
        prediction = predict_pressure(user_data)
        result = "The person has pressure" if prediction == 1 else "The person does not have pressure"
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
