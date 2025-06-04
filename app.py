from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            features = [
                float(request.form['bedrooms']),
                float(request.form['bathrooms']),
                float(request.form['sqft_living']),
                float(request.form['sqft_lot']),
                float(request.form['floors']),
                float(request.form['waterfront']),
                float(request.form['view']),
                float(request.form['condition']),
                float(request.form['grade']),
                float(request.form['sqft_above']),
                float(request.form['sqft_basement']),
                float(request.form['yr_built']),
                float(request.form['yr_renovated']),
                float(request.form['zipcode']),
                float(request.form['lat']),
                float(request.form['long']),
                float(request.form['sqft_living15']),
                float(request.form['sqft_lot15'])
            ]
            input_df = pd.DataFrame([features])
            scaled = scaler.transform(input_df)
            prediction = round(model.predict(scaled)[0], 2)
        except:
            prediction = "Invalid input!"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
