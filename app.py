from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
 
model = pickle.load(open("house_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = int(request.form['mainroad'])
    guestroom = int(request.form['guestroom'])
    basement = int(request.form['basement'])
    hotwaterheating = int(request.form['hotwaterheating'])
    airconditioning = int(request.form['airconditioning'])
    parking = int(request.form['parking'])
    prefarea = int(request.form['prefarea'])
    furnishingstatus = int(request.form['furnishingstatus'])

    features = np.array([[area, bedrooms, bathrooms, stories,
                          mainroad, guestroom, basement,
                          hotwaterheating, airconditioning,
                          parking, prefarea, furnishingstatus]])

    prediction = model.predict(features)

    return render_template('index.html',
                           prediction_text="Predicted House Price = {}".format(int(prediction[0])))

if __name__ == "__main__":
    app.run(debug=True)