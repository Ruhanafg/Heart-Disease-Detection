from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # collect data from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    
    prediction = model.predict(final_features)[0]
    
    if prediction == 1:
        result = "The patient is likely to have heart disease."
    else:
        result = "The patient is unlikely to have heart disease."
    
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
