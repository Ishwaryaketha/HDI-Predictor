from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    life = float(request.form["life"])
    expected = float(request.form["expected"])
    mean = float(request.form["mean"])
    gni = float(request.form["gni"])

    features = np.array([[life, expected, mean, gni]])
    prediction = model.predict(features)[0]

    if prediction >= 0.800:
        category = "Very High Human Development"
    elif prediction >= 0.700:
        category = "High Human Development"
    elif prediction >= 0.550:
        category = "Medium Human Development"
    else:
        category = "Low Human Development"

    return render_template(
        "result.html",
        prediction=round(prediction, 3),
        category=category
    )

if __name__ == "__main__":
    app.run(debug=True)