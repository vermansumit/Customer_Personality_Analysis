from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/random_forest_pipeline.joblib")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.json  # expect singhl record as JSON dict
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = float(model.predict_proba(df)[:,1])
    return jsonify({"prediction":int(pred), "probability":prob})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)