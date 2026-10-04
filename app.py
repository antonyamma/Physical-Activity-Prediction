import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the exported decision tree model
model_path = "decision_tree_model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON input data
        data = request.get_json(force=True)
        
        # Convert input into a pandas DataFrame (handles single or multiple rows)
        # Expected structure: {"features": [{col1: val1, col2: val2, ...}]}
        features_list = data["features"]
        input_df = pd.DataFrame(features_list)
        
        # Generate prediction
        predictions = model.predict(input_df)
        
        # Return predictions as JSON
        return jsonify({"predictions": predictions.tolist()})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/", methods=["GET"])
def home():
    return "Decision Tree Model Flask API is Running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
