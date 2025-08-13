from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load K-Means model
with open("model.pkl", "rb") as f:
    kmeans = pickle.load(f)

# Load cluster descriptions
with open("cluster_descriptions.pkl", "rb") as f:
    cluster_descriptions = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        gender = request.form['gender']   # Male/Female
        age = float(request.form['age'])
        income = float(request.form['income'])
        spending_score = float(request.form['spending'])

        # Encode gender manually (must match training encoding)
        gender_encoded = 1 if gender.lower() == "male" else 0

        # Prepare data for prediction
        features = np.array([[gender_encoded, age, income, spending_score]])
        cluster = kmeans.predict(features)[0]

        # Get description
        meaning = cluster_descriptions[cluster]

        return jsonify({
            "cluster": int(cluster),
            "meaning": meaning
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
