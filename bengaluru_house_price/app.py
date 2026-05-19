from flask import Flask, request, jsonify, render_template
import pickle, json, numpy as np

app = Flask(__name__)

# Load model, encoder, locations
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
locations = json.load(open("locations.json"))

@app.route("/")
def home():
    return render_template("index.html", locations=locations)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    location = data["location"]
    sqft = float(data["sqft"])
    bath = float(data["bath"])
    balcony = float(data["balcony"])
    bhk = float(data["bhk"])

    # Encode location
    if location in encoder.classes_:
        loc_enc = encoder.transform([location])[0]
    else:
        loc_enc = encoder.transform(["Other"])[0]

    features = np.array([[sqft, bath, balcony, bhk, loc_enc]])
    price = model.predict(features)[0]
    price = max(price, 5)  # minimum 5 lakhs

    return jsonify({"price": round(price, 2)})

if __name__ == "__main__":
    app.run(debug=True)
