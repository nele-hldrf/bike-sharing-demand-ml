from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

# Feature-Namen laden, falls vorhanden
try:
    feature_names = joblib.load("feature_names.pkl")
except Exception:
    feature_names = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    #Minimal Check
    if not data or "features" not in data:
        return jsonify({"error": "Bitte JSON mit Key 'features' senden"}), 400
    
    f = data["features"]

    season = f["Season"]  # Wert aus dem Dropdown

    season_spring = 1 if season == "Spring" else 0
    season_summer = 1 if season == "Summer" else 0
    season_winter = 1 if season == "Winter" else 0
    # Autumn = 0,0,0 => passt, weil nicht im Training

    # ---- FEATURE REIHENFOLGE EXACT WIE BEIM TRAINING ----
    X_list = [
        f["Hour"],
        f["Temperature"],
        f["Humidity"],
        f["Wind_speed"],
        f["Visibility"],
        f["Dew_point_temperature"],
        f["Solar_Radiation"],
        f["Rainfall"],
        f["Snowfall"],
        season_spring,
        season_summer,
        season_winter,
        f["Holiday_No_Holiday"]
    ]
    
    # In numpy-Array und auf 2D-Form bringen: shape (1, 4)
    X = np.array(X_list).reshape(1,-1)

    # Vorhersage und Wahrscheinlichkeiten vom Modell holen
    pred = float(model.predict(X) [0])

    response = {
        "prediction": pred,
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
