from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_data = {
                "Age": int(request.form["age"]),
                "Sex": request.form["sex"],
                "ChestPainType": request.form["chest_pain"],
                "RestingBP": int(request.form["resting_bp"]),
                "Cholesterol": int(request.form["cholesterol"]),
                "FastingBS": int(request.form["fasting_bs"]),
                "RestingECG": request.form["resting_ecg"],
                "MaxHR": int(request.form["max_hr"]),
                "ExerciseAngina": request.form["exercise_angina"],
                "Oldpeak": float(request.form["oldpeak"]),
                "ST_Slope": request.form["st_slope"]
            }

            df = pd.DataFrame([input_data])
            prediction = model.predict(df)[0]
            result_text = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
            return render_template("result.html", result=result_text)

        except Exception as e:
            return render_template("index.html", result="Invalid input. Please check and try again.")

    return render_template("index.html", result=None)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
