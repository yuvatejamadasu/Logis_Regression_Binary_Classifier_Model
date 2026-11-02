
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ==========================================
# LOAD THE TRAINED MODEL
# ==========================================

MODEL_PATH = "Logistic_Regression_Binary_Breast_Cancer.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ==========================================
# FEATURE NAMES
# ==========================================

# Use the exact feature names from the trained model.
# This prevents feature-name and feature-order mismatches.

if hasattr(model, "feature_names_in_"):

    FEATURES = list(model.feature_names_in_)

else:

    FEATURES = [
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",

        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",

        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst"
    ]


# ==========================================
# FEATURE LABELS
# ==========================================

def get_label(feature):

    label = feature.replace("_", " ")

    label = label.replace("concave points", "concave points")

    label = label.replace("fractal dimension", "fractal dimension")

    return label.title()


FEATURE_LABELS = {
    feature: get_label(feature)
    for feature in FEATURES
}


# ==========================================
# COMMON TEMPLATE DATA
# ==========================================

def template_data():

    return {
        "features": FEATURES,
        "labels": FEATURE_LABELS
    }


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        **template_data(),
        values={},
        result=None,
        error=None
    )


# ==========================================
# PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    values = {}

    try:

        # ----------------------------------
        # GET ALL 30 INPUT VALUES
        # ----------------------------------

        input_values = []

        for feature in FEATURES:

            raw_value = request.form.get(feature, "").strip()

            values[feature] = raw_value

            if raw_value == "":

                raise ValueError(
                    f"Please enter a value for {FEATURE_LABELS[feature]}."
                )

            try:

                number = float(raw_value)

            except (TypeError, ValueError):

                raise ValueError(
                    f"{FEATURE_LABELS[feature]} must be a numeric value."
                )

            if not np.isfinite(number):

                raise ValueError(
                    f"{FEATURE_LABELS[feature]} must be a valid number."
                )

            if number < 0:

                raise ValueError(
                    f"{FEATURE_LABELS[feature]} cannot be negative."
                )

            input_values.append(number)


        # ----------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------

        input_data = pd.DataFrame(
            [input_values],
            columns=FEATURES
        )


        # ----------------------------------
        # MODEL PREDICTION
        # ----------------------------------

        prediction = int(model.predict(input_data)[0])

        # Your training mapping:
        # M = 0
        # B = 1

        if prediction == 0:

            result = "Malignant"

            result_class = "malignant"

            description = (
                "The model predicts a malignant classification. "
                "Please consult a qualified medical professional "
                "for further evaluation."
            )

        else:

            result = "Benign"

            result_class = "benign"

            description = (
                "The model predicts a benign classification. "
                "This result does not replace professional "
                "medical examination."
            )


        # ----------------------------------
        # PREDICTION PROBABILITY
        # ----------------------------------

        probabilities = model.predict_proba(input_data)[0]

        class_index = list(model.classes_).index(prediction)

        confidence = round(
            float(probabilities[class_index]) * 100,
            2
        )


        # ----------------------------------
        # DISPLAY RESULT
        # ----------------------------------

        return render_template(
            "index.html",

            **template_data(),

            values=values,

            result=result,

            result_class=result_class,

            description=description,

            confidence=confidence,

            error=None
        )


    except ValueError as e:

        return render_template(
            "index.html",

            **template_data(),

            values=values,

            result=None,

            error=str(e)
        )


    except Exception as e:

        app.logger.exception("Prediction error")

        return render_template(
            "index.html",

            **template_data(),

            values=values,

            result=None,

            error=(
                "Prediction failed. Please check your model file "
                "and input features."
            )
        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )

