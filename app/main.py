from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular
import traceback

# ==== Load artifacts ====
with open("sales_classifier.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["model"]
scaler = artifacts["scaler"]
features = artifacts["features"]

# Wrapper predict function for Anchor (convert class labels -> index)
def predict_fn(x):
    preds = model.predict(x)
    return np.array([np.where(model.classes_ == p)[0][0] for p in preds])

# ==== Prepare explainers ====
background = np.random.randn(100, len(features))  # NOTE: ideally pakai data training asli

lime_explainer = LimeTabularExplainer(
    training_data=background,
    feature_names=features,
    class_names=["Low", "Mid", "High"],
    mode="classification"
)

anchor_explainer = AnchorTabular(predict_fn, feature_names=features)
anchor_explainer.fit(background, disc_perc=[25, 50, 75])

# ==== FastAPI App ====
app = FastAPI(title="Sales Performance XAI API")

# ==== Request Schema ====
class SalesInput(BaseModel):
    salesname: str
    month: str
    year: str
    performance: dict

# ==== Helper: kategorisasi Z-score ====
def categorize_value(z):
    if z < -0.5:
        return "rendah"
    elif z > 0.5:
        return "tinggi"
    else:
        return "sedang"

# ==== Generate Narrative Text ====
def generate_explanation_text(salesname, month, year, prediction, shap_res, lime_res, anchor_res, X_scaled):
    texts = []

    # Mapping fitur -> kategori
    feature_categories = {f: categorize_value(val) for f, val in zip(features, X_scaled[0])}

    # SHAP narrative
    top_shap = sorted(
        shap_res["contributions"],
        key=lambda x: abs(x["shap_value"]),
        reverse=True
    )[:3]
    positive = [c["feature"] for c in top_shap if c["shap_value"] > 0]
    negative = [c["feature"] for c in top_shap if c["shap_value"] < 0]

    shap_txt = f"Model SHAP menilai {salesname} pada periode {month} {year} menunjukan {prediction} performance."
    if positive or negative:
        shap_txt += " Indikator utama: "
        if positive:
            shap_txt += ", ".join([f"{p} yang {feature_categories[p]}" for p in positive])
        if negative:
            if positive:
                shap_txt += ", sedangkan "
            shap_txt += ", ".join([f"{n} yang {feature_categories[n]}" for n in negative])
    texts.append(shap_txt + ".")

    # LIME narrative
    if lime_res:
        lime_txt = f"Model LIME menilai {salesname} pada periode {month} {year} menunjukan {prediction} performance. "
        lime_txt += f"Hal ini dipengaruhi oleh kondisi seperti "
        lime_txt += f"{list(feature_categories.items())[0][0]} yang {list(feature_categories.items())[0][1]} dan "
        lime_txt += f"{list(feature_categories.items())[1][0]} yang {list(feature_categories.items())[1][1]}."
        texts.append(lime_txt)

    # Anchor narrative
    if anchor_res and anchor_res["anchor"]:
        anchor_txt = f"Model Anchors menilai {salesname} pada periode {month} {year} menunjukan {prediction} performance. "
        anchor_txt += f"Penilaian ini terutama karena indikator {', '.join([f for f in feature_categories.keys() if f in anchor_res['anchor']])} berada pada kategori yang sesuai, " \
                      f"dengan tingkat kepastian sekitar {anchor_res['precision']:.0%}."
        texts.append(anchor_txt)

    return texts

# ==== Combined Analyze Endpoint ====
@app.post("/analyze")
def analyze(data: SalesInput):
    try:
        # --- Extract features ---
        perf = data.performance
        row = [[
            perf["attendance"]["ontime"],
            perf["attendance"]["late"],
            perf["visit"],
            perf["productSold"],
            perf["salesValue"]
        ]]
        X_scaled = scaler.transform(row)

        # --- Prediction ---
        pred_class = model.predict(X_scaled)[0]
        probas = model.predict_proba(X_scaled)[0].tolist()
        prediction_result = {
            "salesname": data.salesname,
            "prediction": pred_class,
            "probabilities": {c: p for c, p in zip(model.classes_, probas)},
            "features": {f: float(v) for f, v in zip(features, row[0])}
        }

        # --- SHAP ---
        shap_explainer = shap.TreeExplainer(model)
        shap_values = shap_explainer.shap_values(X_scaled)
        shap_result = {
            "base_value": float(shap_explainer.expected_value[0]),
            "contributions": [
                {"feature": f, "shap_value": float(v)}
                for f, v in zip(features, shap_values[0][0])
            ]
        }

        # --- LIME ---
        lime_exp = lime_explainer.explain_instance(
            data_row=X_scaled[0],
            predict_fn=model.predict_proba,
            num_features=len(features)
        )
        lime_result = lime_exp.as_list()

        # --- Anchor ---
        anchor_exp = anchor_explainer.explain(X_scaled[0])
        anchor_result = {
            "precision": float(anchor_exp.data['precision']),
            "coverage": float(anchor_exp.data['coverage']),
            "anchor": anchor_exp.data['anchor']
        }

        # --- Narrative Text ---
        narrative_texts = generate_explanation_text(
            data.salesname, data.month, data.year,
            pred_class, shap_result, lime_result, anchor_result, X_scaled
        )

        # --- Final response ---
        return {
            "prediction": prediction_result,
            "explanations": {
                "shap": shap_result,
                "lime": lime_result,
                "anchor": anchor_result,
                "narrative": narrative_texts
            }
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
