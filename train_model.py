import json
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Load dataset JSON
# -----------------------------
with open("sales_dataset.json", "r") as f:
    raw_data = json.load(f)

# Flatten JSON
records = []
for row in raw_data:
    perf = row["performance"]
    records.append({
        "salesname": row["salesname"],
        "month": row["month"],
        "year": row["year"],
        "attendance_ontime": perf["attendance"]["ontime"],
        "attendance_late": perf["attendance"]["late"],
        "visit": perf["visit"],
        "productSold": perf["productSold"],
        "salesValue": perf["salesValue"],
    })

df = pd.DataFrame(records)

# -----------------------------
# 2. Features & Z-score transform
# -----------------------------
features = ["attendance_ontime", "attendance_late", "visit", "productSold", "salesValue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
df_z = pd.DataFrame(X_scaled, columns=features)

# -----------------------------
# 3. Generate score & label
# -----------------------------
weights = {
    "attendance_ontime": 0.2,
    "attendance_late": -0.1,
    "visit": 0.1,
    "productSold": 0.3,
    "salesValue": 0.5
}
df_z["score"] = sum(df_z[f] * w for f, w in weights.items())
df_z["performance_class"] = pd.qcut(df_z["score"], q=3, labels=["Low", "Mid", "High"])

print("Label distribution:\n", df_z["performance_class"].value_counts())

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X = df_z[features]
y = df_z["performance_class"]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    print("⚠️ Stratify gagal (kelas terlalu sedikit), fallback tanpa stratify")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# -----------------------------
# 5. Train RandomForest
# -----------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Training Accuracy:", clf.score(X_train, y_train))
print("Test Accuracy:", clf.score(X_test, y_test))

# -----------------------------
# 6. Save artifacts
# -----------------------------
artifacts = {
    "model": clf,
    "scaler": scaler,
    "features": features,
    "weights": weights
}

with open("sales_classifier.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✅ Model disimpan ke sales_classifier.pkl")
