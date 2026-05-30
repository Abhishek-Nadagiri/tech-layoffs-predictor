"""
train_models.py
Trains and saves all ML models for the Tech Layoffs Analysis project.
Run once: python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report, accuracy_score,
    mean_squared_error, r2_score, silhouette_score
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "tech_layoffs_hiring_trends_elite_v2.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load & basic prep ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows × {df.shape[1]} cols")

# ── Encode categoricals ────────────────────────────────────────────────────────
cat_cols = ["industry", "country", "company_size", "hiring_trend",
            "reason_for_layoffs", "market_condition", "month", "top_hiring_role"]

encoders = {}
df_enc = df.copy()
for col in cat_cols:
    le = LabelEncoder()
    df_enc[col + "_enc"] = le.fit_transform(df_enc[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
print("✔ Label encoders saved")

# ── Feature list shared across models ─────────────────────────────────────────
NUMERIC_FEATURES = [
    "year", "layoffs_count", "layoff_percentage",
    "ai_automation_impact", "ai_replacement_risk", "open_roles",
    "remote_jobs_percentage", "stock_growth_percent",
    "revenue_growth_percent", "salary_budget_change",
    "ai_adoption_level", "employee_sentiment", "job_security_score"
]
ENC_CAT_FEATURES = [c + "_enc" for c in cat_cols]
ALL_FEATURES = NUMERIC_FEATURES + ENC_CAT_FEATURES

# ── 1. Hiring-trend classifier (4-class) ──────────────────────────────────────
print("\n[1] Training Hiring Trend Classifier …")
X_ht = df_enc[ALL_FEATURES]
y_ht = df_enc["hiring_trend_enc"]
X_tr, X_te, y_tr, y_te = train_test_split(X_ht, y_ht, test_size=0.2, random_state=42)

rf_ht = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_ht.fit(X_tr, y_tr)
acc = accuracy_score(y_te, rf_ht.predict(X_te))
print(f"   Accuracy: {acc:.4f}")

joblib.dump(rf_ht, os.path.join(MODELS_DIR, "hiring_trend_classifier.pkl"))
joblib.dump(ALL_FEATURES, os.path.join(MODELS_DIR, "feature_list.pkl"))
print("✔ Hiring trend classifier saved")

# ── 2. Layoff-count regressor ──────────────────────────────────────────────────
print("\n[2] Training Layoff Count Regressor …")
reg_features = [f for f in ALL_FEATURES if f not in ("layoffs_count",)]
X_lc = df_enc[reg_features]
y_lc = df_enc["layoffs_count"]
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_lc, y_lc, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf_reg.fit(X_tr2, y_tr2)
r2 = r2_score(y_te2, rf_reg.predict(X_te2))
print(f"   R² score: {r2:.4f}")

joblib.dump(rf_reg, os.path.join(MODELS_DIR, "layoff_count_regressor.pkl"))
joblib.dump(reg_features, os.path.join(MODELS_DIR, "reg_feature_list.pkl"))
print("✔ Layoff count regressor saved")

# ── 3. AI-risk classifier (High/Medium/Low) ───────────────────────────────────
print("\n[3] Training AI Replacement Risk Classifier …")
df_enc["risk_category"] = pd.cut(
    df_enc["ai_replacement_risk"], bins=[0, 4, 7, 10],
    labels=["Low", "Medium", "High"]
).astype(str)
risk_le = LabelEncoder()
df_enc["risk_cat_enc"] = risk_le.fit_transform(df_enc["risk_category"])
encoders["risk_category"] = risk_le
joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))

X_risk = df_enc[ALL_FEATURES]
y_risk = df_enc["risk_cat_enc"]
X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)

gb_risk = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
gb_risk.fit(X_tr3, y_tr3)
acc3 = accuracy_score(y_te3, gb_risk.predict(X_te3))
print(f"   Accuracy: {acc3:.4f}")

joblib.dump(gb_risk, os.path.join(MODELS_DIR, "ai_risk_classifier.pkl"))
print("✔ AI risk classifier saved")

# ── 4. Job Security Score regressor ───────────────────────────────────────────
print("\n[4] Training Job Security Score Regressor …")
jss_features = [f for f in ALL_FEATURES if f != "job_security_score"]
X_jss = df_enc[jss_features]
y_jss = df_enc["job_security_score"]
X_tr4, X_te4, y_tr4, y_te4 = train_test_split(X_jss, y_jss, test_size=0.2, random_state=42)

rf_jss = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
rf_jss.fit(X_tr4, y_tr4)
r2_jss = r2_score(y_te4, rf_jss.predict(X_te4))
print(f"   R² score: {r2_jss:.4f}")

joblib.dump(rf_jss, os.path.join(MODELS_DIR, "job_security_regressor.pkl"))
joblib.dump(jss_features, os.path.join(MODELS_DIR, "jss_feature_list.pkl"))
print("✔ Job security regressor saved")

# ── 5. KMeans clustering (company risk profiles) ──────────────────────────────
print("\n[5] Training KMeans Clustering …")
cluster_features = [
    "layoffs_count", "layoff_percentage", "ai_automation_impact",
    "ai_replacement_risk", "open_roles", "stock_growth_percent",
    "revenue_growth_percent", "employee_sentiment", "job_security_score"
]
scaler = StandardScaler()
X_cluster = scaler.fit_transform(df_enc[cluster_features])

km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_cluster)
df_enc["cluster"] = km.labels_
sil = silhouette_score(X_cluster, km.labels_, sample_size=2000)
print(f"   Silhouette score: {sil:.4f}")

cluster_labels = {
    0: "Stable & Growing",
    1: "High Risk / Downsizing",
    2: "AI-Disrupted",
    3: "Recovering"
}
# Assign meaningful labels based on cluster centroids
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=cluster_features)
layoff_order = centers["layoffs_count"].argsort().values
cluster_map = {
    layoff_order[0]: "Stable & Growing",
    layoff_order[1]: "Moderate Risk",
    layoff_order[2]: "High Risk / Downsizing",
    layoff_order[3]: "Severe Layoffs",
}

joblib.dump(km, os.path.join(MODELS_DIR, "kmeans_cluster.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "cluster_scaler.pkl"))
joblib.dump(cluster_features, os.path.join(MODELS_DIR, "cluster_features.pkl"))
joblib.dump(cluster_map, os.path.join(MODELS_DIR, "cluster_map.pkl"))
print("✔ KMeans clustering saved")

# ── 6. Feature importance (global) ────────────────────────────────────────────
feat_imp = pd.Series(rf_ht.feature_importances_, index=ALL_FEATURES).sort_values(ascending=False)
feat_imp.to_csv(os.path.join(MODELS_DIR, "feature_importances.csv"))
print("✔ Feature importances saved")

# ── Save model metrics ─────────────────────────────────────────────────────────
metrics = {
    "hiring_trend_classifier": {"accuracy": round(acc, 4), "type": "classification"},
    "layoff_count_regressor": {"r2": round(r2, 4), "type": "regression"},
    "ai_risk_classifier": {"accuracy": round(acc3, 4), "type": "classification"},
    "job_security_regressor": {"r2": round(r2_jss, 4), "type": "regression"},
    "kmeans_cluster": {"silhouette": round(sil, 4), "type": "clustering"},
}
joblib.dump(metrics, os.path.join(MODELS_DIR, "model_metrics.pkl"))

print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print("All models trained & saved to ./models/")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
