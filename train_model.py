"""Reproduces the midterm modelling step ONCE and saves the fitted preprocessing+model pipeline.
Adapt FEATURES/TARGET and the estimator to match your submitted midterm notebook exactly."""
import json, datetime as dt
import joblib, pandas as pd, sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "cancel_or_noshow"
CATEGORICAL = ["facility_type","requester_type","day_of_week","time_slot","purpose"]
NUMERIC = ["lead_time_days","duration_hours","expected_attendees","prior_cancellations",
           "prior_noshows","is_recurring","reminder_sent","requires_equipment"]
LEAKAGE = ["final_status"]
FEATURES = CATEGORICAL + NUMERIC
VERSION = "1.0.0"

df = pd.read_csv("data/sample_reservations.csv")
X, y = df[FEATURES], df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUMERIC),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)])
pipe = Pipeline([("prep", pre),
                 ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42, n_jobs=-1))])
pipe.fit(Xtr, ytr)

proba = pipe.predict_proba(Xte)[:, 1]; pred = (proba >= 0.5).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
names = pipe.named_steps["prep"].get_feature_names_out()
imp = sorted(zip(names, pipe.named_steps["clf"].feature_importances_), key=lambda t: -t[1])[:12]

meta = {
    "model_name": "RandomForestClassifier (preprocessing + model pipeline)",
    "model_version": VERSION, "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
    "sklearn_version": sklearn.__version__, "target": TARGET, "positive_class": "Cancel / No-show (1)",
    "features": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC, "leakage_excluded": LEAKAGE,
    "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
    "numeric_ranges": {c: [float(df[c].min()), float(df[c].max()), float(df[c].median())] for c in NUMERIC},
    "train_rows": int(len(Xtr)), "test_rows": int(len(Xte)),
    "metrics": {"accuracy": round(accuracy_score(yte, pred), 4), "precision": round(precision_score(yte, pred), 4),
                "recall": round(recall_score(yte, pred), 4), "f1": round(f1_score(yte, pred), 4),
                "roc_auc": round(roc_auc_score(yte, proba), 4),
                "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}},
    "feature_importance": [[n, float(v)] for n, v in imp],
    "limitations": ["Trained on synthetic data; replace with authorized midterm data.",
                    "Probabilities are not calibrated guarantees.",
                    "Predicts risk only; cannot justify denying a reservation or service."],
}
joblib.dump(pipe, "models/trained_model.joblib")
json.dump(meta, open("models/model_metadata.json", "w"), indent=2)
ref = Xte.head(25).copy(); ref["expected_probability"] = proba[:25].round(6)
ref.to_csv("data/reference_test_cases.csv", index=False)
print(json.dumps(meta["metrics"], indent=1))
