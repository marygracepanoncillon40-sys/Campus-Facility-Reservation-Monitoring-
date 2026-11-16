"""Module 3: Load saved pipeline, accept new inputs, predict, show model metrics."""
import datetime as dt
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.data_processing import load_metadata, prepare_features, validate_schema

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
REF_PATH = ROOT / "data" / "reference_test_cases.csv"
HIGH, MEDIUM = 0.60, 0.35  # review-queue thresholds (documented in System Design)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def risk_tier(p: float) -> str:
    return "High" if p >= HIGH else "Medium" if p >= MEDIUM else "Low"


def score_dataframe(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Add risk_probability and risk_tier. Uses only the training features."""
    model = load_model()
    out = df.copy()
    out["risk_probability"] = model.predict_proba(prepare_features(df, meta))[:, 1].round(4)
    out["risk_tier"] = out["risk_probability"].map(risk_tier)
    return out


def _input_form(meta: dict) -> pd.DataFrame:
    r, cats = meta["numeric_ranges"], meta["categories"]
    c1, c2, c3 = st.columns(3)
    v = {}
    v["facility_type"] = c1.selectbox("Facility type", cats["facility_type"])
    v["requester_type"] = c2.selectbox("Requester type", cats["requester_type"])
    v["purpose"] = c3.selectbox("Purpose", cats["purpose"])
    v["day_of_week"] = c1.selectbox("Day of week", cats["day_of_week"])
    v["time_slot"] = c2.selectbox("Time slot", cats["time_slot"])
    v["lead_time_days"] = c3.number_input("Lead time (days before event)", 0, 120, int(r["lead_time_days"][2]))
    v["duration_hours"] = c1.number_input("Duration (hours)", 0.5, 12.0, float(r["duration_hours"][2]), 0.5)
    v["expected_attendees"] = c2.number_input("Expected attendees", 1, 1000, int(r["expected_attendees"][2]))
    v["prior_cancellations"] = c3.number_input("Requester's prior cancellations", 0, 50, 0)
    v["prior_noshows"] = c1.number_input("Requester's prior no-shows", 0, 50, 0)
    v["is_recurring"] = int(c2.checkbox("Recurring booking"))
    v["reminder_sent"] = int(c3.checkbox("Reminder scheduled", value=True))
    v["requires_equipment"] = int(c1.checkbox("Requires equipment"))
    return pd.DataFrame([v], columns=meta["features"])


def render():
    meta = load_metadata()
    st.header("3 · Predictive Modeling")
    try:
        model = load_model()
    except Exception as e:  # missing/incompatible joblib
        st.error(f"Saved model could not be loaded: {e}")
        return
    m = meta["metrics"]
    st.caption(f"Model: {meta['model_name']} · version {meta['model_version']} · trained {meta['trained_at']} · "
               f"scikit-learn {meta['sklearn_version']} · predicts: **{meta['positive_class']}**")

    tab1, tab2, tab3 = st.tabs(["Single prediction", "Batch prediction", "Model metrics"])
    with tab1:
        st.write("Enter details known **at booking time** (the prediction checkpoint). Post-event fields are not accepted.")
        X = _input_form(meta)
        if st.button("Predict", type="primary"):
            p = float(model.predict_proba(X)[0, 1])
            st.session_state["last_pred"] = {"p": p, "ts": dt.datetime.now().isoformat(timespec="seconds"), "X": X}
        lp = st.session_state.get("last_pred")
        if lp:
            p, tier = lp["p"], risk_tier(lp["p"])
            a, b, c = st.columns(3)
            a.metric("Predicted class", "Cancel / No-show" if p >= 0.5 else "Likely to attend")
            b.metric("Cancel/no-show probability", f"{p:.1%}")
            c.metric("Review tier", tier)
            st.caption(f"Prediction timestamp: {lp['ts']} · Model v{meta['model_version']}")
            msg = {"High": "Recommend administrator review: confirm with the requester and consider a waitlist.",
                   "Medium": "Recommend an automated reminder and a check-in before the slot.",
                   "Low": "No action needed beyond the normal workflow."}[tier]
            (st.warning if tier == "High" else st.info)(msg)
            st.dataframe(lp["X"], use_container_width=True)
        st.info("Limitations: a probabilistic estimate from historical patterns. It is **not** authorization to "
                "reject a reservation or deny a service; a Facility Administrator makes the final decision.")
    with tab2:
        up = st.file_uploader("Upload reservations to score (CSV)", type="csv", key="batch")
        df = None
        if up is not None:
            from modules.data_processing import read_uploaded_csv
            df, err = read_uploaded_csv(up)
            if err:
                st.error(err); df = None
        elif st.session_state.get("df") is not None:
            df = st.session_state["df"]; st.caption("Scoring the dataset loaded on the Dataset page.")
        if df is not None:
            v = validate_schema(df, meta)
            if not v["ok"]:
                st.error("Cannot score: " + (f"missing columns {v['missing']}. " if v["missing"] else "") +
                         (f"non-numeric values in {list(v['bad_numeric'])}." if v["bad_numeric"] else ""))
            else:
                scored = score_dataframe(df, meta)
                st.dataframe(scored.sort_values("risk_probability", ascending=False).head(50), use_container_width=True)
                st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_reservations.csv")
    with tab3:
        a, b, c, d, e = st.columns(5)
        a.metric("Accuracy", f"{m['accuracy']:.2f}"); b.metric("Precision", f"{m['precision']:.2f}")
        c.metric("Recall", f"{m['recall']:.2f}"); d.metric("F1", f"{m['f1']:.2f}"); e.metric("ROC-AUC", f"{m['roc_auc']:.2f}")
        cm = m["confusion_matrix"]
        fig = px.imshow([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: attend", "Pred: cancel/no-show"], y=["Actual: attend", "Actual: cancel/no-show"],
                        title=f"Confusion matrix (hold-out test, n={meta['test_rows']})")
        st.plotly_chart(fig, use_container_width=True)
        fi = pd.DataFrame(meta["feature_importance"], columns=["feature", "importance"])
        st.plotly_chart(px.bar(fi.iloc[::-1], x="importance", y="feature", orientation="h", title="Top feature importances"),
                        use_container_width=True)
        st.write("**Limitations**"); [st.write("- " + x) for x in meta["limitations"]]
        if REF_PATH.exists() and st.button("Run consistency check vs. midterm reference cases"):
            ref = pd.read_csv(REF_PATH)
            now = model.predict_proba(prepare_features(ref, meta))[:, 1]
            diff = float((now - ref["expected_probability"]).abs().max())
            (st.success if diff < 1e-4 else st.error)(f"Max difference on {len(ref)} reference cases: {diff:.6f}")
