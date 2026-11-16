"""Module 1: Dataset upload, schema validation, preview, null/duplicate summary."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "sample_reservations.csv"
META_PATH = ROOT / "models" / "model_metadata.json"


@st.cache_data
def load_metadata() -> dict:
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def read_uploaded_csv(file):
    """Return (DataFrame | None, error message | None)."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError:
        return None, "The file is empty or has no header row."
    except pd.errors.ParserError as e:
        return None, f"The file could not be parsed as CSV: {e}"
    except UnicodeDecodeError:
        return None, "The file is not UTF-8 text. Re-export it as CSV (UTF-8)."
    if df.empty:
        return None, "The CSV has headers but contains no rows."
    df.columns = [str(c).strip() for c in df.columns]
    return df, None


def validate_schema(df: pd.DataFrame, meta: dict) -> dict:
    """Check required feature columns, optional target, leakage columns and value types."""
    features, target = meta["features"], meta["target"]
    missing = [c for c in features if c not in df.columns]
    leakage = [c for c in meta["leakage_excluded"] if c in df.columns]
    extra = [c for c in df.columns if c not in features + [target] + meta["leakage_excluded"]]
    bad_numeric, unseen = {}, {}
    for c in meta["numeric"]:
        if c in df.columns:
            n_bad = int(pd.to_numeric(df[c], errors="coerce").isna().sum() - df[c].isna().sum())
            if n_bad > 0:
                bad_numeric[c] = n_bad
    for c in meta["categorical"]:
        if c in df.columns:
            new = sorted(set(df[c].dropna().astype(str)) - set(meta["categories"][c]))
            if new:
                unseen[c] = new
    return {"missing": missing, "leakage": leakage, "extra": extra, "bad_numeric": bad_numeric,
            "unseen": unseen, "has_target": target in df.columns, "ok": not missing and not bad_numeric}


def prepare_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Exact training feature order/types; leakage and target columns are never passed to the model."""
    X = df[meta["features"]].copy()
    for c in meta["numeric"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in meta["categorical"]:
        X[c] = X[c].astype("string").str.strip().astype(object)
    return X


@st.cache_data
def quality_summary(df: pd.DataFrame):
    nulls = df.isna().sum().rename("null_count").to_frame()
    nulls["null_pct"] = (nulls["null_count"] / len(df) * 100).round(2)
    return nulls, int(df.duplicated().sum())


def render():
    meta = load_metadata()
    st.header("1 · Dataset Upload and Validation")
    st.caption("Upload a reservation CSV, or use the bundled synthetic sample. "
               "Use only synthetic or properly authorized data in the public deployment.")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        df, err = read_uploaded_csv(uploaded)
        if err:
            st.error(err)
            return
        st.session_state["df"], st.session_state["df_name"] = df, uploaded.name
    elif st.button("Use bundled sample dataset"):
        st.session_state["df"], st.session_state["df_name"] = load_sample(), "sample_reservations.csv"

    df = st.session_state.get("df")
    if df is None:
        st.info("No dataset loaded yet.")
        with st.expander("Expected schema"):
            st.write("**Required feature columns:**", ", ".join(meta["features"]))
            st.write("**Optional target (for evaluation/EDA):**", meta["target"])
            st.write("**Excluded as leakage:**", ", ".join(meta["leakage_excluded"]))
        return

    st.success(f"Loaded **{st.session_state['df_name']}**")
    v = validate_schema(df, meta)
    if v["missing"]:
        st.error("Missing required columns: " + ", ".join(v["missing"]))
    if v["bad_numeric"]:
        st.error("Non-numeric values in numeric columns: " +
                 ", ".join(f"{k} ({n} rows)" for k, n in v["bad_numeric"].items()))
    if v["leakage"]:
        st.warning("Leakage columns detected and excluded from inference: " + ", ".join(v["leakage"]))
    if v["unseen"]:
        st.warning("Categories not seen in training (encoded as all-zero): " +
                   "; ".join(f"{k}: {', '.join(x)}" for k, x in v["unseen"].items()))
    if not v["has_target"]:
        st.info(f"Target column '{meta['target']}' not present: EDA class charts and observed-rate KPIs are limited.")
    if v["ok"]:
        st.success("Schema check passed.")

    c1, c2, c3 = st.columns(3)
    nulls, dups = quality_summary(df)
    c1.metric("Rows", f"{len(df):,}"); c2.metric("Columns", df.shape[1]); c3.metric("Duplicate rows", dups)
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)
    a, b = st.columns(2)
    a.subheader("Null summary"); a.dataframe(nulls, use_container_width=True)
    b.subheader("Column types"); b.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), use_container_width=True)
    if dups and st.checkbox("Drop duplicate rows from the working dataset"):
        st.session_state["df"] = df.drop_duplicates().reset_index(drop=True)
        st.rerun()
