"""Module 2: Descriptive statistics, class distribution and interactive Plotly charts."""
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.data_processing import load_metadata


@st.cache_data
def describe(df: pd.DataFrame, cols: tuple):
    return df[list(cols)].describe().T.round(2)


def render():
    meta = load_metadata()
    st.header("2 · Exploratory Data Analysis")
    df = st.session_state.get("df")
    if df is None:
        st.warning("Load a dataset on the Dataset page first.")
        return
    target = meta["target"]
    num = [c for c in meta["numeric"] if c in df.columns]
    cat = [c for c in meta["categorical"] if c in df.columns]
    has_t = target in df.columns
    if not num and not cat:
        st.error("The dataset has none of the model's feature columns.")
        return

    st.subheader("Descriptive statistics")
    if num:
        st.dataframe(describe(df, tuple(num)), use_container_width=True)

    st.subheader("Class distribution")
    if has_t:
        counts = df[target].value_counts().rename_axis("class").reset_index(name="count")
        counts["class"] = counts["class"].map({0: "Attended (0)", 1: "Cancel/No-show (1)"}).fillna(counts["class"].astype(str))
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.pie(counts, names="class", values="count", hole=.45, title="Share of outcomes"), use_container_width=True)
        c2.plotly_chart(px.bar(counts, x="class", y="count", text="count", title="Outcome counts"), use_container_width=True)
        rate = df[target].mean()
        if min(rate, 1 - rate) < 0.35:
            st.info(f"Imbalanced classes ({rate:.1%} positive). Prefer recall/F1/AUC over accuracy alone.")
    else:
        st.info("No target column in this dataset; class distribution unavailable.")

    st.subheader("Interactive charts")
    t1, t2, t3, t4 = st.tabs(["Histogram", "Rate by category", "Box plot", "Correlation"])
    with t1:
        if num:
            col = st.selectbox("Numeric feature", num, key="h")
            st.plotly_chart(px.histogram(df, x=col, color=target if has_t else None, barmode="overlay",
                                         opacity=.7, nbins=30, title=f"Distribution of {col}"), use_container_width=True)
    with t2:
        if cat and has_t:
            col = st.selectbox("Categorical feature", cat, key="c")
            g = df.groupby(col)[target].agg(rate="mean", n="size").reset_index().sort_values("rate", ascending=False)
            st.plotly_chart(px.bar(g, x=col, y="rate", hover_data=["n"], text=g["rate"].map("{:.0%}".format),
                                   title=f"Cancel/no-show rate by {col}"), use_container_width=True)
        else:
            st.info("Requires a categorical feature and the target column.")
    with t3:
        if num and cat:
            y = st.selectbox("Numeric", num, key="by"); x = st.selectbox("Group by", cat, key="bx")
            st.plotly_chart(px.box(df, x=x, y=y, color=target if has_t else None, title=f"{y} by {x}"), use_container_width=True)
    with t4:
        cols = num + ([target] if has_t else [])
        if len(cols) >= 2:
            st.plotly_chart(px.imshow(df[cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r",
                                      zmin=-1, zmax=1, title="Correlation matrix"), use_container_width=True)
