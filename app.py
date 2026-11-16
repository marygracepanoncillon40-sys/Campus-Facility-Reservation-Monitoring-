import streamlit as st

st.set_page_config(page_title="Campus Facility Reservation Monitoring", page_icon="🏫", layout="wide")

from modules import data_processing, exploratory_analysis, prediction, dashboard, system_design  # noqa: E402

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
menu = st.sidebar.selectbox("Module", PAGES)
st.session_state["role"] = st.sidebar.selectbox("Signed in as (demo role)",
                                                ["Facility Administrator", "Data Analyst", "Facility Requester"])
meta = data_processing.load_metadata()
st.sidebar.caption(f"Model v{meta['model_version']} · {meta['model_name'].split(' (')[0]}")

if menu == "Home":
    st.title("🏫 Campus Facility Reservation Monitoring")
    st.write("Predicts which reservations are likely to be **cancelled or become no-shows**, so a Facility "
             "Administrator can follow up early and reduce unused facility time.")
    c = st.columns(5)
    for col, (t, d) in zip(c, [("1 Dataset", "Upload & validate"), ("2 Analysis", "Stats & charts"),
                               ("3 Prediction", "Single & batch"), ("4 Dashboard", "KPIs & review queue"),
                               ("5 System Design", "Requirements & UML")]):
        col.info(f"**{t}**\n\n{d}")
    m = meta["metrics"]
    st.subheader("Model at a glance")
    st.write(f"Target: `{meta['target']}` · Features: {len(meta['features'])} · ROC-AUC {m['roc_auc']:.2f} · "
             f"Recall {m['recall']:.2f} · Precision {m['precision']:.2f}")
    st.warning("Decision-support only. Predictions never authorize rejecting a reservation or denying student services. "
               "This deployment uses synthetic data.")
elif menu == "Dataset":
    data_processing.render()
elif menu == "Data Analysis":
    exploratory_analysis.render()
elif menu == "Prediction":
    prediction.render()
elif menu == "Dashboard":
    dashboard.render()
else:
    system_design.render()
