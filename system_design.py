"""Module 5: SAD integration: requirements, actors, use case diagram, process flow."""
from pathlib import Path
import pandas as pd
import streamlit as st
from modules.prediction import HIGH, MEDIUM

ASSETS = Path(__file__).resolve().parent.parent / "assets"

FR = [
    ("FR1", "Dataset upload & validation", "System shall accept a CSV, check required columns/types, flag leakage fields and report nulls and duplicates.", "Dataset page"),
    ("FR2", "Exploratory analysis", "System shall show descriptive statistics, class distribution and interactive charts of cancellation patterns.", "Data Analysis page"),
    ("FR3", "Risk prediction", "System shall predict cancel/no-show probability for a new or batch reservation using the saved pipeline.", "Prediction page"),
    ("FR4", "Review queue", "System shall rank reservations into High/Medium/Low tiers and list those needing administrator review.", "Dashboard"),
    ("FR5", "Decision recording", "System shall let only a Facility Administrator record a final decision with justification, timestamp and model version.", "Dashboard"),
]
NFR = [
    ("NFR1", "Performance", "Model loads once (cached); a single prediction returns in under 2 seconds."),
    ("NFR2", "Privacy & security", "Only synthetic/authorized data is deployed; no credentials or personal records in the repository."),
    ("NFR3", "Transparency & usability", "Model version, timestamp, limitations and human-review notice are visible on prediction output."),
]
ACTORS = pd.DataFrame([
    ["Facility Requester", "Student org / faculty / staff", "Submits reservations (source of input data)", "No access to risk scores"],
    ["Data Analyst", "IS staff", "Uploads data, runs EDA, checks model metrics", "View results; cannot record decisions"],
    ["Facility Administrator", "Decision maker", "Reviews queue, records the final decision", "View and record decisions"],
    ["Predictive Model (system actor)", "Saved sklearn pipeline", "Produces probability and tier", "Advisory only"],
], columns=["Actor", "Who", "Responsibility", "Access"])


def render():
    st.header("5 · SAD Integration")
    st.subheader("Functional requirements")
    st.dataframe(pd.DataFrame(FR, columns=["ID", "Name", "Requirement", "Implemented in"]), use_container_width=True, hide_index=True)
    st.subheader("Non-functional requirements")
    st.dataframe(pd.DataFrame(NFR, columns=["ID", "Category", "Requirement"]), use_container_width=True, hide_index=True)
    st.subheader("Actors and permissions")
    st.dataframe(ACTORS, use_container_width=True, hide_index=True)
    st.subheader("Updated use case diagram")
    p = ASSETS / "use_case_diagram.png"
    st.image(str(p)) if p.exists() else st.error("assets/use_case_diagram.png is missing.")
    st.subheader("Analytical process flow")
    p2 = ASSETS / "analytical_process_flow.png"
    st.image(str(p2)) if p2.exists() else st.error("assets/analytical_process_flow.png is missing.")
    st.subheader("Decision workflow")
    st.markdown(f"""
1. Requester books a facility → reservation record created.
2. Analyst uploads/validates the data; model scores each booking (checkpoint: booking time).
3. Tier assigned: **High ≥ {HIGH:.0%}**, **Medium ≥ {MEDIUM:.0%}**, otherwise Low.
4. High/Medium items enter the review queue with recommended follow-up.
5. **Facility Administrator** reviews, contacts requester if needed, and records the final decision with justification.
6. Outcomes feed the next retraining cycle.

*Predictions are decision support only; they never automatically reject a reservation or deny a service.*""")
