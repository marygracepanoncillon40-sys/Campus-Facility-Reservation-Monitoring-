"""Module 4: Decision-support dashboard: KPIs, review queue, interpretation, human follow-up."""
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
from modules.data_processing import load_metadata, validate_schema
from modules.prediction import score_dataframe, HIGH, MEDIUM

FOLLOW_UP = {
    "High": "Facility Administrator contacts the requester to confirm; offer waitlist/overbooking release if unconfirmed.",
    "Medium": "Send a confirmation reminder 48 h before; re-check risk if no reply.",
    "Low": "Standard workflow; no extra action.",
}
DECISIONS = ["Keep reservation", "Send reminder", "Contact requester", "Offer slot to waitlist", "Escalate to supervisor"]


def render():
    meta = load_metadata()
    st.header("4 · Decision-Support Dashboard")
    df = st.session_state.get("df")
    if df is None:
        st.warning("Load a dataset on the Dataset page first.")
        return
    v = validate_schema(df, meta)
    if not v["ok"]:
        st.error("Dataset does not match the model schema; cannot score. Missing: " + ", ".join(v["missing"] or ["n/a"]))
        return
    scored = score_dataframe(df, meta)
    st.caption(f"Model v{meta['model_version']} · scored {dt.datetime.now():%Y-%m-%d %H:%M} · "
               f"tiers: High ≥ {HIGH:.0%}, Medium ≥ {MEDIUM:.0%}")

    hi = scored[scored.risk_tier == "High"]
    lost = float((scored.risk_probability * scored.duration_hours).sum())
    k = st.columns(5)
    k[0].metric("Reservations", f"{len(scored):,}")
    k[1].metric("High-risk", f"{len(hi):,}", f"{len(hi)/len(scored):.1%} of total", delta_color="off")
    k[2].metric("Avg predicted risk", f"{scored.risk_probability.mean():.1%}")
    k[3].metric("Expected unused hours", f"{lost:,.0f} h")
    obs = scored[meta["target"]].mean() if meta["target"] in scored else None
    k[4].metric("Observed cancel/no-show", f"{obs:.1%}" if obs is not None else "n/a")

    a, b = st.columns(2)
    tier_counts = scored.risk_tier.value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    tier_counts.columns = ["tier", "count"]
    a.plotly_chart(px.bar(tier_counts, x="tier", y="count", text="count", title="Reservations by review tier"), use_container_width=True)
    scored["exp_lost_hours"] = scored.risk_probability * scored.duration_hours
    fac = scored.groupby("facility_type").agg(avg_risk=("risk_probability", "mean"),
                                              exp_lost_hours=("exp_lost_hours", "sum")).reset_index()
    b.plotly_chart(px.bar(fac.sort_values("exp_lost_hours", ascending=False), x="facility_type", y="exp_lost_hours",
                          hover_data=["avg_risk"], title="Expected unused hours by facility"), use_container_width=True)

    st.subheader("Interpretation")
    worst = fac.sort_values("exp_lost_hours", ascending=False).iloc[0]
    top_req = scored.groupby("requester_type").risk_probability.mean().idxmax()
    st.markdown(
        f"- **{len(hi)}** reservations ({len(hi)/len(scored):.1%}) are high-risk and should be reviewed first.\n"
        f"- **{worst.facility_type}** carries the most expected unused time (~{worst.exp_lost_hours:,.0f} h).\n"
        f"- **{top_req}** requesters have the highest average predicted risk.\n"
        f"- Scores indicate likelihood, not certainty. Validate against the model's recall ({meta['metrics']['recall']:.0%}) "
        f"and precision ({meta['metrics']['precision']:.0%}) before acting at scale.")

    st.subheader("Risk / review queue")
    tiers = st.multiselect("Tiers to show", ["High", "Medium", "Low"], default=["High"])
    fac_f = st.multiselect("Facility filter", sorted(scored.facility_type.unique()))
    q = scored[scored.risk_tier.isin(tiers)]
    if fac_f:
        q = q[q.facility_type.isin(fac_f)]
    q = q.sort_values("risk_probability", ascending=False)
    id_col = "reservation_id" if "reservation_id" in q else None
    show = ([id_col] if id_col else []) + ["facility_type", "requester_type", "day_of_week", "time_slot", "lead_time_days",
                                          "prior_cancellations", "prior_noshows", "risk_probability", "risk_tier"]
    st.dataframe(q[show].head(100), use_container_width=True)
    st.download_button("Download review queue (CSV)", q[show].to_csv(index=False), "review_queue.csv")

    st.subheader("Recommended human follow-up")
    st.table(pd.DataFrame({"Tier": FOLLOW_UP.keys(), "Recommended action": FOLLOW_UP.values()}))

    st.subheader("Record final decision")
    role = st.session_state.get("role", "Facility Administrator")
    if role != "Facility Administrator":
        st.info(f"Signed in as **{role}**: view-only. Only a Facility Administrator can record decisions.")
    elif q.empty:
        st.info("No reservations in the selected queue.")
    else:
        with st.form("decision"):
            pick = st.selectbox("Reservation", q[id_col].head(100).tolist() if id_col else q.index[:100].tolist())
            dec = st.selectbox("Decision", DECISIONS)
            who = st.text_input("Reviewer name")
            note = st.text_area("Justification (required)")
            if st.form_submit_button("Save decision"):
                if not who.strip() or not note.strip():
                    st.error("Reviewer name and justification are required.")
                else:
                    row = q[q[id_col] == pick].iloc[0] if id_col else q.loc[pick]
                    st.session_state.setdefault("decision_log", []).append({
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"), "reservation": pick,
                        "risk_probability": row.risk_probability, "decision": dec, "reviewer": who, "note": note,
                        "model_version": meta["model_version"]})
                    st.success("Decision recorded (session log).")
    log = st.session_state.get("decision_log", [])
    if log:
        st.dataframe(pd.DataFrame(log), use_container_width=True)
        st.download_button("Download decision log", pd.DataFrame(log).to_csv(index=False), "decision_log.csv")
