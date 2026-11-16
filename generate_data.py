"""Generate SYNTHETIC campus facility reservation data (no real student records).
Replace with your midterm dataset: keep the same column names or edit modules/data_processing.py."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
N = 1500
facility = rng.choice(["Classroom","Gymnasium","Auditorium","Laboratory","Conference Room","Sports Field"], N, p=[.25,.12,.10,.15,.23,.15])
requester = rng.choice(["Student Org","Faculty","Staff","External"], N, p=[.40,.30,.20,.10])
day = rng.choice(["Mon","Tue","Wed","Thu","Fri","Sat"], N, p=[.2,.2,.2,.18,.15,.07])
slot = rng.choice(["Morning","Afternoon","Evening"], N, p=[.4,.4,.2])
purpose = rng.choice(["Academic","Meeting","Event","Sports","Training"], N, p=[.3,.25,.2,.15,.1])
lead = np.clip(rng.gamma(2.2, 6, N).round(), 0, 60).astype(int)
dur = np.clip(rng.normal(2.5, 1.2, N).round(1), 0.5, 8)
att = np.clip(rng.normal(40, 30, N).round(), 2, 300).astype(int)
prior_c = rng.poisson(np.where(requester == "Student Org", 1.0, 0.4))
prior_n = rng.poisson(np.where(requester == "External", 0.8, 0.3))
rec = rng.binomial(1, .2, N); rem = rng.binomial(1, .6, N); eq = rng.binomial(1, .35, N)
z = (-2.3 + 0.045*lead + 0.35*prior_c + 0.55*prior_n - 0.8*rem - 0.4*rec
     + 0.3*(slot == "Evening") + 0.4*(day == "Sat") + 0.35*(purpose == "Event")
     + 0.3*(requester == "Student Org") + 0.5*(requester == "External") + 0.08*dur
     + rng.normal(0, .6, N))
y = rng.binomial(1, 1/(1+np.exp(-z)))
final = np.where(y == 0, "Completed", np.where(rng.random(N) < .55, "Cancelled", "No-show"))
df = pd.DataFrame({
    "reservation_id": [f"R{100000+i}" for i in range(N)],
    "facility_type": facility, "requester_type": requester, "day_of_week": day,
    "time_slot": slot, "purpose": purpose, "lead_time_days": lead, "duration_hours": dur,
    "expected_attendees": att, "prior_cancellations": prior_c, "prior_noshows": prior_n,
    "is_recurring": rec, "reminder_sent": rem, "requires_equipment": eq,
    "final_status": final,            # LEAKAGE: known only after the event -> never a feature
    "cancel_or_noshow": y,            # target
})
df.to_csv("data/sample_reservations.csv", index=False)
print(df.shape, df.cancel_or_noshow.mean().round(3))
