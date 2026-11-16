import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyBboxPatch, FancyArrowPatch

# ---------- Use case diagram ----------
fig, ax = plt.subplots(figsize=(12, 8)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
ax.add_patch(Rectangle((3, .3), 6, 7.4, fill=False, lw=2))
ax.text(6, 7.45, "Campus Facility Reservation Monitoring System", ha="center", fontweight="bold")
cases = {1: "UC1 Upload & validate dataset", 2: "UC2 Explore data (EDA)", 3: "UC3 Predict cancel/no-show risk",
         4: "UC4 View KPI dashboard", 5: "UC5 Review risk queue", 6: "UC6 Record final decision",
         7: "UC7 Submit reservation request"}
pos = {1: (6, 6.7), 2: (6, 5.75), 3: (6, 4.8), 4: (6, 3.85), 5: (6, 2.9), 6: (6, 1.95), 7: (6, 1.0)}
for k, t in cases.items():
    ax.add_patch(Ellipse(pos[k], 4.6, .8, fc="#e8f1fb", ec="#1f4e79")); ax.text(*pos[k], t, ha="center", va="center", fontsize=9)
def actor(x, y, name):
    ax.add_patch(plt.Circle((x, y + .6), .18, fill=False, lw=1.5))
    ax.plot([x, x], [y + .42, y - .1], "k", lw=1.5); ax.plot([x - .3, x + .3], [y + .25, y + .25], "k", lw=1.5)
    ax.plot([x, x - .25], [y - .1, y - .6], "k", lw=1.5); ax.plot([x, x + .25], [y - .1, y - .6], "k", lw=1.5)
    ax.text(x, y - .95, name, ha="center", fontsize=9, fontweight="bold")
actor(1.2, 6.0, "Data Analyst"); actor(1.2, 2.6, "Facility\nAdministrator"); actor(10.9, 1.3, "Facility\nRequester")
actor(10.9, 4.6, "Predictive Model\n(system)")
links = [((1.6, 6.0), 1), ((1.6, 5.8), 2), ((1.6, 5.6), 3), ((1.6, 2.8), 3), ((1.6, 2.8), 4), ((1.6, 2.6), 5),
         ((1.6, 2.4), 6), ((10.5, 1.3), 7), ((10.5, 4.6), 3)]
for (x, y), k in links:
    px_, py_ = pos[k]; side = -2.3 if x < 6 else 2.3
    ax.plot([x, px_ + side], [y, py_], color="#555", lw=1)
ax.text(6, .05, "Administrator decisions are final; model output is advisory only.", ha="center", fontsize=8, style="italic")
fig.savefig("assets/use_case_diagram.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- Analytical process flow ----------
steps = ["Upload CSV", "Schema check &\nleakage removal", "EDA\n(stats, charts)", "Saved pipeline\n(joblib, cached)",
         "Probability &\nrisk tier", "Dashboard &\nreview queue", "Administrator\nfinal decision", "Outcome log →\nretraining"]
fig, ax = plt.subplots(figsize=(13, 3.2)); ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")
w, gap = 1.35, .28
for i, s in enumerate(steps):
    x = .2 + i * (w + gap)
    col = "#fde9d9" if i == 6 else "#e8f1fb"
    ax.add_patch(FancyBboxPatch((x, 1), w, 1.1, boxstyle="round,pad=0.03", fc=col, ec="#1f4e79"))
    ax.text(x + w / 2, 1.55, s, ha="center", va="center", fontsize=8)
    ax.text(x + w / 2, .75, str(i + 1), ha="center", fontsize=9, fontweight="bold")
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + w + .03, 1.55), (x + w + gap - .03, 1.55), arrowstyle="->", mutation_scale=10))
ax.text(6.5, 2.6, "Analytical Process Flow", ha="center", fontweight="bold")
fig.savefig("assets/analytical_process_flow.png", dpi=150, bbox_inches="tight"); plt.close(fig)
