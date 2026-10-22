"""Plot test AUC per modelling stage. Values are taken from the saved notebook outputs."""
import matplotlib.pyplot as plt

stages = [
    ("Popularity baseline", 0.6055, "03"),
    ("Logistic regression", 0.6527, "03"),
    ("LightGBM baseline", 0.6777, "03"),
    ("Hybrid LightGBM\n(affinity + CF + context)", 0.6846, "07"),
    ("Hybrid, final\n(retrained on train + val)", 0.7452, "08"),
]
labels = [s[0] for s in stages]
aucs = [s[1] for s in stages]
colors = ["#b8c4d6"] * (len(stages) - 1) + ["#3b6ea5"]

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.barh(labels[::-1], aucs[::-1], color=colors[::-1])
for bar, auc in zip(bars, aucs[::-1]):
    ax.text(auc + 0.003, bar.get_y() + bar.get_height() / 2, f"{auc:.3f}", va="center")
ax.set_xlim(0.5, 0.78)
ax.set_xlabel("ROC-AUC on held-out test set (0.5 = random guessing)")
ax.set_title("Will the user listen for more than 30 seconds?")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("docs/auc_progression.png", dpi=150)
