"""Generate paired before/after bar chart for compaction results (Fig 1 replacement)."""
import json, matplotlib, matplotlib.pyplot as plt, matplotlib.patches as mpatches
matplotlib.use("pdf")

def load(path):
    with open(path) as f:
        d = json.load(f)
    r = d["results"]
    def t(key):
        x = r[key]["timing"]
        return x["median_s"], x["min_s"], x["max_s"]
    return t("pre_compaction"), t("post_compaction")

BASE = "/Users/swalia/Desktop/Personal/Immig_US/IEEEBigData2026/results"
local_trials = [
    ("200-file\nSpark 3.5",
     load(f"{BASE}/iceberg_compaction_20260816T204818Z.json")),
    ("1000-file\nSpark 3.5",
     load(f"{BASE}/iceberg_compaction_20260816T231512Z.json")),
    ("200-file\nSpark 4.0",
     load(f"{BASE}/iceberg_compaction_spark4_20260817T201449Z.json")),
]
s3_trial = ("200-file\nS3/Spark 4.0",
            load(f"{BASE}/iceberg_s3_20260817T185910Z.json"))

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(3.4, 2.2),
                                  gridspec_kw={"width_ratios": [3, 1.1]})
fig.subplots_adjust(wspace=0.45, left=0.13, right=0.97, top=0.87, bottom=0.22)

C_BEFORE = "#555555"
C_AFTER  = "#2196F3"
BAR_W = 0.32
FONT = 7

def draw_panel(ax, trials, ylabel=True):
    for i, (label, (pre, post)) in enumerate(trials):
        x = i
        pre_med, pre_lo, pre_hi = pre
        post_med, post_lo, post_hi = post
        ax.bar(x - BAR_W/2, pre_med, BAR_W, color=C_BEFORE,
               yerr=[[pre_med - pre_lo], [pre_hi - pre_med]],
               error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#333"), zorder=3)
        ax.bar(x + BAR_W/2, post_med, BAR_W, color=C_AFTER,
               yerr=[[post_med - post_lo], [post_hi - post_med]],
               error_kw=dict(elinewidth=0.8, capsize=2, ecolor="#1565C0"), zorder=3)
        speedup = pre_med / post_med
        top = max(pre_hi, post_hi)
        ax.annotate(f"{speedup:.2f}×",
                    xy=(x, top), xytext=(x, top * 1.06),
                    ha="center", fontsize=FONT - 0.5, color="#111")
    ax.set_xticks(range(len(trials)))
    ax.set_xticklabels([t[0] for t in trials], fontsize=FONT - 0.5)
    ax.tick_params(axis="y", labelsize=FONT)
    if ylabel:
        ax.set_ylabel("Median scan latency (s)", fontsize=FONT)
    ax.yaxis.grid(True, linewidth=0.4, color="#ddd", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

draw_panel(ax_l, local_trials, ylabel=True)
ax_l.set_ylim(0, 1.18)
ax_l.set_title("Local filesystem", fontsize=FONT, pad=3)

draw_panel(ax_r, [s3_trial], ylabel=False)
ax_r.set_ylim(0, 11.5)
ax_r.set_title("Amazon S3", fontsize=FONT, pad=3)
ax_r.set_ylabel("Median scan latency (s)", fontsize=FONT)

before_p = mpatches.Patch(color=C_BEFORE, label="Before")
after_p  = mpatches.Patch(color=C_AFTER,  label="After")
fig.legend(handles=[before_p, after_p], loc="upper center",
           ncol=2, fontsize=FONT, frameon=False,
           bbox_to_anchor=(0.55, 1.0))

out = f"{BASE}/compaction_paired_bars.pdf"
fig.savefig(out, format="pdf", dpi=150)
print(f"Saved: {out}")
