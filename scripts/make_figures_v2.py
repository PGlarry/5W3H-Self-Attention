"""
Multi-model figure set v2 — 三模型平权版
生成 Fig 1–5，输出到 IST_EXPERIMENT_ARCHIVE/figures/v2/
"""
import csv, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

# ── 路径 ──────────────────────────────────────────────────────────────────────
DATA  = "d:/pps/paperSELFatten/IST_EXPERIMENT_ARCHIVE/data/"
OUT   = "d:/pps/paperSELFatten/IST_EXPERIMENT_ARCHIVE/figures/v2/"
import os; os.makedirs(OUT, exist_ok=True)

# ── 加载数据 ──────────────────────────────────────────────────────────────────
attn = pd.read_csv(DATA + "stage7_attention_metrics_ALL3.csv", encoding="utf-8-sig")
attn["AS_ratio"]   = pd.to_numeric(attn["AS_ratio"],   errors="coerce")
attn["AE_norm"]    = pd.to_numeric(attn["AE_norm"],    errors="coerce")
attn["LP_private"] = pd.to_numeric(attn["LP_private"], errors="coerce")
attn["model_tag"]  = attn["model_tag"].fillna("qwen3_8b")

beh = pd.read_csv(DATA + "adjudicated_scores.csv", encoding="utf-8-sig")
beh["s_ICMw"] = pd.to_numeric(beh["s_ICMw"], errors="coerce")
beh["f_ICMw"] = pd.to_numeric(beh["f_ICMw"], errors="coerce")
beh["Split"]  = pd.to_numeric(beh["Split"],  errors="coerce")
beh["Ddrift"] = pd.to_numeric(beh["Ddrift"], errors="coerce")

# ── 颜色/样式 ─────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "qwen3_8b":   "#2196F3",   # blue
    "gemma3_4b":  "#4CAF50",   # green
    "ministral8b":"#FF9800",   # orange
}
MODEL_LABELS = {
    "qwen3_8b":   "Qwen3-8B",
    "gemma3_4b":  "Gemma3-4b",
    "ministral8b":"Ministral-8B",
}
COND_COLORS = {"FULL": "#1976D2", "PRIV-ABSENT": "#E53935", "PUB-ABSENT": "#757575"}
MODELS = ["qwen3_8b", "gemma3_4b", "ministral8b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

# ═════════════════════════════════════════════════════════════════════════════
# Fig 1: Multi-model mechanism overview (2×2)
# A: behavioral split  B: AS_ratio by model  C: M1-M4 β forest  D: mechanism schematic
# ═════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
fig1.suptitle("Figure 1: Multi-Model Attention-Anchor Effect Overview", fontsize=13, fontweight="bold")

# ── Panel A: Behavioral split ─────────────────────────────────────────────────
ax = axes[0, 0]
conds = ["FULL", "PUB-ABSENT", "PRIV-ABSENT"]
s_vals = [beh[beh["condition"]==c]["s_ICMw"].mean() for c in conds]
f_vals = [beh[beh["condition"]==c]["f_ICMw"].mean() for c in conds]
x = np.arange(len(conds)); w = 0.35
bars1 = ax.bar(x - w/2, s_vals, w, label="s_ICMw (structure)", color="#1976D2", alpha=0.85)
bars2 = ax.bar(x + w/2, f_vals, w, label="f_ICMw (fidelity)",  color="#E53935", alpha=0.85)
for bar, v in zip(bars2, f_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 1.12); ax.set_xticks(x); ax.set_xticklabels(conds, fontsize=9)
ax.set_ylabel("Score"); ax.set_title("A. Behavioral Structural-Fidelity Split\n(Stage 6, n=108)", fontsize=10)
ax.legend(fontsize=8); ax.axhline(1.0, color="gray", lw=0.5, ls="--")
split_val = beh[beh["condition"]=="PRIV-ABSENT"]["Split"].mean()
ax.text(2, 0.65, f"Split={split_val:.3f}", ha="center", fontsize=9, color="#E53935", fontweight="bold")

# ── Panel B: AS_ratio by model ────────────────────────────────────────────────
ax = axes[0, 1]
full_means  = [attn[(attn["model_tag"]==m) & (attn["condition"]=="FULL")]["AS_ratio"].mean() for m in MODELS]
priv_means  = [attn[(attn["model_tag"]==m) & (attn["condition"]=="PRIV-ABSENT")]["AS_ratio"].mean() for m in MODELS]
# combined
full_comb = attn[attn["condition"]=="FULL"]["AS_ratio"].mean()
priv_comb = attn[attn["condition"]=="PRIV-ABSENT"]["AS_ratio"].mean()
all_labels = [MODEL_LABELS[m] for m in MODELS] + ["Combined\n(n=36)"]
all_full   = full_means + [full_comb]
all_priv   = priv_means + [priv_comb]
x = np.arange(len(all_labels)); w = 0.35
for i, (m, c) in enumerate(zip(MODELS + ["combined"], ["#2196F3","#4CAF50","#FF9800","#9C27B0"])):
    ax.bar(i - w/2, all_full[i],  w, color=c, alpha=0.85, label="FULL" if i==0 else "")
    ax.bar(i + w/2, all_priv[i], w, color=c, alpha=0.45, hatch="//", label="PRIV-ABSENT" if i==0 else "")
# p-value annotations
p_vals = [0.026, 0.058, 0.005, 0.0001]
sigs   = ["*",   "†",   "**",  "***"]
for i, (pv, sg) in enumerate(zip(p_vals, sigs)):
    y_top = max(all_full[i], all_priv[i]) + 0.04
    ax.annotate("", xy=(i+w/2, y_top), xytext=(i-w/2, y_top),
                arrowprops=dict(arrowstyle="-", color="black", lw=1))
    ax.text(i, y_top+0.02, sg, ha="center", fontsize=10, color="black")
ax.axhline(1.0, color="gray", lw=0.5, ls="--")
ax.set_ylim(0, 2.1); ax.set_xticks(x); ax.set_xticklabels(all_labels, fontsize=8.5)
ax.set_ylabel("AS_ratio"); ax.set_title("B. AS_ratio by Model: FULL vs PRIV-ABSENT\n(Stage 7 attention, per-model n=12)", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="gray", alpha=0.85, label="FULL"),
                   Patch(color="gray", alpha=0.45, hatch="//", label="PRIV-ABSENT")],
          fontsize=8, loc="upper right")

# ── Panel C: Mixed-effects β forest plot ─────────────────────────────────────
ax = axes[1, 0]
specs  = ["M1\ncondition\n+(1|task)",
          "M2\n+length\ncovariates",
          "M3\n+model\nfixed FE",
          "M4\n+cond×model\ninteraction"]
betas  = [-0.316, -0.433, -0.491, -0.457]
ci_lo  = [-0.473, -0.665, -0.721, -0.767]
ci_hi  = [-0.158, -0.202, -0.261, -0.148]
y = np.arange(len(specs))
ax.axvline(0, color="black", lw=0.8, ls="--")
for i, (b, lo, hi) in enumerate(zip(betas, ci_lo, ci_hi)):
    ax.plot([lo, hi], [i, i], color="#1976D2", lw=2)
    ax.plot(b, i, "o", color="#1976D2", ms=8)
    ax.text(hi+0.01, i, f"β={b:.3f}", va="center", fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(specs, fontsize=8.5)
ax.set_xlabel("β (PRIV-ABSENT effect on AS_ratio)")
ax.set_title("C. Mixed-Effects Robustness: M1–M4\n(all p < 0.001, 95% CI shown)", fontsize=10)
ax.set_xlim(-0.9, 0.3)

# ── Panel D: Mechanism schematic ─────────────────────────────────────────────
ax = axes[1, 1]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
ax.set_title("D. Attention Anchor Hypothesis\n(schematic)", fontsize=10)
# Boxes
def box(ax, x, y, w, h, text, color="#1976D2", fontsize=9):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.2",
                                fc=color, alpha=0.18, ec=color, lw=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color, fontweight="bold")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

box(ax, 5, 9.0, 7,  0.9, "Private content present (FULL)\nAS_ratio = 1.443 > 1 ✓", "#1976D2")
arrow(ax, 5, 8.55, 5, 7.8)
box(ax, 5, 7.3, 7,  0.9, "Private replaced by generic (PRIV-ABSENT)\nAS_ratio drops to 1.127 (p<0.001)", "#E53935")
arrow(ax, 5, 6.85, 5, 6.1)
box(ax, 5, 5.6, 7,  0.9, "Attention redistributes toward\npublic/contextual scaffold", "#FF9800")
arrow(ax, 5, 5.15, 5, 4.4)
box(ax, 2.5, 3.9, 3.5, 0.9, "Structure preserved\ns_ICMw = 0.906", "#388E3C")
box(ax, 7.5, 3.9, 3.5, 0.9, "Fidelity drops\nf_ICMw = 0.600", "#E53935")
arrow(ax, 3.5, 5.15, 2.5, 4.35)
arrow(ax, 6.5, 5.15, 7.5, 4.35)
arrow(ax, 2.5, 3.45, 5, 2.7)
arrow(ax, 7.5, 3.45, 5, 2.7)
box(ax, 5, 2.2, 5, 0.9, "Split = s − f = 0.306", "#7B1FA2")

plt.tight_layout()
fig1.savefig(OUT + "fig1_multimodel_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Fig 1 saved")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 2: Per-task paired AS_ratio — 3 panels (one per model) + combined
# ═════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig2.suptitle("Figure 2: Per-Task AS_ratio Shifts (FULL vs PRIV-ABSENT) across Models",
              fontsize=13, fontweight="bold")

tasks_ordered = sorted(attn["task_id"].dropna().unique()) if "task_id" in attn.columns else []
if not tasks_ordered:
    attn["task_id"] = attn["sample_id"].str.split("__").str[0]
    tasks_ordered = sorted(attn["task_id"].unique())

for ax, m in zip(axes2, MODELS):
    df_m = attn[attn["model_tag"] == m]
    full_r = df_m[df_m["condition"]=="FULL"].set_index("task_id")["AS_ratio"]
    priv_r = df_m[df_m["condition"]=="PRIV-ABSENT"].set_index("task_id")["AS_ratio"]
    shared = sorted(set(full_r.index) & set(priv_r.index))
    ups, downs = 0, 0
    for tid in shared:
        fv, pv = full_r[tid], priv_r[tid]
        c = "#1976D2" if fv > pv else "#E53935"
        if fv > pv: ups += 1
        else: downs += 1
        ax.plot([0, 1], [fv, pv], "o-", color=c, alpha=0.65, lw=1.5, ms=5)
    # mean line
    ax.plot([0, 1], [full_r[shared].mean(), priv_r[shared].mean()],
            "s--", color="black", lw=2.5, ms=8, label="Mean", zorder=5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["FULL", "PRIV-ABSENT"], fontsize=10)
    ax.set_title(f"{MODEL_LABELS[m]}\n({ups}/{len(shared)} FULL > PRIV)", fontsize=10)
    ax.set_ylabel("AS_ratio" if m == "qwen3_8b" else "")
    ax.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax.set_ylim(0.3, 2.8)
    # sig annotation
    pvals = {"qwen3_8b": 0.026, "gemma3_4b": 0.058, "ministral8b": 0.005}
    sigs  = {"qwen3_8b": "*",   "gemma3_4b": "†",    "ministral8b": "**"}
    ax.text(0.5, 2.6, f"p = {pvals[m]:.3f} {sigs[m]}", ha="center", fontsize=10,
            color="black", bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

plt.tight_layout()
fig2.savefig(OUT + "fig2_pertask_AS_ratio_3models.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Fig 2 saved")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 3: Behavioral summary — Stage 6 (unchanged, verify numbers)
# ═════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8, 5))
fig3.suptitle("Figure 3: Behavioral Structural-Fidelity Split across Conditions\n(Stage 6: GPT-4o / DeepSeek-chat / Qwen3-8B, n=108)",
              fontsize=11, fontweight="bold")
conds3 = ["FULL", "PUB-ABSENT", "PRIV-ABSENT"]
metrics = {"s_ICMw": "#1976D2", "f_ICMw": "#E53935", "Ddrift": "#FF9800"}
x = np.arange(len(conds3)); w = 0.25
for j, (metric, color) in enumerate(metrics.items()):
    vals = [beh[beh["condition"]==c][metric].mean() for c in conds3]
    bars = ax3.bar(x + (j-1)*w, vals, w, label=metric, color=color, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha="center",
                 va="bottom", fontsize=7.5, rotation=0)
# Split annotation
split_vals = [beh[beh["condition"]==c]["Split"].mean() for c in conds3]
ax3.plot(x, split_vals, "D--", color="#7B1FA2", lw=2, ms=8, label="Split (s−f)", zorder=5)
for xi, sv, cond in zip(x, split_vals, conds3):
    if cond == "FULL": continue   # skip crowded FULL=0.003 label
    ax3.text(xi+0.15, sv+0.02, f"Split={sv:.3f}", fontsize=8.5, color="#7B1FA2", fontweight="bold")
ax3.set_ylim(0, 1.15); ax3.set_xticks(x); ax3.set_xticklabels(conds3, fontsize=11)
ax3.set_ylabel("Score"); ax3.legend(fontsize=9, loc="upper right")
ax3.axhline(1.0, color="gray", lw=0.5, ls="--")
plt.tight_layout()
fig3.savefig(OUT + "fig3_behavioral_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Fig 3 saved")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 4: Dimension-level attention redistribution — 3 models (3 panels)
# ═════════════════════════════════════════════════════════════════════════════
DIMS = ["WHAT","WHY","WHO","WHEN","WHERE","HOW_TO_DO","HOW_MUCH","HOW_FEEL"]
dim_cols_full = [f"AS_{d}" for d in DIMS]

fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig4.suptitle("Figure 4: Dimension-Level Attention Redistribution (FULL vs PRIV-ABSENT)",
              fontsize=12, fontweight="bold")

for ax, m in zip(axes4, MODELS):
    df_m = attn[attn["model_tag"] == m]
    full_d  = df_m[df_m["condition"]=="FULL"][dim_cols_full].mean()
    priv_d  = df_m[df_m["condition"]=="PRIV-ABSENT"][dim_cols_full].mean()
    delta_d = priv_d.values - full_d.values

    x = np.arange(len(DIMS)); w = 0.35
    ax.bar(x - w/2, full_d.values, w, label="FULL",        color="#1976D2", alpha=0.8)
    ax.bar(x + w/2, priv_d.values, w, label="PRIV-ABSENT", color="#E53935", alpha=0.8)

    # shade private dims
    for xi in [2, 6]:  # WHO, HOW_MUCH
        ax.axvspan(xi-0.5, xi+0.5, alpha=0.08, color="#7B1FA2", label="_nolegend_")

    ax2 = ax.twinx()
    ax2.bar(x, delta_d, 0.06, color=["#E53935" if d > 0 else "#1976D2" for d in delta_d],
            alpha=0.35, label="Δ (PRIV−FULL)")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_ylabel("Δ (PRIV−FULL)" if m == "ministral8b" else "", fontsize=8, color="gray")
    ax2.tick_params(labelsize=7, colors="gray")
    ax2.spines["right"].set_color("gray")

    ax.set_xticks(x); ax.set_xticklabels(DIMS, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{MODEL_LABELS[m]}", fontsize=10)
    ax.set_ylabel("Norm. attention" if m == "qwen3_8b" else "", fontsize=9)
    if m == "qwen3_8b":
        ax.legend(fontsize=8, loc="upper left")
    # label private dims
    ax.text(2, ax.get_ylim()[1]*0.95, "Private", ha="center", fontsize=7, color="#7B1FA2")
    ax.text(6, ax.get_ylim()[1]*0.95, "Private", ha="center", fontsize=7, color="#7B1FA2")

plt.tight_layout()
fig4.savefig(OUT + "fig4_dimension_attention_3models.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Fig 4 saved")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 5: IST attention-behavior chain — combined 3-model values
# ═════════════════════════════════════════════════════════════════════════════
fig5, ax5 = plt.subplots(figsize=(11, 4.5))
fig5.suptitle("Figure 5: IST Attention-Behavior Chain (Combined 3-Model)",
              fontsize=12, fontweight="bold")
ax5.axis("off")

full_ratio_comb  = attn[attn["condition"]=="FULL"]["AS_ratio"].mean()
priv_ratio_comb  = attn[attn["condition"]=="PRIV-ABSENT"]["AS_ratio"].mean()
full_f  = beh[beh["condition"]=="FULL"]["f_ICMw"].mean()
priv_f  = beh[beh["condition"]=="PRIV-ABSENT"]["f_ICMw"].mean()
full_s  = beh[beh["condition"]=="FULL"]["s_ICMw"].mean()
priv_s  = beh[beh["condition"]=="PRIV-ABSENT"]["s_ICMw"].mean()
full_sp = beh[beh["condition"]=="FULL"]["Split"].mean()
priv_sp = beh[beh["condition"]=="PRIV-ABSENT"]["Split"].mean()

# Two rows: FULL (top) and PRIV-ABSENT (bottom)
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, x, y, w, h, lines, color, fontsize=9.5):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15",
                                fc=color, alpha=0.20, ec=color, lw=2,
                                transform=ax.transAxes))
    text = "\n".join(lines)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=color, fontweight="bold", transform=ax.transAxes, linespacing=1.4)

def draw_arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                xycoords="axes fraction", textcoords="axes fraction")

def draw_label(ax, x, y, text, color="gray"):
    ax.text(x, y, text, ha="center", va="center", fontsize=8, color=color,
            transform=ax.transAxes, style="italic")

# Column positions
xs = [0.10, 0.30, 0.50, 0.70, 0.90]
FULL_Y, PRIV_Y = 0.72, 0.28
BW, BH = 0.16, 0.32

BLUE, RED, GREEN, PURPLE = "#1565C0", "#C62828", "#2E7D32", "#6A1B9A"

# FULL row
draw_box(ax5, xs[0], FULL_Y, BW, BH,
         ["Private dims", "present", "(FULL)"], BLUE)
draw_box(ax5, xs[1], FULL_Y, BW, BH,
         ["AS_ratio", f"= {full_ratio_comb:.3f}", "(> 1 ✓)"], BLUE)
draw_box(ax5, xs[2], FULL_Y, BW, BH,
         ["Relative attention", "anchored on", "private dims"], BLUE)
draw_box(ax5, xs[3], FULL_Y, BW, BH,
         ["s_ICMw", f"= {full_s:.3f}"], GREEN)
draw_box(ax5, xs[4], FULL_Y, BW, BH,
         ["f_ICMw", f"= {full_f:.3f}", f"Split={full_sp:.3f}"], GREEN)

# PRIV-ABSENT row
draw_box(ax5, xs[0], PRIV_Y, BW, BH,
         ["Private replaced", "by generic", "(PRIV-ABSENT)"], RED)
draw_box(ax5, xs[1], PRIV_Y, BW, BH,
         ["AS_ratio", f"= {priv_ratio_comb:.3f}", "(↓ p<0.001)"], RED)
draw_box(ax5, xs[2], PRIV_Y, BW, BH,
         ["Relative attention shifts", "to public scaffold", "(exploratory: AE_norm ↑)"], RED)
draw_box(ax5, xs[3], PRIV_Y, BW, BH,
         ["s_ICMw", f"= {priv_s:.3f}", "(≈ stable)"], "#388E3C")
draw_box(ax5, xs[4], PRIV_Y, BW, BH,
         ["f_ICMw", f"= {priv_f:.3f}", f"Split={priv_sp:.3f}↑"], RED)

# Arrows within rows
for i in range(len(xs)-1):
    draw_arrow(ax5, xs[i]+0.09, FULL_Y, xs[i+1]-0.09, FULL_Y, BLUE)
    draw_arrow(ax5, xs[i]+0.09, PRIV_Y, xs[i+1]-0.09, PRIV_Y, RED)

# Vertical change annotation
for xi in xs[1:]:
    ax5.annotate("", xy=(xi, PRIV_Y+0.17), xytext=(xi, FULL_Y-0.17),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, ls="dashed"),
                xycoords="axes fraction", textcoords="axes fraction")

# Row labels
ax5.text(-0.01, FULL_Y, "FULL", ha="right", va="center", fontsize=10,
         color=BLUE, fontweight="bold", transform=ax5.transAxes)
ax5.text(-0.01, PRIV_Y, "PRIV\nABSENT", ha="right", va="center", fontsize=9,
         color=RED, fontweight="bold", transform=ax5.transAxes)

# Column headers
headers = ["Condition", "Attention\nAnchor (AS_ratio)", "Attention\nDistribution", "Structure\n(s_ICMw)", "Fidelity\n(f_ICMw / Split)"]
for xi, h in zip(xs, headers):
    ax5.text(xi, 0.95, h, ha="center", va="center", fontsize=8.5,
             color="gray", transform=ax5.transAxes, fontweight="bold")

plt.tight_layout()
fig5.savefig(OUT + "fig5_IST_chain_multimodel.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("Fig 5 saved")

print(f"\n✅ 所有图已保存到 {OUT}")
print("文件列表：")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
