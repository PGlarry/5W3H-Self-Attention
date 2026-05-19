"""
Fig 6: Layer-wise attention heatmap across three models
Answers: Does the attention-anchor effect concentrate in specific layer ranges?

Layout:
  Top: 3×2 heatmap grid (model × condition), dim on y, layer on x
  Bottom: 3-panel per-layer AS_ratio lines (FULL vs PRIV-ABSENT per model)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path("d:/pps/paperSELFatten/ist_attention_v1_runs")
FILES = {
    "Qwen3-8B":       BASE / "07_stage7"          / "stage7_layerwise_qwen3_8b.csv",
    "Gemma3-4b-it":   BASE / "07_stage7_gemma3_4b" / "stage7_layerwise_gemma3_4b.csv",
    "Ministral-8B":   BASE / "07_stage7_ministral8b"/ "stage7_layerwise_ministral8b.csv",
}
OUT_DIR = Path("d:/pps/paperSELFatten/IST_EXPERIMENT_ARCHIVE/figures/v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_DIMS   = ["WHAT","WHY","WHO","WHEN","WHERE","HOW_TO_DO","HOW_MUCH","HOW_FEEL"]
PRIV_DIMS  = ["WHO","HOW_MUCH"]
PUB_DIM    = "HOW_TO_DO"
CONDITIONS = ["FULL","PRIV-ABSENT"]

MODEL_COLORS = {
    "Qwen3-8B":     "#2E86AB",
    "Gemma3-4b-it": "#A23B72",
    "Ministral-8B": "#F18F01",
}

DIM_LABELS = {
    "WHAT":"WHAT","WHY":"WHY","WHO":"WHO","WHEN":"WHEN",
    "WHERE":"WHERE","HOW_TO_DO":"HOW_TO_DO","HOW_MUCH":"HOW_MUCH","HOW_FEEL":"HOW_FEEL",
}

def load_layerwise(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    layer_cols = [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]
    return df, layer_cols


def compute_mean_heatmap(df, layer_cols, condition):
    """Mean normalized attention per dim × layer for a given condition."""
    sub = df[df["condition"] == condition]
    result = {}
    for dim in ALL_DIMS:
        rows = sub[sub["dim"] == dim]
        if len(rows) == 0:
            result[dim] = np.zeros(len(layer_cols))
        else:
            result[dim] = rows[layer_cols].mean().values
    return result  # dict dim → array[n_layers]


def compute_per_layer_AS(df, layer_cols, condition):
    """Per-layer AS_ratio = mean(WHO, HOW_MUCH) / HOW_TO_DO across samples."""
    sub = df[df["condition"] == condition]
    # get per-sample per-layer values
    sample_ids = sub["sample_id"].unique()
    as_vals = []
    for sid in sample_ids:
        s = sub[sub["sample_id"] == sid]
        priv_rows = s[s["dim"].isin(PRIV_DIMS)]
        pub_rows  = s[s["dim"] == PUB_DIM]
        if len(priv_rows) == 0 or len(pub_rows) == 0:
            continue
        priv_mean = priv_rows[layer_cols].values.mean(axis=0)  # avg across priv dims
        pub_val   = pub_rows[layer_cols].values[0]
        denom = np.where(pub_val == 0, 1e-9, pub_val)
        as_vals.append(priv_mean / denom)
    if not as_vals:
        return np.ones(len(layer_cols))
    return np.mean(as_vals, axis=0)


def normalize_layers(arr, n_out=100):
    """Linearly interpolate arr to n_out points (normalizes different layer counts)."""
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n_out)
    return np.interp(x_new, x_old, arr)


def main():
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("white")

    # ── Grid: top 2 rows = heatmaps, bottom row = AS_ratio lines ────────────
    outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.42,
                              height_ratios=[2.0, 2.0, 1.6])

    # Row 0–1: heatmaps (3 models × 2 conditions)
    heat_gs = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[0:2],
                                               hspace=0.35, wspace=0.25)

    # Row 2: AS_ratio per layer
    line_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                               wspace=0.30)

    model_names = list(FILES.keys())
    N_NORM = 100  # normalize all models to 100 virtual layers

    all_heat = {}  # model → {cond → {dim → arr(100)}}
    all_as   = {}  # model → {cond → arr(100)}

    for mname, fpath in FILES.items():
        df, layer_cols = load_layerwise(fpath)
        all_heat[mname] = {}
        all_as[mname]   = {}
        for cond in CONDITIONS:
            hm = compute_mean_heatmap(df, layer_cols, cond)
            all_heat[mname][cond] = {d: normalize_layers(v, N_NORM) for d, v in hm.items()}
            all_as[mname][cond] = normalize_layers(
                compute_per_layer_AS(df, layer_cols, cond), N_NORM)

    # ── Heatmaps ────────────────────────────────────────────────────────────
    cond_labels = {"FULL": "FULL", "PRIV-ABSENT": "PRIV-ABSENT"}
    vmin, vmax = 0.0, 0.35  # shared color scale

    for ci, cond in enumerate(CONDITIONS):
        for mi, mname in enumerate(model_names):
            ax = fig.add_subplot(heat_gs[ci, mi])
            mat = np.array([all_heat[mname][cond][d] for d in ALL_DIMS])  # (8, 100)
            im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                           cmap="YlOrRd", vmin=vmin, vmax=vmax,
                           extent=[0, 100, -0.5, len(ALL_DIMS)-0.5],
                           origin="lower")
            ax.set_yticks(range(len(ALL_DIMS)))
            ax.set_yticklabels(ALL_DIMS, fontsize=7.5)
            ax.set_xlabel("Layer (%)", fontsize=8)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=7)

            color = MODEL_COLORS[mname]
            title = f"{mname}\n[{cond_labels[cond]}]"
            ax.set_title(title, fontsize=9, fontweight="bold", color=color, pad=4)

            # shared colorbar on rightmost column
            if mi == 2:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
                if ci == 0:
                    cbar.set_label("Norm. attention", fontsize=7.5)

            # Highlight private dims (WHO, HOW_MUCH) with bracket on y-axis
            priv_idx = [ALL_DIMS.index("WHO"), ALL_DIMS.index("HOW_MUCH")]
            for pi in priv_idx:
                ax.axhline(pi, color="royalblue", linewidth=0.6, linestyle="--", alpha=0.5)
            pub_idx = ALL_DIMS.index("HOW_TO_DO")
            ax.axhline(pub_idx, color="tomato", linewidth=0.6, linestyle="--", alpha=0.5)

    # ── Per-layer AS_ratio ───────────────────────────────────────────────────
    x = np.arange(N_NORM)
    cond_styles = {"FULL": ("-", 1.8), "PRIV-ABSENT": ("--", 1.5)}

    for mi, mname in enumerate(model_names):
        ax = fig.add_subplot(line_gs[0, mi])
        color = MODEL_COLORS[mname]
        for cond, (ls, lw) in cond_styles.items():
            ys = all_as[mname][cond]
            ax.plot(x, ys, color=color, linestyle=ls, linewidth=lw,
                    label=cond, alpha=0.9)
        # shade the difference
        y_full = all_as[mname]["FULL"]
        y_priv = all_as[mname]["PRIV-ABSENT"]
        ax.fill_between(x, y_priv, y_full,
                        where=(y_full > y_priv), color=color, alpha=0.12, label="_")
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_xlim(0, 99)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Layer (%)", fontsize=8)
        if mi == 0:
            ax.set_ylabel("AS_ratio (per layer)", fontsize=8)
        ax.set_title(mname, fontsize=9, fontweight="bold", color=color)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.6)
        ax.spines[["top","right"]].set_visible(False)

    # ── Global title ────────────────────────────────────────────────────────
    fig.suptitle(
        "Fig 6  |  Layer-wise attention profiles across three models\n"
        "Top: mean normalized attention per dimension (YlOrRd scale)  ·  "
        "Bottom: per-layer AS_ratio (private/public), FULL vs PRIV-ABSENT",
        fontsize=10, y=0.995, va="top"
    )

    out_path = OUT_DIR / "fig6_layerwise_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"✅ Fig 6 saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
