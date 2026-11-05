import sys
from pathlib import Path
PROJECT = Path("/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein")
sys.path.insert(0, str(PROJECT / "scripts/multi_protein"))
import generate_sept_2_publication_plots as g
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
rows = []
for fam in g.MODEL_ORDER:
    d = pd.read_csv(g.REGRESSION_ROOT / fam / "subgroup_metrics/by_spatial_distance.csv",
                    dtype={"category_label": str})
    assert set(d.support_status) == {"sufficient"} and len(d) == 48 * 5
    d["sum_err"] = d.pooled_bias_pred_minus_true * d.n_evaluated_pairs
    d["category_label"] = d.category_label.replace({"10-20": "10-40", "20-40": "10-40"})
    c = d.groupby(["layer", "category_label"], as_index=False)[["sum_err", "n_evaluated_pairs"]].sum()
    c["mean"] = c.sum_err / c.n_evaluated_pairs
    c["analysis_name"] = fam
    rows.append(c)
s = pd.concat(rows, ignore_index=True)
s["bootstrap_ci_low"] = s["bootstrap_ci_high"] = s["mean"]  # pooled values: no CI band
s.to_csv(OUT / "spatial_distance_pooled_bias_by_layer.csv", index=False)

g.configure_style(font_scale=g.MAIN_FIGURE_FONT_SCALE)
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9), sharex=True)
for ax, cat in zip(axes, g.SPATIAL_BINS):
    g.plot_model_lines(ax, s[s.category_label.eq(cat)], "unbounded")
    g.style_layer_axis(ax, "", "unbounded")
    ax.axvspan(22, 26, color="#999999", alpha=0.12, linewidth=0, zorder=0)
    ax.tick_params(axis="both", labelsize=g.MAIN_FACET_TICK_LABEL_SIZE)
    ax.set(xlabel="", ylabel="")
    ax.set_title(f"{cat} Å", pad=14)
h, l = axes[0].get_legend_handles_labels()
yl = fig.supylabel("Pooled bias,\npred − true (Å)", x=0.045, multialignment="center",
                   fontsize=g.MAIN_FACET_AXIS_LABEL_SIZE)
fig.supxlabel("Evoformer layer", y=0.20, fontsize=g.MAIN_FACET_AXIS_LABEL_SIZE)
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
fig.tight_layout(rect=(0.025, 0.16, 1, 0.98))
b = axes[0].get_position(); yl.set_y((b.y0 + b.y1) / 2)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"spatial_distance_pooled_bias.{ext}")

mid = s[s.layer.between(14, 34)].pivot_table(index="layer", columns=["category_label", "analysis_name"], values="mean")
pd.set_option("display.width", 250)
for cat in g.SPATIAL_BINS:
    t = mid[cat][list(g.MODEL_ORDER)].round(2)
    t.columns = [g.MODEL_SPECS[c]["label"] for c in t.columns]
    print(f"== {cat}\n{t.iloc[::2].to_string()}\n")
