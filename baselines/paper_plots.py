"""Paper-quality baseline comparison plots.

Reads existing benchmark data from baselines/figures/data/.
No rerunning needed — uses JSON files from previous benchmark runs.

Generates:
    baselines/figures/paper_pareto.{pdf,png}
    baselines/figures/paper_memory_vs_alpha.{pdf,png}
    baselines/figures/paper_table.tex

Usage:
    python baselines/paper_plots.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_dir = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_dir, "figures")
DATA_DIR = os.path.join(FIGURES_DIR, "data")

# ── Layout ────────────────────────────────────────────────────────────────────

DISTS = ["uniform_100", "zipfian", "unique"]
DIST_LABEL = {"uniform_100": "Uniform-100", "zipfian": "Zipfian", "unique": "Unique"}
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
TABLE_ALPHAS = [0.5, 0.8, 0.95]
N = 100_000

# ── Method styling (order = legend order) ─────────────────────────────────────

STYLES = {
    "csf_filter_optimal_binary_fuse": dict(
        label="Caramel", color="#2166ac", marker="o", lw=2.0, ms=7
    ),
    "csf_filter_shibuya_bloom": dict(
        label="Caramel (Shibuya)", color="#67a9cf", marker="^", lw=1.5, ms=6
    ),
    "java_csf": dict(
        label="Sux4J CSF", color="#1b7837", marker="s", lw=1.5, ms=6
    ),
    "lsf_ourcsf_filtered-huffmancsf_opt": dict(
        label="LSF CSF", color="#e08214", marker="X", lw=1.5, ms=6
    ),
    "lsf_burr_burr": dict(
        label="LSF BuRR", color="#d6604d", marker="D", lw=1.5, ms=6
    ),
    "java_mph": dict(
        label="Java MPH", color="#762a83", marker="P", lw=1.0, ms=5
    ),
    "cpp_hash_table": dict(
        label="C++ Hash Table", color="#878787", marker="d", lw=1.0, ms=5
    ),
}

# Subsets for different figures — drop non-competitive methods from Pareto
# (Java MPH and C++ Hash Table are in the table but distort the plot scale)
PARETO_METHODS = [
    "csf_filter_optimal_binary_fuse",
    "csf_filter_shibuya_bloom",
    "java_csf",
    "lsf_ourcsf_filtered-huffmancsf_opt",
    "lsf_burr_burr",
]
MEMORY_ALPHA_METHODS = [
    "csf_filter_optimal_binary_fuse",
    "csf_filter_shibuya_bloom",
    "java_csf",
    "lsf_ourcsf_filtered-huffmancsf_opt",
    "lsf_burr_burr",
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_experiments():
    """Load all individual experiment files. Returns dict[(dist, alpha)] -> data."""
    exps = {}
    for dist in DISTS:
        for alpha in ALPHAS:
            path = os.path.join(
                DATA_DIR, f"baselines_n{N}_a{alpha}_{dist}_binary_fuse.json"
            )
            if os.path.exists(path):
                with open(path) as f:
                    exps[(dist, alpha)] = json.load(f)
    return exps


def mem_bpk(result, n):
    """Extract memory in bits per key."""
    m = result["memory"]
    raw = m.get("theoretical") or m.get("serialized") or m.get("serialized_bytes")
    return raw * 8 / n


def lat_ns(result):
    return result["inference_ns"]["mean"]


def find(exp, method):
    return next((r for r in exp["results"] if r["method"] == method), None)


def _collect(exps, dist, method, alphas=ALPHAS):
    """Collect (alpha, memory_bpk, latency_ns) tuples for a method+dist."""
    out = []
    for a in alphas:
        e = exps.get((dist, a))
        if not e:
            continue
        r = find(e, method)
        if not r:
            continue
        out.append((a, mem_bpk(r, e["dataset"]["N"]), lat_ns(r)))
    return out


# ── Shared legend ─────────────────────────────────────────────────────────────


def _add_legend(fig, methods):
    handles = [
        Line2D(
            [0],
            [0],
            marker=STYLES[m]["marker"],
            color=STYLES[m]["color"],
            linewidth=STYLES[m]["lw"],
            markersize=5,
            label=STYLES[m]["label"],
        )
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 7),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        edgecolor="#cccccc",
    )


# ── Figure 1: Pareto (Memory vs Latency) ─────────────────────────────────────


def plot_pareto(exps):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, sharex=True)

    for i, dist in enumerate(DISTS):
        ax = axes[i]
        for mkey in PARETO_METHODS:
            s = STYLES[mkey]
            pts = _collect(exps, dist, mkey)
            if not pts:
                continue
            alps, ms, ls = zip(*pts)

            # Trajectory line
            if len(ms) > 1:
                ax.plot(ms, ls, color=s["color"], lw=s["lw"] * 0.7, alpha=0.5, zorder=2)

            # Points — larger at key alphas
            for a, m, l in zip(alps, ms, ls):
                sz = s["ms"] if a in TABLE_ALPHAS else s["ms"] * 0.6
                ax.scatter(
                    m,
                    l,
                    s=sz**2,
                    marker=s["marker"],
                    color=s["color"],
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )

            # Annotate alpha at endpoints for Caramel only (avoids clutter)
            if mkey == "csf_filter_optimal_binary_fuse" and len(alps) >= 2:
                ax.annotate(
                    f"$\\alpha$={alps[0]}",
                    (ms[0], ls[0]),
                    fontsize=6.5,
                    color=s["color"],
                    textcoords="offset points",
                    xytext=(5, 3),
                )
                ax.annotate(
                    f"$\\alpha$={alps[-1]}",
                    (ms[-1], ls[-1]),
                    fontsize=6.5,
                    color=s["color"],
                    textcoords="offset points",
                    xytext=(5, -2),
                )

        ax.set_xscale("log")
        ax.set_xlabel("Memory (bits/key)")
        ax.set_title(DIST_LABEL[dist])
        ax.grid(True, alpha=0.15, which="both")
        if i == 0:
            ax.set_ylabel("Avg. inference latency (ns)")

    _add_legend(fig, PARETO_METHODS)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── Figure 2: Memory vs Alpha ────────────────────────────────────────────────


def plot_memory_vs_alpha(exps):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)

    for i, dist in enumerate(DISTS):
        ax = axes[i]
        for mkey in MEMORY_ALPHA_METHODS:
            s = STYLES[mkey]
            pts = _collect(exps, dist, mkey)
            if not pts:
                continue
            alps, ms, _ = zip(*pts)
            ax.plot(
                alps,
                ms,
                color=s["color"],
                marker=s["marker"],
                markersize=s["ms"],
                linewidth=s["lw"],
            )

        ax.set_xlabel(r"$\alpha$ (majority fraction)")
        ax.set_title(DIST_LABEL[dist])
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.set_ylabel("Memory (bits/key)")

    _add_legend(fig, MEMORY_ALPHA_METHODS)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── Summary Table (LaTeX) ────────────────────────────────────────────────────


def generate_table(exps):
    """One table per distribution: methods × (alpha → memory, latency)."""
    lines = []
    for dist in DISTS:
        lines.append(f"\n% {DIST_LABEL[dist]}")
        lines.append(r"\begin{tabular}{l" + "rr" * len(TABLE_ALPHAS) + "}")
        lines.append(r"\toprule")

        h1 = "Method"
        for a in TABLE_ALPHAS:
            h1 += rf" & \multicolumn{{2}}{{c}}{{$\alpha\!=\!{a}$}}"
        lines.append(h1 + r" \\")

        h2 = ""
        for _ in TABLE_ALPHAS:
            h2 += r" & bits/key & ns"
        lines.append(h2 + r" \\")
        lines.append(r"\midrule")

        # Find best (lowest) per column for bolding
        best_mem, best_lat = {}, {}
        for a in TABLE_ALPHAS:
            e = exps.get((dist, a))
            if not e:
                continue
            mem_vals = [
                (mk, mem_bpk(r, e["dataset"]["N"]))
                for mk in STYLES
                if (r := find(e, mk)) is not None
            ]
            lat_vals = [
                (mk, lat_ns(r))
                for mk in STYLES
                if (r := find(e, mk)) is not None
            ]
            if mem_vals:
                best_mem[a] = min(mem_vals, key=lambda x: x[1])[0]
            if lat_vals:
                best_lat[a] = min(lat_vals, key=lambda x: x[1])[0]

        for mkey, s in STYLES.items():
            row = s["label"]
            for a in TABLE_ALPHAS:
                e = exps.get((dist, a))
                if not e:
                    row += " & --- & ---"
                    continue
                r = find(e, mkey)
                if not r:
                    row += " & --- & ---"
                    continue
                m = mem_bpk(r, e["dataset"]["N"])
                l = lat_ns(r)
                m_str = f"{m:.1f}"
                l_str = f"{l:.0f}"
                if best_mem.get(a) == mkey:
                    m_str = rf"\textbf{{{m_str}}}"
                if best_lat.get(a) == mkey:
                    l_str = rf"\textbf{{{l_str}}}"
                row += f" & {m_str} & {l_str}"
            lines.append(row + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    print("Loading data...")
    exps = load_experiments()
    print(f"  Loaded {len(exps)} experiments")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\nFigure 1: Pareto frontier...")
    fig = plot_pareto(exps)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"paper_pareto.{ext}")
        fig.savefig(path)
        print(f"  {path}")
    plt.close(fig)

    print("\nFigure 2: Memory vs alpha...")
    fig = plot_memory_vs_alpha(exps)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"paper_memory_vs_alpha.{ext}")
        fig.savefig(path)
        print(f"  {path}")
    plt.close(fig)

    print("\nSummary table (LaTeX)...")
    table = generate_table(exps)
    path = os.path.join(FIGURES_DIR, "paper_table.tex")
    with open(path, "w") as f:
        f.write(table)
    print(f"  {path}")
    print()
    print(table)


if __name__ == "__main__":
    main()
