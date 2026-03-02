"""Paper-quality baseline comparison plots for AutoCSF (VLDB 2026).

Reads existing benchmark data from baselines/figures/data/.
No rerunning needed — uses JSON files from previous benchmark runs.

Generates:
    baselines/figures/paper_pareto.{pdf,png}
    baselines/figures/paper_memory_vs_alpha.{pdf,png}
    baselines/figures/paper_table.tex
    baselines/figures/paper_genomics_table.tex

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
HIGHLIGHT_ALPHAS = {0.5, 0.8, 0.95}
TABLE_ALPHAS = [0.5, 0.8, 0.95]
N = 100_000

GENOMICS_FILES = [
    ("baselines_ecoli_sakai_k15_binary_fuse.json", "E.\\ coli"),
    ("baselines_srr10211353_k15_binary_fuse.json", "SRR"),
    ("baselines_celegans_k15_binary_fuse.json", "C.\\ elegans"),
]

# ── Method styling (order = legend order) ─────────────────────────────────────

METHODS = [
    "csf_filter_optimal_binary_fuse",
    "csf_filter_shibuya_bloom",
    "java_csf",
    "java_mph",
    "lsf_ours_filtered-huffman_opt",
]

STYLES = {
    "csf_filter_optimal_binary_fuse": dict(
        label="AutoCSF", color="#2166ac", marker="o", lw=2.0, ms=7
    ),
    "csf_filter_shibuya_bloom": dict(
        label="BCSF (Shibuya)", color="#67a9cf", marker="^", lw=1.5, ms=6
    ),
    "java_csf": dict(
        label="Sux4J CSF", color="#1b7837", marker="s", lw=1.5, ms=6
    ),
    "java_mph": dict(
        label="MPH Table", color="#762a83", marker="P", lw=1.5, ms=6
    ),
    "lsf_ours_filtered-huffman_opt": dict(
        label="Learned CSF", color="#e08214", marker="X", lw=1.5, ms=6
    ),
}

# Sux4J CSF has a latency outlier at alpha=0.7 zipfian (1065ns vs ~356ns).
# Replace with interpolated value for plotting only.
LATENCY_OVERRIDES = {
    ("zipfian", 0.7, "java_csf"): 356.0,
}

# ── Updated Learned CSF numbers (from tables.md) ─────────────────────────────
# The JSON files contain older LCSF benchmarks. These overrides reflect the
# latest runs with improved hyperparameters. Only fields that changed
# meaningfully are patched; latency is unchanged for synthetic data.

_LCSF = "lsf_ours_filtered-huffman_opt"

# (dist, [bpk per alpha]) — memory overrides for LCSF
_LCSF_MEM_BPK = {
    "zipfian": [4.86, 4.00, 3.60, 2.39, 1.70, 1.40, 0.38],
    "unique":  [11.44, 10.44, 8.06, 6.94, 5.03, 3.54, 1.13],
}

# (dist, [ct per alpha]) — construction time overrides for LCSF
_LCSF_CT = {
    "unique": [983.0, 785.224, 589.425, 359.657, 184.074, 85.772, 21.813],
}

# New genomics LCSF result for SRR (was missing from JSON)
_LCSF_SRR = {
    "method": _LCSF,
    "construction_time_s": 501.4,
    "inference_ns": {"mean": 8312.0},
    "memory": {"serialized": int(3.49 * 9762863 / 8)},
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_synthetic():
    """Load all synthetic experiment files. Returns dict[(dist, alpha)] -> data."""
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


def _patch_lcsf(exps):
    """Patch loaded experiment data with updated Learned CSF numbers."""
    for dist, bpk_list in _LCSF_MEM_BPK.items():
        for i, alpha in enumerate(ALPHAS):
            e = exps.get((dist, alpha))
            if not e:
                continue
            r = find(e, _LCSF)
            if not r:
                continue
            n = e["dataset"]["N"]
            raw_bytes = int(bpk_list[i] * n / 8)
            if "serialized" in r["memory"]:
                r["memory"]["serialized"] = raw_bytes
            elif "serialized_bytes" in r["memory"]:
                r["memory"]["serialized_bytes"] = raw_bytes
            elif "theoretical" in r["memory"]:
                r["memory"]["theoretical"] = raw_bytes
    for dist, ct_list in _LCSF_CT.items():
        for i, alpha in enumerate(ALPHAS):
            e = exps.get((dist, alpha))
            if not e:
                continue
            r = find(e, _LCSF)
            if not r:
                continue
            r["construction_time_s"] = ct_list[i]


def _patch_genomics_lcsf(genomics):
    """Inject new Learned CSF result into SRR genomics data."""
    for _, g_data in genomics:
        if "srr" in g_data.get("dataset", {}).get("source", "").lower():
            if not find(g_data, _LCSF):
                g_data["results"].append(_LCSF_SRR)
            break
    else:
        # Find SRR by label
        for label, g_data in genomics:
            if "SRR" in label:
                existing = find(g_data, _LCSF)
                if not existing:
                    g_data["results"].append(_LCSF_SRR)
                break


def load_genomics():
    """Load genomics experiment files. Returns list of (label, data) tuples."""
    out = []
    for fname, label in GENOMICS_FILES:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                out.append((label, json.load(f)))
        else:
            print(f"  Warning: {fname} not found, skipping")
    return out


def mem_bpk(result, n):
    """Extract memory in bits per key."""
    m = result["memory"]
    raw = m.get("theoretical") or m.get("serialized") or m.get("serialized_bytes")
    return raw * 8 / n


def lat_ns(result, dist=None, alpha=None):
    """Extract inference latency in ns, applying overrides if needed."""
    key = (dist, alpha, result["method"])
    if key in LATENCY_OVERRIDES:
        return LATENCY_OVERRIDES[key]
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
        out.append((a, mem_bpk(r, e["dataset"]["N"]), lat_ns(r, dist, a)))
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
        if m in STYLES
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


# ── Figure 1: Pareto (Memory vs Latency) — 1×4 with genomics ────────────────


def plot_pareto(exps, genomics):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))

    # --- Synthetic subplots (1-3) ---
    for i, dist in enumerate(DISTS):
        ax = axes[i]
        for mkey in METHODS:
            s = STYLES[mkey]
            pts = _collect(exps, dist, mkey)
            if not pts:
                continue
            alps, ms, ls = zip(*pts)

            # Trajectory line
            if len(ms) > 1:
                ax.plot(
                    ms, ls,
                    color=s["color"], lw=s["lw"] * 0.7, alpha=0.5, zorder=2,
                )

            # Points — larger at highlight alphas
            for a, m, l in zip(alps, ms, ls):
                sz = s["ms"] if a in HIGHLIGHT_ALPHAS else s["ms"] * 0.6
                ax.scatter(
                    m, l,
                    s=sz**2, marker=s["marker"], color=s["color"],
                    edgecolors="white", linewidths=0.4, zorder=3,
                )

            # Annotate alpha at endpoints for AutoCSF only
            if mkey == "csf_filter_optimal_binary_fuse" and len(alps) >= 2:
                ax.annotate(
                    r"$\alpha\!=\!0.5$",
                    (ms[0], ls[0]),
                    fontsize=6, color=s["color"],
                    textcoords="offset points", xytext=(5, 3),
                )
                ax.annotate(
                    r"$\alpha\!=\!0.99$",
                    (ms[-1], ls[-1]),
                    fontsize=6, color=s["color"],
                    textcoords="offset points", xytext=(5, -2),
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Memory (bits/key)")
        ax.set_title(DIST_LABEL[dist])
        ax.grid(True, alpha=0.15, which="both")
        if i == 0:
            ax.set_ylabel("Avg. inference latency (ns)")

    # --- Genomics subplot (4th) ---
    ax = axes[3]
    for mkey in METHODS:
        s = STYLES[mkey]
        for g_label, g_data in genomics:
            r = find(g_data, mkey)
            if not r:
                continue
            n = g_data["dataset"]["N"]
            m = mem_bpk(r, n)
            l = lat_ns(r)
            ax.scatter(
                m, l,
                s=s["ms"] ** 2, marker=s["marker"], color=s["color"],
                edgecolors="white", linewidths=0.4, zorder=3,
            )
            # Label each genomics point with dataset name
            ax.annotate(
                f"${g_label}$",
                (m, l),
                fontsize=5.5,
                textcoords="offset points", xytext=(4, 2),
                color=s["color"], alpha=0.7,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Memory (bits/key)")
    ax.set_title("Genomics")
    ax.grid(True, alpha=0.15, which="both")

    _add_legend(fig, METHODS)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── Figure 2: Memory vs Alpha ────────────────────────────────────────────────


def plot_memory_vs_alpha(exps):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)

    for i, dist in enumerate(DISTS):
        ax = axes[i]
        for mkey in METHODS:
            s = STYLES[mkey]
            pts = _collect(exps, dist, mkey)
            if not pts:
                continue
            alps, ms, _ = zip(*pts)
            ax.plot(
                alps, ms,
                color=s["color"], marker=s["marker"],
                markersize=s["ms"], linewidth=s["lw"],
            )

        ax.set_xlabel(r"$\alpha$ (majority fraction)")
        ax.set_title(DIST_LABEL[dist])
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.set_ylabel("Memory (bits/key)")

    _add_legend(fig, METHODS)
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ── Table 1: Construction Time (LaTeX) ───────────────────────────────────────


def generate_table(exps):
    """Combined table: methods × (alpha → memory, latency, construction time)."""
    lines = []
    for dist in DISTS:
        lines.append(f"\n% {DIST_LABEL[dist]}")
        lines.append(r"\begin{tabular}{l" + "rrr" * len(TABLE_ALPHAS) + "}")
        lines.append(r"\toprule")

        h1 = "Method"
        for a in TABLE_ALPHAS:
            h1 += rf" & \multicolumn{{3}}{{c}}{{$\alpha\!=\!{a}$}}"
        lines.append(h1 + r" \\")

        h2 = ""
        for _ in TABLE_ALPHAS:
            h2 += r" & bpk & ns & build (s)"
        lines.append(h2 + r" \\")
        lines.append(r"\midrule")

        # Find best (lowest) per column for bolding
        best_mem, best_lat, best_ct = {}, {}, {}
        for a in TABLE_ALPHAS:
            e = exps.get((dist, a))
            if not e:
                continue
            mem_vals, lat_vals, ct_vals = [], [], []
            for mk in METHODS:
                r = find(e, mk)
                if r is None:
                    continue
                mem_vals.append((mk, mem_bpk(r, e["dataset"]["N"])))
                lat_vals.append((mk, lat_ns(r, dist, a)))
                ct_vals.append((mk, r["construction_time_s"]))
            if mem_vals:
                best_mem[a] = min(mem_vals, key=lambda x: x[1])[0]
            if lat_vals:
                best_lat[a] = min(lat_vals, key=lambda x: x[1])[0]
            if ct_vals:
                best_ct[a] = min(ct_vals, key=lambda x: x[1])[0]

        for mkey in METHODS:
            s = STYLES[mkey]
            row = s["label"]
            for a in TABLE_ALPHAS:
                e = exps.get((dist, a))
                if not e:
                    row += " & --- & --- & ---"
                    continue
                r = find(e, mkey)
                if not r:
                    row += " & --- & --- & ---"
                    continue
                m = mem_bpk(r, e["dataset"]["N"])
                l = lat_ns(r, dist, a)
                ct = r["construction_time_s"]
                m_str = f"{m:.1f}"
                l_str = f"{l:.0f}"
                ct_str = f"{ct:.3f}"
                if best_mem.get(a) == mkey:
                    m_str = rf"\textbf{{{m_str}}}"
                if best_lat.get(a) == mkey:
                    l_str = rf"\textbf{{{l_str}}}"
                if best_ct.get(a) == mkey:
                    ct_str = rf"\textbf{{{ct_str}}}"
                row += f" & {m_str} & {l_str} & {ct_str}"
            lines.append(row + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    return "\n".join(lines)


# ── Table 2: Genomics Summary (LaTeX) ────────────────────────────────────────


def generate_genomics_table(genomics):
    """Genomics summary: methods × datasets → memory, latency, construction time."""
    lines = []
    lines.append(r"\begin{tabular}{l" + "rrr" * len(genomics) + "}")
    lines.append(r"\toprule")

    h1 = "Method"
    for g_label, g_data in genomics:
        ds = g_data["dataset"]
        n_str = f"{ds['N']/1e6:.1f}M"
        alpha_str = f"{ds['alpha']:.2f}"
        h1 += rf" & \multicolumn{{3}}{{c}}{{{g_label} ($n$={n_str}, $\alpha$={alpha_str})}}"
    lines.append(h1 + r" \\")

    h2 = ""
    for _ in genomics:
        h2 += r" & bpk & ns & build (s)"
    lines.append(h2 + r" \\")
    lines.append(r"\midrule")

    # Find best per dataset-column
    best_mem, best_lat, best_ct = {}, {}, {}
    for gi, (g_label, g_data) in enumerate(genomics):
        n = g_data["dataset"]["N"]
        mem_vals, lat_vals, ct_vals = [], [], []
        for mk in METHODS:
            r = find(g_data, mk)
            if r is None:
                continue
            mem_vals.append((mk, mem_bpk(r, n)))
            lat_vals.append((mk, lat_ns(r)))
            ct_vals.append((mk, r["construction_time_s"]))
        if mem_vals:
            best_mem[gi] = min(mem_vals, key=lambda x: x[1])[0]
        if lat_vals:
            best_lat[gi] = min(lat_vals, key=lambda x: x[1])[0]
        if ct_vals:
            best_ct[gi] = min(ct_vals, key=lambda x: x[1])[0]

    for mkey in METHODS:
        s = STYLES[mkey]
        row = s["label"]
        for gi, (g_label, g_data) in enumerate(genomics):
            r = find(g_data, mkey)
            if not r:
                row += " & --- & --- & ---"
                continue
            n = g_data["dataset"]["N"]
            m = mem_bpk(r, n)
            l = lat_ns(r)
            ct = r["construction_time_s"]
            m_str = f"{m:.2f}"
            l_str = f"{l:.0f}"
            ct_str = f"{ct:.1f}"
            if best_mem.get(gi) == mkey:
                m_str = rf"\textbf{{{m_str}}}"
            if best_lat.get(gi) == mkey:
                l_str = rf"\textbf{{{l_str}}}"
            if best_ct.get(gi) == mkey:
                ct_str = rf"\textbf{{{ct_str}}}"
            row += f" & {m_str} & {l_str} & {ct_str}"
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
    exps = load_synthetic()
    print(f"  Loaded {len(exps)} synthetic experiments")
    _patch_lcsf(exps)
    print("  Patched Learned CSF with updated numbers")
    genomics = load_genomics()
    _patch_genomics_lcsf(genomics)
    print(f"  Loaded {len(genomics)} genomics datasets")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\nFigure 1: Pareto frontier (log-log, 1x4)...")
    fig = plot_pareto(exps, genomics)
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

    print("\nTable 1: Construction time (LaTeX)...")
    table = generate_table(exps)
    path = os.path.join(FIGURES_DIR, "paper_table.tex")
    with open(path, "w") as f:
        f.write(table)
    print(f"  {path}")
    print(table)

    print("\nTable 2: Genomics summary (LaTeX)...")
    g_table = generate_genomics_table(genomics)
    path = os.path.join(FIGURES_DIR, "paper_genomics_table.tex")
    with open(path, "w") as f:
        f.write(g_table)
    print(f"  {path}")
    print(g_table)


if __name__ == "__main__":
    main()
