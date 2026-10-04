"""
paper_figures.py  -  reproduces the paper's WiAR figures and tables, next to the paper's numbers.

    Fig. 7 + Table 1  reference models (SVM, RF, KNN, 1D-CNN) on one volunteer
    Fig. 8            which principal components matter (1D-CNN, one volunteer)
    Fig. 9            adaptive segmentation vs none (PCWCNN, three volunteers)
    Fig. 10 + Table 2 confusion matrix and per-activity results (PCWCNN, three volunteers)
    Table 3           all ten volunteers (PCWCNN)
    Fig. 11           distance (1/3/6 m) and height (60/90/120 cm): SVM, RF, CNN, PCWCNN

Run:   python paper_figures.py                   (everything, ~20-40 min on a GPU)
       python paper_figures.py --figures fig7 fig8
       python paper_figures.py --quick           (1 seed, few epochs: just checks it works)

Results go to  output/paper_figures_<date-time>/ : one PNG per figure, text tables,
COMPARISON.md (every number beside the paper's) and results.json.

Not included: Fig. 7's "all subcarriers" bars and Fig. 9's sliding-window method.
Fig. 11 needs the distance/height data (data/factors_csi.npz, or the raw files in
raw_factors/); without it that step is skipped.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import ALL_MODELS, run_reference_models
from preprocess import ALL_VOLUNTEERS, DISTANCES, HEIGHTS
from model import train_and_test
from run import (BLUE, MUTED, ORANGE, HERE, accuracy, load_data, mean_std, per_class_table,
                 plot_confusion, prepare, random_split, run_random_split, style)

# =====================================================================================
#                                     SETTINGS
# =====================================================================================
FIGURES = ["fig7", "fig8", "fig9", "fig10", "table3", "fig11"]   # which ones to make
SEEDS = [0, 1, 2]                   # every number is the mean over these seeds

ONE_VOLUNTEER = "A"                 # Fig. 7, Table 1, Fig. 8 ("one volunteer" in the paper)
THREE_VOLUNTEERS = ["A", "B", "C"]  # Fig. 9, Fig. 10, Table 2 (our best guess, see README)
FIG11_CONDITIONS = DISTANCES + HEIGHTS   # Fig. 11: each condition trained and tested alone

# Fig. 7 / 8 / 11 reference models use the paper's own pipeline: PC2,3 + adaptive segmentation
REFERENCE_PCS = [2, 3]
REFERENCE_SEGMENTATION = "adaptive"
CNN_EPOCHS = 80                     # the reference 1D-CNN is small, so the paper's 80 is fine

# PCWCNN settings for Fig. 9 / 10 / 11, Table 2 / 3 (same defaults as run.py)
PCWCNN = {"pcs": [2, 3, 4, 5], "segmentation": "none", "epochs": 30, "width": 0.25,
          "augment": True}
# =====================================================================================

PAPER = json.load(open(os.path.join(HERE, "paper_values.json")))
FIG8_GROUPS = {"a": [[1], [1, 2], [1, 2, 3], [2, 3]],
               "b": [[1, 2], [2, 3], [3, 4], [4, 5]],
               "c": [[2, 3], [1, 2, 3], [2, 3, 4], [2, 3, 5]]}


def tag(pcs):
    return "PC-" + ",".join(map(str, pcs))


def fmt(m_s):
    return f"{m_s[0]:.1f} +/- {m_s[1]:.1f}"


def bars(ax, labels, ours, paper, ylabel="Accuracy (%)"):
    """Grouped bars: ours (mean +/- std) and the paper's value, per label."""
    x = np.arange(len(labels))
    om = [o[0] for o in ours]
    os_ = [o[1] for o in ours]
    ax.bar(x - 0.2, om, 0.38, yerr=os_, capsize=3, color=BLUE, label="Reproduced (mean ± std)")
    pv = [np.nan if p is None else p for p in paper]
    ax.bar(x + 0.2, pv, 0.38, color=ORANGE, label="Paper")
    for i in range(len(x)):
        if not np.isnan(om[i]):
            ax.text(x[i] - 0.2, om[i] + os_[i] + 1.5, f"{om[i]:.0f}", ha="center", fontsize=8)
        if paper[i] is not None:
            ax.text(x[i] + 0.2, paper[i] + 1.5, f"{paper[i]:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 110)
    style(ax)


def legend_below(fig):
    """One legend under the plot(s), so it never covers a bar."""
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1))


# ------------------------------------------------------------------ Fig. 7 + Table 1
def fig7(out, seeds):
    print(f"\n=== Fig. 7 / Table 1: reference models on volunteer {ONE_VOLUNTEER}")
    traces, labels, who = load_data([ONE_VOLUNTEER])
    cfg = {"pcs": REFERENCE_PCS, "segmentation": REFERENCE_SEGMENTATION}
    accs = {m: [] for m in ALL_MODELS}
    pooled = {m: ([], []) for m in ALL_MODELS}
    for s in seeds:
        tr, te = random_split(labels, who, s)
        Xtr, Xte = prepare(traces, tr, te, cfg)
        preds = run_reference_models(Xtr, labels[tr], Xte, labels[te], seed=s,
                                     cnn_epochs=CNN_EPOCHS)
        for m, p in preds.items():
            accs[m].append(100 * accuracy(labels[te], p))
            pooled[m][0].extend(labels[te].tolist())
            pooled[m][1].extend(p.tolist())
        print(f"   seed {s}: " + "  ".join(f"{m} {accs[m][-1]:.1f}%" for m in ALL_MODELS))

    summary = {m: mean_std(accs[m]) for m in ALL_MODELS}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars(ax, ALL_MODELS, [summary[m] for m in ALL_MODELS],
         [PAPER["fig7_accuracy"][m] for m in ALL_MODELS])
    ax.set_title(f"Fig. 7 - reference models, one volunteer ({ONE_VOLUNTEER}), "
                 f"PC{''.join(map(str, REFERENCE_PCS))} + DWT")
    legend_below(fig)
    fig.savefig(os.path.join(out, "fig07_reference_models.png"), dpi=150)
    plt.close(fig)

    text = []
    for m in ALL_MODELS:
        t, _ = per_class_table(*pooled[m], paper=PAPER["table1_per_class"][m])
        text.append(f"=== {m}   (accuracy {fmt(summary[m])}%, paper "
                    f"{PAPER['fig7_accuracy'][m]}%)\n{t}\n")
    with open(os.path.join(out, "table1_per_class.txt"), "w") as f:
        f.write(f"Table 1 - per-activity results, one volunteer ({ONE_VOLUNTEER}), "
                f"{len(seeds)} seed(s) pooled\n\n" + "\n".join(text))
    rows = [(f"Fig. 7 {m}", summary[m], PAPER["fig7_accuracy"][m]) for m in ALL_MODELS]
    return {"accuracy": summary}, rows


# ------------------------------------------------------------------ Fig. 8
def fig8(out, seeds):
    print(f"\n=== Fig. 8: principal-component groupings, 1D-CNN, volunteer {ONE_VOLUNTEER}")
    traces, labels, who = load_data([ONE_VOLUNTEER])
    groups = []
    for g in FIG8_GROUPS.values():
        groups += [p for p in g if p not in groups]
    result = {}
    for pcs in groups:
        accs = []
        for s in seeds:
            tr, te = random_split(labels, who, s)
            Xtr, Xte = prepare(traces, tr, te,
                               {"pcs": pcs, "segmentation": REFERENCE_SEGMENTATION})
            p = run_reference_models(Xtr, labels[tr], Xte, labels[te], seed=s,
                                     models=["CNN"], cnn_epochs=CNN_EPOCHS)["CNN"]
            accs.append(100 * accuracy(labels[te], p))
        result[tag(pcs)] = mean_std(accs)
        print(f"   {tag(pcs):10s} {fmt(result[tag(pcs)])}%")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, (panel, g) in zip(axes, FIG8_GROUPS.items()):
        names = [tag(p) for p in g]
        bars(ax, names, [result[n] for n in names],
             [PAPER["fig8_accuracy"][panel][n] for n in names])
        ax.set_title(f"({panel})")
    fig.suptitle(f"Fig. 8 - principal-component groupings (1D-CNN, volunteer {ONE_VOLUNTEER})")
    legend_below(fig)
    fig.savefig(os.path.join(out, "fig08_pc_groupings.png"), dpi=150)
    plt.close(fig)
    paper_all = {k: v for d in PAPER["fig8_accuracy"].values() for k, v in d.items()}
    rows = [(f"Fig. 8 {n}", result[n], paper_all[n]) for n in result]
    return {"accuracy": result}, rows


# ------------------------------------------------------------------ PCWCNN runs (shared)
_cache = {}


def pcwcnn_runs(volunteers, segmentation, seeds):
    """Random-split PCWCNN runs, remembered so Fig. 9 and Fig. 10 can share them."""
    key = (tuple(volunteers), segmentation)
    if key not in _cache:
        traces, labels, who = load_data(volunteers)
        cfg = dict(PCWCNN, segmentation=segmentation)
        _cache[key] = [run_random_split(traces, labels, who, cfg, s) for s in seeds]
    return _cache[key]


# ------------------------------------------------------------------ Fig. 9
def fig9(out, seeds):
    print(f"\n=== Fig. 9: segmentation, PCWCNN, volunteers {THREE_VOLUNTEERS}")
    result = {}
    for seg in ("adaptive", "none"):
        runs = pcwcnn_runs(THREE_VOLUNTEERS, seg, seeds)
        result[seg] = mean_std([100 * r["accuracy"] for r in runs])
        print(f"   segmentation {seg:9s} {fmt(result[seg])}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars(ax, ["Sliding window", "Adaptive", "No segmentation"],
         [(np.nan, 0), result["adaptive"], result["none"]],
         [PAPER["fig9"]["sliding"]["accuracy"], PAPER["fig9"]["adaptive"]["accuracy"], None])
    ax.text(-0.2, 3, "not run", ha="center", va="bottom", fontsize=8, color=MUTED,
            rotation=90)
    ax.text(2.2, 3, "not in paper", ha="center", va="bottom", fontsize=8, color=MUTED,
            rotation=90)
    ax.set_title(f"Fig. 9 - segmentation (PCWCNN, volunteers {', '.join(THREE_VOLUNTEERS)})")
    legend_below(fig)
    fig.savefig(os.path.join(out, "fig09_segmentation.png"), dpi=150)
    plt.close(fig)
    rows = [("Fig. 9 adaptive", result["adaptive"], PAPER["fig9"]["adaptive"]["accuracy"]),
            ("Fig. 9 no segmentation", result["none"], None)]
    return {"accuracy": result}, rows


# ------------------------------------------------------------------ Fig. 10 + Table 2
def fig10(out, seeds):
    print(f"\n=== Fig. 10 / Table 2: PCWCNN, volunteers {THREE_VOLUNTEERS}")
    runs = pcwcnn_runs(THREE_VOLUNTEERS, PCWCNN["segmentation"], seeds)
    y_true = sum((r["y_true"] for r in runs), [])
    y_pred = sum((r["y_pred"] for r in runs), [])
    acc = mean_std([100 * r["accuracy"] for r in runs])
    plot_confusion(y_true, y_pred, os.path.join(out, "fig10_confusion.png"),
                   f"Fig. 10 - confusion matrix, PCWCNN, volunteers "
                   f"{', '.join(THREE_VOLUNTEERS)} (accuracy {acc[0]:.1f}%)")
    table, per_class = per_class_table(y_true, y_pred, paper=PAPER["table2_per_class"])
    with open(os.path.join(out, "table2_per_class.txt"), "w") as f:
        f.write(f"Table 2 - per-activity results, PCWCNN, volunteers "
                f"{', '.join(THREE_VOLUNTEERS)}, {len(seeds)} seed(s) pooled\n"
                f"Accuracy {fmt(acc)}%  (paper {PAPER['table2_accuracy']}%)\n\n{table}\n")

    # per-activity F1, reproduced vs paper
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(16)
    ax.bar(x - 0.2, per_class["f1"], 0.38, color=BLUE, label="Reproduced")
    ax.bar(x + 0.2, PAPER["table2_per_class"]["f1"], 0.38, color=ORANGE, label="Paper")
    ax.set_xticks(x, [str(i + 1) for i in range(16)])
    ax.set_xlabel("Activity (numbered as in the paper's Table 2)")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.1)
    ax.set_title("Table 2 - F1-score per activity")
    style(ax)
    legend_below(fig)
    fig.savefig(os.path.join(out, "table2_f1_per_activity.png"), dpi=150)
    plt.close(fig)
    rows = [("Table 2 accuracy", acc, PAPER["table2_accuracy"]),
            ("Table 2 macro F1", mean_std([100 * r["macro_f1"] for r in runs]),
             100 * np.mean(PAPER["table2_per_class"]["f1"]))]
    return {"accuracy": acc, "per_class": per_class}, rows


# ------------------------------------------------------------------ Table 3
def table3(out, seeds):
    print("\n=== Table 3: PCWCNN on all ten volunteers")
    runs = pcwcnn_runs(ALL_VOLUNTEERS, PCWCNN["segmentation"], seeds)
    acc = mean_std([100 * r["accuracy"] for r in runs])
    lines = [f"{'Approach':<28}{'Volunteers':>11}{'Activities':>11}{'Accuracy':>16}"]
    for name, v, a, p in PAPER["table3"]:
        lines.append(f"{name:<28}{v:>11}{a:>11}{p:>15.2f}%")
    lines.append(f"{'PCWCNN (reproduced)':<28}{10:>11}{16:>11}{fmt(acc):>15}%")
    with open(os.path.join(out, "table3.txt"), "w") as f:
        f.write("Table 3 - comparison of recognition accuracy\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
    return {"accuracy": acc}, [("Table 3 (10 volunteers)", acc, 95.5)]

# ------------------------------------------------------------------ Fig. 11
def fig11(out, seeds):
    """Each distance / height condition on its own: 80/20 split, 4 models, like the paper."""
    print(f"\n=== Fig. 11: distance and height, each condition trained and tested alone")
    models = ["SVM", "RF", "CNN", "PCWCNN"]
    result = {}
    for cond in FIG11_CONDITIONS:
        try:
            traces, labels, who = load_data([cond])
        except SystemExit as e:                       # no distance/height data available
            print(f"   skipping Fig. 11: {e}")
            return None, []
        accs = {m: [] for m in models}
        for s in seeds:
            tr, te = random_split(labels, who, s)
            # reference models: the paper's own pipeline (PC2,3 + adaptive segmentation)
            Xtr, Xte = prepare(traces, tr, te, {"pcs": REFERENCE_PCS,
                                                "segmentation": REFERENCE_SEGMENTATION})
            preds = run_reference_models(Xtr, labels[tr], Xte, labels[te], seed=s,
                                         models=["SVM", "RF", "CNN"], cnn_epochs=CNN_EPOCHS)
            # PCWCNN with the same settings as everywhere else
            Xtr, Xte = prepare(traces, tr, te, PCWCNN)
            preds["PCWCNN"], _, _ = train_and_test(
                Xtr, labels[tr], Xte, labels[te], epochs=PCWCNN["epochs"],
                width=PCWCNN["width"], use_augment=PCWCNN["augment"], seed=s, verbose=False)
            for m in models:
                accs[m].append(100 * accuracy(labels[te], preds[m]))
        result[cond] = {m: mean_std(accs[m]) for m in models}
        print(f"   {cond:6s} ({len(traces)} recordings): " +
              "  ".join(f"{m} {result[cond][m][0]:.1f}%" for m in models))

    # one small chart per condition: distances on top, heights below
    rows = [[c for c in DISTANCES if c in result], [c for c in HEIGHTS if c in result]]
    rows = [r for r in rows if r]
    fig, axes = plt.subplots(len(rows), 3, figsize=(14, 4.3 * len(rows)), squeeze=False)
    for i, conds in enumerate(rows):
        for j in range(3):
            ax = axes[i][j]
            if j >= len(conds):
                ax.axis("off")
                continue
            c = conds[j]
            bars(ax, models, [result[c][m] for m in models],
                 [PAPER["fig11_accuracy"][c][m] for m in models])
            ax.set_title(c)
    fig.suptitle("Fig. 11 - effect of distance (top) and height (bottom), "
                 "each condition trained and tested alone")
    legend_below(fig)
    fig.savefig(os.path.join(out, "fig11_distance_height.png"), dpi=150)
    plt.close(fig)
    rows_out = [(f"Fig. 11 {c} {m}", result[c][m], PAPER["fig11_accuracy"][c][m])
                for c in result for m in models]
    return {"accuracy": result}, rows_out


STEPS = {"fig7": fig7, "fig8": fig8, "fig9": fig9, "fig10": fig10, "table3": table3,
         "fig11": fig11}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", nargs="+", choices=list(STEPS))
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--quick", action="store_true",
                    help="1 seed and very few epochs - only to check that everything runs")
    a = ap.parse_args()
    global CNN_EPOCHS
    figures = a.figures or FIGURES
    seeds = a.seeds if a.seeds is not None else SEEDS
    if a.quick:
        seeds, CNN_EPOCHS, PCWCNN["epochs"] = [0], 3, 2

    out = os.path.join(HERE, "output", "paper_figures_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    print(f"Making {figures} with seeds {seeds}{'  (QUICK TEST)' if a.quick else ''}")
    print(f"Output: {out}")

    t0, results, rows = time.time(), {}, []
    for name in figures:
        res, r = STEPS[name](out, seeds)
        if res is None:
            continue
        results[name] = res
        rows += r
        with open(os.path.join(out, "results.json"), "w") as f:     # saved after each step
            json.dump(results, f, indent=1)

    md = ["# Reproduction vs paper", "",
          f"Seeds: {seeds}. Reproduced values are mean ± std over seeds (accuracy in %).",
          f"PCWCNN settings: {PCWCNN}.", "",
          "| Result | Reproduced | Paper | Difference |", "|---|---|---|---|"]
    for label, (m, s), p in rows:
        diff = "" if p is None else f"{m - p:+.1f}"
        md.append(f"| {label} | {m:.1f} ± {s:.1f} | {'-' if p is None else f'{p:g}'[:5]} "
                  f"| {diff} |")
    md += ["", f"Total time: {(time.time() - t0) / 60:.1f} min"]
    with open(os.path.join(out, "COMPARISON.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n" + "\n".join(md))
    print(f"\nAll figures and tables saved to: {out}")


if __name__ == "__main__":
    main()
