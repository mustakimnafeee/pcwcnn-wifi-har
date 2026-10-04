"""
run.py  -  THE MAIN FILE: choose volunteers and tests, run, get results.

1. Pick your volunteers and tests in the SETTINGS block below.
2. Run this file (in VS Code: open it and press F5, or the "Run" triangle).
3. Results and plots are saved in  output/<name of the run>/

Two tests are available:

  "random" - the paper's method. All recordings of the chosen volunteers are mixed,
             80% train / 20% test. Every person appears in BOTH train and test.

  "loso"   - Leave-One-Subject-Out. For each volunteer: train on all the OTHERS,
             test on that one person, whom the model has never seen.
             This shows how well the model works on a brand-new person.

  Accuracy drop = (random-split accuracy on that volunteer) - (LOSO accuracy on them)

The same works for the DISTANCE / HEIGHT data (one person, recorded in different set-ups):
put condition names instead of volunteer names, e.g. VOLUNTEERS = ["1m", "3m", "6m"].
Then "loso" becomes leave-one-CONDITION-out: train on two distances, test on the third.

With several SEEDS, every test is repeated with a different random split / starting
weights, and the results are reported as  mean +/- standard deviation.

(paper_figures.py reuses the functions in this file to make the paper's figures.)
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from model import train_and_test
from preprocess import (ACTIVITIES, ALL_CONDITIONS, ALL_VOLUNTEERS, DISTANCES, HEIGHTS,
                        fit_pca_basis, load_volunteers, preprocess)

# =====================================================================================
#                                     SETTINGS
# =====================================================================================
VOLUNTEERS = ["A", "B", "F", "8", "9", "10"]   # any of: A B C D E F 7 8 9 10   (or ["all"])
#   ...or distance / height conditions: 1m 3m 6m  /  60cm 90cm 120cm
#   (or ["distance"] / ["height"] for all three). Don't mix volunteers and conditions.
TESTS = ["random", "loso"]                     # "random", "loso", or both
SEEDS = [0]                 # e.g. [0, 1, 2] repeats everything 3 times -> mean +/- std

PCS = [2, 3, 4, 5]          # principal components to use. Paper's exact choice: [2, 3]
SEGMENTATION = "none"       # "none" (works better) or "adaptive" (the paper's method)
EPOCHS = 30                 # paper uses 80. 30 is enough and ~3x faster
MODEL_WIDTH = 0.25          # 1.0 = paper's full-size network (slow). 0.25 = fast
AUGMENT = True              # light random shift/scale/noise while training
# =====================================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "volunteers_csi.npz")          # made by decode.py
FACTOR_DATA = os.path.join(HERE, "data", "factors_csi.npz")      # distance / height data


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def resolve_names(names):
    """Tidy up the names the user typed. Returns (names, kind).

    kind is "volunteer" (A, B, ..., 10) or "condition" (1m, 3m, 6m, 60cm, 90cm, 120cm).
    Shortcuts: "all" = all 10 volunteers, "distance" / "height" = those 3 conditions.
    """
    names = [str(n).strip() for n in names]
    low = [n.lower() for n in names]
    if low == ["all"]:
        return list(ALL_VOLUNTEERS), "volunteer"
    if low == ["distance"]:
        return list(DISTANCES), "condition"
    if low == ["height"]:
        return list(HEIGHTS), "condition"
    if all(n in ALL_CONDITIONS for n in low):
        return low, "condition"
    upper = [n.upper() for n in names]
    if all(n in ALL_VOLUNTEERS for n in upper):
        return upper, "volunteer"
    bad = [n for n, u, l in zip(names, upper, low)
           if u not in ALL_VOLUNTEERS and l not in ALL_CONDITIONS]
    if bad:
        raise SystemExit(f"Unknown name(s) {bad}. Volunteers: {ALL_VOLUNTEERS}. "
                         f"Conditions: {ALL_CONDITIONS}.")
    raise SystemExit("Don't mix volunteers and distance/height conditions in one run.")


def load_data(names):
    """Load the chosen volunteers or conditions (decoding the raw files first if needed)."""
    import decode
    if all(n in ALL_CONDITIONS for n in names):
        path, source = FACTOR_DATA, "raw_factors"
        if not os.path.exists(path):                  # first run: decode the raw .dat files
            decode.build_factors(out_file=path)
    else:
        path, source = DATA, "raw"
        if not os.path.exists(path):
            decode.build(out_file=path)
    traces, labels, who = load_volunteers(path, names)
    missing = [v for v in names if v not in set(who)]
    if missing:
        raise SystemExit(f"No recordings found for {missing}. Their raw folder was probably "
                         f"not included. Add it to {source}/, delete "
                         f"data/{os.path.basename(path)}, and run again.")
    return traces, labels, who


def random_split(labels, who, seed):
    """80/20 split, balanced so every (volunteer, activity) pair keeps 20% for testing."""
    strata = [f"{w}_{l}" for w, l in zip(who, labels)]
    return train_test_split(np.arange(len(labels)), test_size=0.2, random_state=seed,
                            stratify=strata)


def prepare(traces, tr, te, cfg):
    """PCA fitted on the training recordings only, then preprocess both sides."""
    basis = fit_pca_basis([traces[i] for i in tr])
    Xtr = preprocess([traces[i] for i in tr], basis, cfg["pcs"], cfg["segmentation"])
    Xte = preprocess([traces[i] for i in te], basis, cfg["pcs"], cfg["segmentation"])
    return Xtr, Xte


def run_random_split(traces, labels, who, cfg, seed):
    """Paper's test: mix everyone, 80/20 split, one PCWCNN."""
    tr, te = random_split(labels, who, seed)
    Xtr, Xte = prepare(traces, tr, te, cfg)
    pred, history, n_params = train_and_test(
        Xtr, labels[tr], Xte, labels[te], epochs=cfg["epochs"], width=cfg["width"],
        use_augment=cfg["augment"], seed=seed, label=f"[random seed {seed}]")
    per_vol = {v: accuracy(labels[te][who[te] == v], pred[who[te] == v])
               for v in np.unique(who)}
    return {"accuracy": accuracy(labels[te], pred),
            "macro_f1": float(f1_score(labels[te], pred, average="macro")),
            "per_volunteer": per_vol, "n_train": len(tr), "n_test": len(te),
            "n_params": n_params, "history": history,
            "y_true": labels[te].tolist(), "y_pred": pred.tolist()}


def run_loso(traces, labels, who, volunteers, cfg, seed, kind="volunteer"):
    """Leave-one-subject-out: train on everyone except v, test on v.
    (For conditions: train on all other conditions, test on condition v.)"""
    out = {}
    for k, v in enumerate(volunteers):
        tr = np.flatnonzero(who != v)
        te = np.flatnonzero(who == v)
        print(f"\n-- LOSO fold {k + 1}/{len(volunteers)} (seed {seed}): test on {kind} "
              f"{v} ({len(te)} recordings), train on the other {len(volunteers) - 1}")
        Xtr, Xte = prepare(traces, tr, te, cfg)
        pred, _, _ = train_and_test(
            Xtr, labels[tr], Xte, labels[te], epochs=cfg["epochs"], width=cfg["width"],
            use_augment=cfg["augment"], seed=seed, label=f"[loso {v}]")
        out[v] = {"accuracy": accuracy(labels[te], pred),
                  "macro_f1": float(f1_score(labels[te], pred, average="macro"))}
        print(f"   => {kind} {v}: LOSO accuracy {100 * out[v]['accuracy']:.1f}%")
    return out


def per_class_table(y_true, y_pred, paper=None):
    """Precision / recall / F1 for each of the 16 activities (like the paper's Table 2).

    paper: optional {"precision": [...16], "recall": [...], "f1": [...]} to show alongside.
    Returns (text table, dict of the numbers).
    """
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(16),
                                                 zero_division=0)
    head = f"{'#':>2}  {'Activity':<20}{'Precision':>10}{'Recall':>8}{'F1':>7}"
    if paper:
        head += "   | paper:  P     R    F1"
    lines = [head, "-" * len(head)]
    for i in range(16):
        row = f"{i + 1:>2}  {ACTIVITIES[i]:<20}{p[i]:>10.2f}{r[i]:>8.2f}{f[i]:>7.2f}"
        if paper:
            row += (f"   |      {paper['precision'][i]:>5.2f} {paper['recall'][i]:>5.2f} "
                    f"{paper['f1'][i]:>5.2f}")
        lines.append(row)
    row = f"    {'Macro average':<20}{p.mean():>10.2f}{r.mean():>8.2f}{f.mean():>7.2f}"
    if paper:
        row += (f"   |      {np.mean(paper['precision']):>5.2f} "
                f"{np.mean(paper['recall']):>5.2f} {np.mean(paper['f1']):>5.2f}")
    lines += ["-" * len(head), row]
    return "\n".join(lines), {"precision": p.tolist(), "recall": r.tolist(),
                              "f1": f.tolist()}


# ------------------------------------------------------------------------ plots
BLUE, ORANGE = "#2a78d6", "#eb6834"     # "ours" / "paper" (colour-blind-safe pair)
MUTED, GRID = "#898781", "#e1e0d9"


def style(ax):
    """Quiet axes: light horizontal grid, no top/right frame."""
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors="#444444")


def plot_confusion(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(16), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(16):
        for j in range(16):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cm[i, j] > 0.5 else "black")
    names = [f"{i + 1} {a}" for i, a in enumerate(ACTIVITIES)]
    ax.set_xticks(range(16), [str(i + 1) for i in range(16)])
    ax.set_yticks(range(16), names, fontsize=7)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(runs, path):
    """Training loss (left) and test accuracy (right) per epoch, one line per seed."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for r in runs:
        ep = [e["epoch"] for e in r["history"]]
        ax1.plot(ep, [e["train_loss"] for e in r["history"]], color=BLUE, lw=2, alpha=0.8)
        ax2.plot(ep, [100 * e["test_acc"] for e in r["history"]], color=BLUE, lw=2, alpha=0.8)
    ax1.set_title("Training loss")
    ax2.set_title("Test accuracy (%)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        style(ax)
    fig.suptitle(f"Training progress - random split ({len(runs)} seed(s), one line each)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loso(rnd, loso, volunteers, path, kind="volunteer"):
    """rnd / loso: {volunteer: (mean, std)} in %. rnd may be None."""
    what = "person" if kind == "volunteer" else "condition"
    seen = f"Random split ({what} seen in training)"
    unseen = f"{'LOSO' if kind == 'volunteer' else 'Leave-one-out'} ({what} never seen)"
    x = np.arange(len(volunteers))
    lo_m = [loso[v][0] for v in volunteers]
    lo_s = [loso[v][1] for v in volunteers]
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(volunteers) + 2), 5))
    if rnd:
        rs_m = [rnd[v][0] for v in volunteers]
        rs_s = [rnd[v][1] for v in volunteers]
        ax.bar(x - 0.2, rs_m, 0.4, yerr=rs_s, capsize=3,
               label=seen, color=BLUE)
        ax.bar(x + 0.2, lo_m, 0.4, yerr=lo_s, capsize=3,
               label=unseen, color=ORANGE)
        for i in range(len(x)):
            ax.text(x[i], max(rs_m[i], lo_m[i]) + 4, f"-{rs_m[i] - lo_m[i]:.0f}",
                    ha="center", fontsize=8, color=MUTED)
    else:
        ax.bar(x, lo_m, 0.5, yerr=lo_s, capsize=3, label=unseen, color=ORANGE)
    ax.axhline(100 / 16, color=MUTED, ls="--", lw=1, label="Chance (6.25%)")
    ax.set_xticks(x, volunteers)
    ax.set_xlabel(kind.title())
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_title(f"Accuracy per {kind} (numbers = accuracy drop, points)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3,
              frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Anything given here overrides SETTINGS.")
    ap.add_argument("--volunteers", nargs="+")
    ap.add_argument("--tests", nargs="+", choices=["random", "loso"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--width", type=float)
    ap.add_argument("--segmentation", choices=["none", "adaptive"])
    ap.add_argument("--pcs", nargs="+", type=int)
    ap.add_argument("--no-augment", action="store_true")
    a = ap.parse_args()

    volunteers = a.volunteers or VOLUNTEERS
    tests = a.tests or TESTS
    seeds = a.seeds if a.seeds is not None else SEEDS
    cfg = {"pcs": a.pcs or PCS, "segmentation": a.segmentation or SEGMENTATION,
           "epochs": a.epochs or EPOCHS, "width": a.width or MODEL_WIDTH,
           "augment": False if a.no_augment else AUGMENT}

    vols, kind = resolve_names(volunteers)
    if "loso" in tests and len(vols) < 2:
        raise SystemExit(f"LOSO needs at least 2 {kind}s.")

    # folder name records the settings, so ablation runs are easy to tell apart
    name = (f"{'-'.join(vols)}_pc{''.join(map(str, cfg['pcs']))}_seg-{cfg['segmentation']}"
            f"_aug{int(cfg['augment'])}_w{cfg['width']}_e{cfg['epochs']}"
            f"_seeds{''.join(map(str, seeds))}_{time.strftime('%Y%m%d-%H%M%S')}")
    out_dir = os.path.join(HERE, "output", name)
    os.makedirs(out_dir, exist_ok=True)

    settings = (f"PCs {cfg['pcs']}, segmentation {cfg['segmentation']}, {cfg['epochs']} "
                f"epochs, width {cfg['width']}, augment {cfg['augment']}, seeds {seeds}")
    print(f"{kind.title()}s: {vols}   tests: {tests}\n{settings}")
    traces, labels, who = load_data(vols)
    print(f"Loaded {len(traces)} recordings\n")

    t0 = time.time()
    results = {"settings": dict(cfg, volunteers=vols, tests=tests, seeds=seeds)}
    lines = [f"{kind.title()}s: {', '.join(vols)}   ({len(traces)} recordings)",
             f"Settings: {settings}", ""]
    rnd_vol = None

    if "random" in tests:
        print("== Random 80/20 split (the paper's method)")
        runs = [run_random_split(traces, labels, who, cfg, s) for s in seeds]
        results["random"] = runs
        # predictions of all seeds pooled -> one confusion matrix and one per-class table
        y_true = sum((r["y_true"] for r in runs), [])
        y_pred = sum((r["y_pred"] for r in runs), [])
        acc_m, acc_s = mean_std([100 * r["accuracy"] for r in runs])
        f1_m, f1_s = mean_std([100 * r["macro_f1"] for r in runs])
        rnd_vol = {v: mean_std([100 * r["per_volunteer"][v] for r in runs]) for v in vols}
        table, per_class = per_class_table(y_true, y_pred)
        results["random_summary"] = {"accuracy": [acc_m, acc_s], "macro_f1": [f1_m, f1_s],
                                     "per_class": per_class}
        plot_confusion(y_true, y_pred, os.path.join(out_dir, "confusion_random.png"),
                       f"Confusion matrix - random split (accuracy {acc_m:.1f}%)")
        plot_curves(runs, os.path.join(out_dir, "training_curve.png"))
        with open(os.path.join(out_dir, "per_class.txt"), "w") as f:
            f.write(f"Per-activity results, random split, {len(seeds)} seed(s) pooled\n\n"
                    + table + "\n")
        lines.append(f"Random split accuracy (all chosen {kind}s): {acc_m:.1f}% "
                     f"+/- {acc_s:.1f}   macro-F1 {f1_m:.1f}% +/- {f1_s:.1f}")
        lines.append("(per-activity precision / recall / F1: see per_class.txt)")
        print(f"   => overall accuracy {acc_m:.1f}% +/- {acc_s:.1f}")

    if "loso" in tests:
        print(f"\n== Leave-one-{'subject' if kind == 'volunteer' else 'condition'}-out")
        runs = [run_loso(traces, labels, who, vols, cfg, s, kind) for s in seeds]
        results["loso"] = runs
        loso_vol = {v: mean_std([100 * r[v]["accuracy"] for r in runs]) for v in vols}
        results["loso_summary"] = {v: list(loso_vol[v]) for v in vols}
        plot_loso(rnd_vol, loso_vol, vols, os.path.join(out_dir, f"loso_per_{kind}.png"), kind)
        left_out = "LOSO" if kind == "volunteer" else "Left out"
        lines += ["", f"{kind.title():<10}{'Random split':>16}{left_out:>16}{'Drop':>8}"]
        for v in vols:
            lo = f"{loso_vol[v][0]:.1f} +/- {loso_vol[v][1]:.1f}"
            if rnd_vol:
                rs = f"{rnd_vol[v][0]:.1f} +/- {rnd_vol[v][1]:.1f}"
                lines.append(f"{v:<10}{rs:>16}{lo:>16}{rnd_vol[v][0] - loso_vol[v][0]:>8.1f}")
            else:
                lines.append(f"{v:<10}{'-':>16}{lo:>16}{'-':>8}")
        mean_lo = np.mean([loso_vol[v][0] for v in vols])
        if rnd_vol:
            mean_rs = np.mean([rnd_vol[v][0] for v in vols])
            lines.append(f"{'MEAN':<10}{mean_rs:>16.1f}{mean_lo:>16.1f}"
                         f"{mean_rs - mean_lo:>8.1f}")
        else:
            lines.append(f"{'MEAN':<10}{'-':>16}{mean_lo:>16.1f}")
        lines.append("(accuracy in %, mean +/- std over seeds; chance level = 6.25%)")

    results["minutes"] = (time.time() - t0) / 60
    lines.append(f"\nTotal time: {results['minutes']:.1f} min")
    summary = "\n".join(lines)
    print("\n" + "=" * 64 + "\n" + summary + "\n" + "=" * 64)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved results and plots to: {out_dir}")


if __name__ == "__main__":
    main()
