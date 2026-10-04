# Ablation guide

An **ablation** means removing or changing **one part** of the method and measuring how
much the accuracy changes. That tells you how much that part matters.

This guide shows how to do ablations with `run.py`, how to keep them valid, and how to
report them. Run every command below from inside the repository folder, in the VS Code
terminal. (Setup is in `README.md`.)

In the commands below, `python` stands for your environment's Python:

- Windows: `..\.venv\Scripts\python.exe`
- Mac / Linux: `../.venv/bin/python`

So `python run.py --seeds 0 1 2` means, on Windows,
`..\.venv\Scripts\python.exe run.py --seeds 0 1 2`.

> **Some ablations are already automated.** `paper_figures.py` runs the paper's own
> ablations: Fig. 8 (which PCs) and Fig. 9 (segmentation). Use this guide for those
> *and* for the extra ones the paper doesn't have (augmentation, network size, epochs),
> or to run any of them on different volunteers or under LOSO.

---

## 1. The five rules that make an ablation valid

1. **Fix a baseline first.** Pick one set of settings as the reference and write it down.
   Every variant is compared against it.
2. **Change exactly one thing per variant.** Keep the volunteers, test, epochs and
   everything else identical. Only then can the difference be blamed on that one setting.
3. **Use 3 seeds** (`--seeds 0 1 2`). `run.py` then repeats the run 3 times and reports
   **mean ± std**. A single run wobbles by about ±1–2 accuracy points, so a difference
   smaller than the std is not a real effect.
4. **Don't choose the "best" setting by test accuracy and then present it as your final
   result.** Comparing settings is fine; that's what an ablation is. But a number picked
   *because* it was the highest is slightly optimistic (about +1–2 points in this project).
   Say so if you do it: "best of N settings".
5. **State which test and which volunteers you used.** For example: "random split,
   volunteers A/B/C, 30 epochs, 3 seeds". An ablation under LOSO answers a different
   question (what helps with *new people*) and can give a different answer.

---

## 2. Recommended baseline

| Setting | Baseline value | Command-line flag |
|---|---|---|
| Volunteers | A, B, C (our best guess at the paper's Table 2 pool, see note) | `--volunteers A B C` |
| Test | random split | `--tests random` |
| Seeds | 0, 1, 2 | `--seeds 0 1 2` |
| Principal components | 2, 3, 4, 5 | `--pcs 2 3 4 5` |
| Segmentation | none | `--segmentation none` |
| Augmentation | on | *(default; `--no-augment` turns it off)* |
| Network width | 0.25 | `--width 0.25` |
| Epochs | 30 | `--epochs 30` |

Why A, B, C and random split: it is quick (about 1 minute per seed on a GPU), and accuracy
is high enough (about 90%) that differences show up clearly.

> **Note: the paper does not say which volunteers it used.** A, B, C is our deduction.
> The confusion matrix in the paper's Fig. 10 only contains fractions of 18 (e.g. 17/18 =
> 0.94), so there were 18 test recordings per activity. With a 20% test split that means
> 90 recordings per activity, 1440 in total, which is exactly 3 volunteers × 480. A, B and C
> are the three volunteers the WiAR dataset's own README describes. The paper's Table 3
> says "10 volunteers", which does not fit Fig. 10's numbers.

The baseline, one command (3 seeds):

```
python run.py --volunteers A B C --tests random --seeds 0 1 2
```

Every flag you don't give keeps its value from the SETTINGS block at the top of
`run.py`. To be safe, **don't edit the SETTINGS block during an ablation study**. Change
things only through the flags.

---

## 3. The ablations you can run

Each ablation below is **the baseline command plus one extra flag**.

### A. Which principal components matter? (paper Fig. 8)

| Variant | Extra flag |
|---|---|
| PC1 only | `--pcs 1` |
| PC1, 2 | `--pcs 1 2` |
| PC1, 2, 3 | `--pcs 1 2 3` |
| PC2, 3 (the paper's choice) | `--pcs 2 3` |
| PC3, 4 | `--pcs 3 4` |
| PC4, 5 | `--pcs 4 5` |
| PC2, 3, 4, 5 (baseline) | *(none)* |

Example: `python run.py --volunteers A B C --tests random --seeds 0 1 2 --pcs 2 3`

**What to expect:** the paper claims PC2,3 beats PC1, because PC1 mostly captures slow
background drift rather than body motion. In our experiments PC2,3 did beat PC1, as the paper reports.
(`paper_figures.py --figures fig8` does the paper's version of this: 1D-CNN, volunteer A.)

### B. Does segmentation help? (paper Fig. 9)

| Variant | Extra flag |
|---|---|
| No segmentation (baseline) | *(none)* |
| Adaptive segmentation (the paper's method) | `--segmentation adaptive` |

**What to expect:** the paper says adaptive segmentation helps (+11 points over a sliding
window). With this code it **did not help**: on volunteers A, B, C it gave 93.4% against
94.6% without segmentation, because it often cuts away part of the movement.

### C. Does augmentation help?

| Variant | Extra flag |
|---|---|
| With augmentation (baseline) | *(none)* |
| Without augmentation | `--no-augment` |

The paper does not use augmentation. This shows how much it adds here.

### D. Does network size matter?

| Variant | Extra flag | Parameters | Speed |
|---|---|---|---|
| Small (baseline) | *(none)* | 0.8 M | fast |
| Medium | `--width 0.5` | 3.3 M | ~2–3× slower |
| Paper's full size | `--width 1.0` | 13 M* | ~5–10× slower |

\* 13 M with 4 PCs. With the paper's 2 PCs (`--pcs 2 3`) it is exactly the paper's 9.6 M.

This checks that shrinking the network (to make it fast) did not cost much accuracy.

### E. Does longer training matter?

| Variant | Extra flag |
|---|---|
| 30 epochs (baseline) | *(none)* |
| 10 epochs | `--epochs 10` |
| 80 epochs (paper) | `--epochs 80` |

### F. The paper's exact method vs this version

Not a one-change ablation, but a useful summary row: everything set as the paper does it.

```
python run.py --volunteers A B C --tests random --seeds 0 1 2 --pcs 2 3 --segmentation adaptive --no-augment --width 1.0 --epochs 80
```

(Slow: use `--seeds 0` if time is short, and say so.)

### Not possible yet

These two are in the paper but have no switch in this code yet:

- **Without the wavelet (DWT) branches.** Tests whether the "Wavelet" in PCWCNN is worth it.
- **Without PCA / subcarrier fusion.** Uses all 90 raw streams (paper Fig. 7).

---

## 4. Ablation under LOSO (optional, slower)

Everything above can also be done with `--tests loso` (or `--tests random loso` for both).
This asks *"what helps the model work on people it has never seen?"*

```
python run.py --volunteers A B C --tests loso --seeds 0 1 2 --pcs 2 3
```

LOSO trains one model per volunteer per seed, so with 3 volunteers and 3 seeds that is 9
models. LOSO accuracies are low (about 10–20%, chance is 6.25%), so differences between
settings are small and noisy. Be careful about claiming effects.

---

## 4b. Ablation on the distance / height data (optional)

The same flags work with condition names instead of volunteers (see README section 7.4).
Example: does segmentation help at 3 m?

```
python run.py --volunteers 3m --tests random --seeds 0 1 2 --pcs 2 3
python run.py --volunteers 3m --tests random --seeds 0 1 2 --pcs 2 3 --segmentation adaptive
```

Use **one condition at a time** with `--tests random`, as the paper's Fig. 11 does. Each
distance has only 2 test recordings per activity, so results jump by ~3 points per
mistake: use at least 3 seeds (5 is better: `--seeds 0 1 2 3 4`), and only trust large
differences. The height conditions (480 recordings each) are more stable.

---

## 5. Where the results go

Each run saves a folder in `output/` whose name contains the settings, for example:

```
A-B-C_pc23_seg-none_aug1_w0.25_e30_seeds012_20260925-181137
│      │    │        │    │     │   │        └ date-time
│      │    │        │    │     │   └ seeds used
│      │    │        │    │     └ epochs
│      │    │        │    └ network width
│      │    │        └ augmentation on (1) / off (0)
│      │    └ segmentation
│      └ principal components (2 and 3)
└ volunteers
```

Open `summary.txt` inside it. The line to copy is:

```
Random split accuracy (all chosen volunteers): 90.6% +/- 1.0   macro-F1 90.5% +/- 1.1
```

That is the mean ± std over the seeds. `per_class.txt` has the per-activity breakdown,
which is useful to see *which* activities a change helps or hurts.

---

## 6. Recording and reporting

Copy this table into your notes and fill it in as runs finish:

| Ablation | Variant | Accuracy (mean ± std) | Δ vs baseline |
|---|---|---|---|
| — | **Baseline** (PC2-5, no seg, aug, w0.25, 30 ep) | | 0 |
| PCs | PC1 | | |
| PCs | PC1,2 | | |
| PCs | PC2,3 | | |
| PCs | PC3,4 | | |
| PCs | PC4,5 | | |
| Segmentation | adaptive | | |
| Augmentation | off | | |
| Width | 0.5 | | |
| Width | 1.0 | | |
| Epochs | 10 | | |
| Epochs | 80 | | |

**Δ vs baseline** = variant mean − baseline mean. A negative Δ means removing or changing
that part made things worse, so that part helps.

**How to read a Δ:**

- If |Δ| is **bigger** than both stds added together, it is a real effect: say "X improves
  accuracy by about Δ points".
- If |Δ| is **smaller**, there is no clear effect: say "no significant difference". Don't
  rank variants by it.

**One sentence to put in the report**, describing the method:

> Ablations were run on volunteers A, B and C with the random 80/20 split, changing one
> component at a time relative to the baseline configuration; each setting was repeated
> with three random seeds and we report the mean and standard deviation.

---

## 7. How long will it take?

On an NVIDIA GPU, one random-split run on A, B, C takes about a minute per seed (longer for
`--width 1.0` or `--epochs 80`). The full table above is 12 variants × 3 seeds = 36 runs,
roughly **45–60 minutes** in total. Section 3F adds a few more minutes per seed.

Short on time? Run `paper_figures.py --figures fig8 fig9` first (the paper's own
ablations), then sections C and D here.
