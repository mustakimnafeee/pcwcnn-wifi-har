# PCWCNN: Wi-Fi CSI Human Activity Recognition on WiAR

A reproduction and extended evaluation of

> I. A. Showmik, T. F. Sanam and H. Imtiaz, **"Human Activity Recognition from Wi-Fi CSI data using
> Principal Component-based Wavelet CNN"**, *Digital Signal Processing*, vol. 138, 104056, 2023.

on the public **WiAR** dataset (16 activities, 10 volunteers, Intel 5300 Wi-Fi card).

The code rebuilds the paper's full pipeline (CSI decoding, PCA subcarrier fusion, Savitzky-Golay
smoothing, adaptive segmentation, discrete wavelet transform and the PCWCNN network), its reference
models (SVM, Random Forest, KNN and a 1D-CNN), and every WiAR figure and table of the paper. It also
adds two tests the paper does not run: **leave-one-subject-out (LOSO)**, which checks whether the
model works on a person it has never seen, and **leave-one-condition-out (LOCO)**, which does the same
for an unseen distance or receiver height.

This was our final project for **EEE 402 (Artificial Intelligence and Machine Learning Laboratory)**,
Department of EEE, BUET, January 2026 term (Section G2, Group 03).

---

## Contents

1. [Main results](#1-main-results)
2. [How the method works](#2-how-the-method-works)
3. [Files in this repository](#3-files-in-this-repository)
4. [Getting the data](#4-getting-the-data)
5. [Installation](#5-installation)
6. [Quick test](#6-quick-test)
7. [Running your own experiments (`run.py`)](#7-running-your-own-experiments-runpy)
8. [Reproducing the paper's figures (`paper_figures.py`)](#8-reproducing-the-papers-figures-paper_figurespy)
9. [Ablation studies](#9-ablation-studies)
10. [Differences from the paper](#10-differences-from-the-paper)
11. [Known limitations](#11-known-limitations)
12. [Running time](#12-running-time)
13. [Troubleshooting](#13-troubleshooting)
14. [References](#14-references)
15. [Authors](#15-authors)

---

## 1. Main results

All numbers are accuracy in %, as mean ± standard deviation over three random seeds unless stated
otherwise. With 16 activities, guessing gives 6.25%.

| Experiment | This code | Paper |
|---|---|---|
| Reference models on one volunteer (Fig. 7): SVM / RF / KNN / 1D-CNN | 89.9 ± 1.0 / 86.5 ± 4.7 / 78.1 ± 3.1 / 93.8 ± 0.9 | 88 / 85 / 83 / 96 |
| PCWCNN, volunteers A, B, C (Fig. 10, Table 2) | **94.6 ± 0.8** | 95.5 |
| PCWCNN, all 10 volunteers (Table 3) | 72.9 ± 0.7 | 95.5 |
| Adaptive segmentation vs none, A, B, C (Fig. 9) | 93.4 ± 1.0 vs 94.6 ± 0.8 | 94 (adaptive) |
| **Leave-one-subject-out**, 10 volunteers | **13.7** on average (9-20 per person) | not tested |
| **Leave-one-condition-out**, distance / height | **25.8** / **8.8** | not tested |

What this means:

- The pipeline reproduces the paper closely when every person appears in both the training and the
  test data (the paper's random 80/20 split).
- Adaptive segmentation, one of the paper's main components, did not improve accuracy.
- The model learns person- and set-up-specific patterns: on someone it has never seen, or at a
  distance/height it was not trained on, accuracy falls close to guessing. The paper's evaluation
  does not reveal this.

Ablation study (volunteers A, B, F, 8, 9, 10, random split; one setting changed at a time):

| Setting | Accuracy |
|---|---|
| Baseline: PC2-5, no segmentation, width 0.25, augmentation on, 80 epochs (3 seeds) | 92.3 ± 1.5 |
| Augmentation off (1 seed) | 88.5 |
| Full-width network, 13.0 M parameters (1 seed) | 92.6 |
| Only PC2 and PC3, as in the paper (1 seed) | 86.5 |
| Adaptive segmentation (1 seed) | 92.0 |

---

## 2. How the method works

```
raw .dat files  ->  CSI amplitude in dB (T packets x 90 streams = 3 antennas x 30 subcarriers)
                ->  remove the static background (subtract each stream's mean)
                ->  PCA subcarrier fusion with ONE basis shared by all recordings
                ->  keep principal components 2-5 (the paper keeps 2 and 3)
                ->  Savitzky-Golay smoothing (window 5, order 2)
                ->  optional adaptive segmentation (the paper's method; off by default)
                ->  resample every recording to 384 samples
                ->  PCWCNN: 3-level Haar wavelet branches + 4 convolution stages -> 16 activities
```

PCWCNN follows Fig. 6 of the paper. The main path has four stages, each a 3x1 convolution followed
by a stride-2 convolution, so the signal length halves at every stage (384 -> 192 -> 96 -> 48 -> 24).
In parallel, a Haar wavelet transform halves the input three times, and each level passes through a
small convolution branch before being concatenated with the main path at the matching length. Global
average pooling and one fully connected layer give the 16 class scores.

A `width` factor scales all channel counts. Width 1.0 is the paper's size (9.6 M parameters with two
PCs); the default width 0.25 has 0.82 M parameters, trains several times faster and is just as
accurate.

---

## 3. Files in this repository

| File | What it does |
|---|---|
| `run.py` | **Main script.** Choose volunteers (or distance/height conditions), run the random split and/or LOSO/LOCO, and get tables and plots. |
| `paper_figures.py` | Runs the experiments behind the paper's Figs. 7-11 and Tables 1-3 and plots each one next to the paper's values. |
| `model.py` | The PCWCNN network, data augmentation and the training/testing loop. |
| `preprocess.py` | Static removal, shared PCA basis, Savitzky-Golay smoothing, adaptive segmentation, resampling. |
| `baselines.py` | The paper's reference models: SVM, Random Forest, KNN and the 1D-CNN, with their wavelet features. |
| `decode.py` | Reads the raw Intel 5300 `.dat` files. Runs automatically the first time. |
| `paper_values.json` | The numbers published in the paper, used for the side-by-side comparisons. |
| `requirements.txt` | Python packages (PyTorch is installed separately, see section 5). |
| `ABLATION_GUIDE.md` | How to run valid ablation studies with `run.py`. |

The folders `data/` (decoded recordings) and `output/` (results) are created automatically when you
run the code. They are not part of the repository.

---

## 4. Getting the data

The WiAR dataset is published by its authors at **<https://github.com/linteresa/WiAR>**
(L. Guo et al., IEEE Access, 2019). It is not included here. Put the dataset folders **next to** this
repository's folder, not inside it:

```
my_project/
├── pcwcnn-wifi-har/            <- this repository (any folder name works)
│   ├── run.py
│   └── ...
├── raw/                        <- WiAR volunteer recordings (required)
│   ├── volunteer_a_all_data/   (csi_a1_1.dat, csi_a1_2.dat, ...)
│   ├── volunteer_b_all_data/
│   ├── volunteer_c_all_data/
│   ├── volunteer_f_all_data/
│   ├── volunteerD_1/  volunteerD_2/
│   ├── volunteerE_1/  volunteerE_2/
│   └── volunteer_7/  volunteer_8/  volunteer_9/  volunteer_10/
└── raw_factors/                <- distance / height recordings (optional)
    ├── distance_factor_activity_data/
    │   ├── 1_alldata/   3_alldata/   6_alldata/
    └── height_factor_activity_data/
        ├── 60_rawdata/  90_rawdata/  120_rawdata/
```

- Keep the folder names inside `raw/` exactly as shown. You only need the volunteers you want to use.
- Inside `raw_factors/`, the code searches every sub-folder for `1_alldata`, `3_alldata`, `6_alldata`,
  `60_rawdata`, `90_rawdata`, `120_rawdata` (the dataset's own names) or simply `1m`, `3m`, `6m`,
  `60cm`, `90cm`, `120cm`.
- `raw_factors/` is only needed for the distance/height experiments (Fig. 11 and LOCO).
- If your data is somewhere else, change `RAW_DIR` and `FACTOR_DIR` at the top of `decode.py`.

On the first run, `decode.py` converts the raw files into `data/volunteers_csi.npz` and
`data/factors_csi.npz` (about 2 minutes, once). After that the raw folders are no longer read.

---

## 5. Installation

You need **Python 3.10, 3.11 or 3.12** (3.12 recommended). An NVIDIA GPU is optional; it makes
training much faster, but everything also runs on a CPU. The commands below create a virtual
environment in `my_project/`, next to the repository.

1. **Download the code.** Either click **Code → Download ZIP** on GitHub and unzip it inside
   `my_project/`, or clone it:

   ```
   cd my_project
   git clone https://github.com/mustakimnafeee/pcwcnn-wifi-har.git
   ```

   A downloaded ZIP unpacks to a folder called `pcwcnn-wifi-har-main`. Rename it to
   `pcwcnn-wifi-har`, or use that name in the commands below.

2. **Create a virtual environment** (from `my_project/`):

   | System | Command |
   |---|---|
   | Windows | `py -3.12 -m venv .venv` |
   | macOS / Linux | `python3 -m venv .venv` |

   On Windows, if `py -3.12` is not found, run `py -0` to see your installed versions and use one of
   them (3.10-3.12).

3. **Install PyTorch.** Pick one:

   | Your machine | Command |
   |---|---|
   | Windows / Linux with an NVIDIA GPU | `.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126` |
   | Windows, no GPU | `.venv\Scripts\python.exe -m pip install torch` |
   | macOS, or Linux without a GPU | `.venv/bin/python -m pip install torch` |

   On Linux, write `.venv/bin/python` instead of `.venv\Scripts\python.exe`.

4. **Install the other packages:**

   ```
   .venv\Scripts\python.exe -m pip install -r pcwcnn-wifi-har\requirements.txt
   ```

   (macOS / Linux: `.venv/bin/python -m pip install -r pcwcnn-wifi-har/requirements.txt`)

5. *(Optional)* **Check that the GPU is found:**

   ```
   .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
   ```

   `True` means training will use the GPU.

**In the rest of this guide, `python` means the environment's Python**, run from inside the
repository folder: `..\.venv\Scripts\python.exe` on Windows, `../.venv/bin/python` on macOS/Linux.
You do not need to activate the environment.

If you use VS Code: open `my_project` with **File → Open Folder**, press **Ctrl+Shift+P**, choose
**Python: Select Interpreter** and pick the one in `.venv`. You can then run any script with **F5**.

---

## 6. Quick test

```
cd pcwcnn-wifi-har
python run.py --volunteers A B --epochs 2
```

The first time, you will see `Decoding ... raw recordings` (about 2 minutes, only once). The run ends
with a small results table and `Saved results and plots to: ...`. With only 2 epochs the numbers are
lower than in a full run; this test only checks that everything works.

---

## 7. Running your own experiments (`run.py`)

### 7.1 Settings

Open `run.py`. At the top is the **SETTINGS** block:

```python
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
```

Change what you need, save, and run the file. Upper or lower case both work (`"a"` = `"A"`).

| You want | Set |
|---|---|
| all ten volunteers | `VOLUNTEERS = ["all"]` |
| the distance data | `VOLUNTEERS = ["distance"]` or e.g. `["3m"]` |
| three volunteers, like the paper's Table 2 | `VOLUNTEERS = ["A", "B", "C"]` |
| only the leave-one-out test | `TESTS = ["loso"]` |
| more reliable numbers (mean ± std) | `SEEDS = [0, 1, 2]` (3x the time) |
| the paper's exact preprocessing | `PCS = [2, 3]` and `SEGMENTATION = "adaptive"` |
| the paper's full-size network | `MODEL_WIDTH = 1.0` and `EPOCHS = 80` (much slower) |

### 7.2 Or use command-line flags

Anything given on the command line overrides SETTINGS:

```
python run.py --volunteers A B F 8 9 10
python run.py --volunteers all --tests loso --seeds 0 1 2
python run.py --volunteers A B C --tests random --pcs 2 3 --segmentation adaptive
python run.py --volunteers A B C --epochs 80 --width 1.0 --no-augment
```

All flags: `--volunteers`, `--tests`, `--seeds`, `--pcs`, `--segmentation`, `--epochs`, `--width`,
`--no-augment`.

### 7.3 What the two tests mean

- **Random split** (the paper's method): all recordings of the chosen volunteers are pooled and split
  80/20, so that every person keeps 20% of their own recordings for testing. One model is trained.
  The model has seen *other* recordings of each test person during training.
- **LOSO** (leave-one-subject-out): for each volunteer, a new model is trained on all the *other*
  volunteers and tested on every recording of that volunteer. The model has never seen that person.
- **Drop** = random-split accuracy - LOSO accuracy, for the same person: how much worse the model does
  on someone new.
- **Seeds**: each seed gives a different random split and different starting weights. With several
  seeds, every number is reported as mean ± standard deviation.

### 7.4 Distance and height data

WiAR also has recordings of one person doing the same 16 activities in different set-ups:

| Condition names | What changes | Recordings |
|---|---|---|
| `1m`, `3m`, `6m` | distance between the Wi-Fi sender and receiver | 160 each (10 per activity) |
| `60cm`, `90cm`, `120cm` | height of the receiver above the floor | 480 each (30 per activity) |

Use the condition names in place of volunteer names:

```
python run.py --volunteers 3m --tests random          # one condition alone (as in the paper's Fig. 11)
python run.py --volunteers distance --seeds 0 1 2     # all distances: random split + leave-one-condition-out
python run.py --volunteers height --seeds 0 1 2       # all heights
```

With conditions, `loso` becomes **leave-one-condition-out**: train on two distances (or heights) and
test on the third, a set-up the model has never seen. Each distance has only 2 test recordings per
activity, so one mistake changes accuracy by about 3 points; always use several seeds.

### 7.5 Output

Each run creates its own folder in `output/`, named after the volunteers, settings and date-time
(e.g. `A-B-C_pc2345_seg-none_aug1_w0.25_e30_seeds012_20260925-181137`), so old runs are never
overwritten.

| File | What it shows |
|---|---|
| `summary.txt` | Overall accuracy and macro-F1; per volunteer: random split, LOSO and the drop |
| `per_class.txt` | Precision / recall / F1 for each of the 16 activities |
| `confusion_random.png` | Which activities are confused with which |
| `training_curve.png` | Training loss and test accuracy per epoch |
| `loso_per_volunteer.png` | Random split vs LOSO per volunteer (`loso_per_condition.png` for distance/height) |
| `results.json` | Every number, for later use |

The test accuracy in `training_curve.png` is only for monitoring: the reported result always uses the
model after the last epoch, never the best epoch on the test set.

---

## 8. Reproducing the paper's figures (`paper_figures.py`)

```
python paper_figures.py                        # everything (about 30-50 min on a GPU)
python paper_figures.py --figures fig7 fig8    # only some
python paper_figures.py --quick                # 1 seed, few epochs: just checks that it works
```

| Paper result | What is run | Output files |
|---|---|---|
| Fig. 7 + Table 1 | SVM, RF, KNN and the 1D-CNN on volunteer A (PC2,3 + adaptive segmentation) | `fig07_reference_models.png`, `table1_per_class.txt` |
| Fig. 8 | 1D-CNN on volunteer A with 8 groupings of principal components | `fig08_pc_groupings.png` |
| Fig. 9 | PCWCNN on A, B, C with adaptive segmentation vs none | `fig09_segmentation.png` |
| Fig. 10 + Table 2 | PCWCNN on A, B, C: confusion matrix and per-activity results | `fig10_confusion.png`, `table2_per_class.txt`, `table2_f1_per_activity.png` |
| Table 3 | PCWCNN on all 10 volunteers | `table3.txt` |
| Fig. 11 | SVM, RF, 1D-CNN and PCWCNN, each distance/height trained and tested on its own (skipped if the distance/height data is missing) | `fig11_distance_height.png` |
| all of the above | | `COMPARISON.md`: every number beside the paper's, with the difference |

Everything goes to `output/paper_figures_<date-time>/`. By default every number is the mean ±
standard deviation over 3 seeds. `results.json` is saved after each figure, so an interrupted run
keeps the figures that were finished; re-run the missing ones with `--figures`.

The settings block at the top of `paper_figures.py` chooses the figures, seeds and volunteers. The
paper does not say which volunteers it used: "one volunteer" is taken as A, and "three volunteers" as
A, B and C, because the paper's Fig. 10 implies 18 test recordings per activity, i.e. 1,440 recordings
= three volunteers.

---

## 9. Ablation studies

To measure how much each part of the method matters (principal components, segmentation,
augmentation, network size, number of epochs), follow **[ABLATION_GUIDE.md](ABLATION_GUIDE.md)**. It
explains how to keep an ablation fair, gives ready-to-copy commands and a table to fill in.
`paper_figures.py` already covers the paper's own two ablations (Fig. 8 and Fig. 9).

---

## 10. Differences from the paper

| | Paper | This code | Why |
|---|---|---|---|
| PCA basis | not stated | one basis shared by all recordings, fitted on the training data only | a separate PCA per recording gives about 55% instead of 84% (SVM, one volunteer) |
| PCA formula | h = R·v (a 90-element vector) | h = H·v (a time series) | the printed formula cannot produce a signal |
| PCs for PCWCNN | 2, 3 | 2, 3, 4, 5 (changeable) | higher accuracy |
| Segmentation for PCWCNN | adaptive | none (changeable) | adaptive did not help |
| Savitzky-Golay window, segmentation windows | not stated | 5 / 2 and 15 / 8 | chosen on a validation split |
| Network size | 9.6 M parameters | 0.82 M (changeable) | same accuracy, much faster |
| Epochs | 80 | 30 (changeable) | enough for convergence; the ablation study uses 80 |
| Augmentation | none | light shift / scale / noise (can be switched off) | about +4 points |
| SVM input | features reduced to 3-D with PCA | full 672-element wavelet feature vector | the literal 3-D version performs far worse |

The reference models (Figs. 7, 8 and 11) always use the paper's own pipeline: PC2,3 and adaptive
segmentation.

---

## 11. Known limitations

- **1D-CNN input length.** The reference 1D-CNN takes a fixed 384-value input (the level-1 wavelet
  coefficients of each principal component, side by side). With three principal components the
  vector is cut to 384, so only the first two components reach the network. In Fig. 8 this affects
  the groupings PC-1,2,3, PC-2,3,4 and PC-2,3,5. PCWCNN is not affected.
- **Small data.** Each distance condition has only 2 test recordings per activity, and volunteers D,
  E and F have fewer recordings than the others.
- **Not bit-exact.** GPU training is not perfectly deterministic, so re-runs give numbers close to,
  but not identical with, the ones above.
- The paper's sliding-window segmentation baseline (Fig. 9) and a no-PCA ("all subcarriers") switch
  (Fig. 7) are not implemented.

---

## 12. Running time

| Run | NVIDIA GPU | CPU only |
|---|---|---|
| `run.py`, 2 volunteers, both tests, 1 seed | under 1 min | about 2 min |
| `run.py`, 6 volunteers, both tests, 1 seed | a few minutes | about 30 min |
| `run.py`, all 10, both tests, 1 seed | 5-10 min | about 1 h |
| `run.py`, distance or height, both tests, 3 seeds | a few minutes | 15-30 min |
| `paper_figures.py`, everything, 3 seeds | 30-50 min | several hours |

More seeds multiply the time (3 seeds = 3x).

---

## 13. Troubleshooting

| Problem | Fix |
|---|---|
| `No suitable Python runtime found` | Run `py -0` to list your Python versions and use one of them (3.10-3.12) in step 2. |
| `No module named torch` (or numpy, scipy, ...) | You are running a different Python. Use `..\.venv\Scripts\python.exe` as shown, or select the `.venv` interpreter in VS Code. |
| `No .dat files found in ...` | `raw/` is not next to the repository folder. Move it, or set `RAW_DIR` in `decode.py`. |
| `Folder ...raw_factors not found` | You asked for distance/height data but `raw_factors/` does not exist. Add it (section 4), or set `FACTOR_DIR` in `decode.py`. |
| `No recordings found for [...]` | That volunteer's or condition's raw folder is missing. Add it, delete the matching `.npz` file in `data/`, and run again. |
| `Unknown name(s)` | Use A B C D E F 7 8 9 10 (or `all`), or 1m 3m 6m 60cm 90cm 120cm (or `distance` / `height`). |
| `Don't mix volunteers and ... conditions` | Run volunteers and distance/height conditions separately. |
| `running scripts is disabled on this system` (PowerShell) | You do not need to activate the environment; call `..\.venv\Scripts\python.exe` directly. |
| Very slow | You are on the CPU. Check step 5, or try fewer volunteers and `--epochs 10` first. |

To rebuild the decoded data (for example after adding raw folders), delete the `.npz` files in
`data/` or run `python decode.py`.

---

## 14. References

1. I. A. Showmik, T. F. Sanam and H. Imtiaz, "Human Activity Recognition from Wi-Fi CSI data using
   Principal Component-based Wavelet CNN," *Digital Signal Processing*, vol. 138, p. 104056, 2023.
2. L. Guo, L. Wang, C. Lin, J. Liu, B. Lu, J. Fang, Z. Liu, Z. Shan, J. Yang and S. Guo, "WiAR: A
   Public Dataset for WiFi-Based Activity Recognition," *IEEE Access*, vol. 7, pp. 154935-154945, 2019.
3. S. Fujieda, K. Takayama and T. Hachisuka, "Wavelet Convolutional Neural Networks,"
   arXiv:1805.08620, 2018.
4. D. Halperin, W. Hu, A. Sheth and D. Wetherall, "Tool release: gathering 802.11n traces with
   channel state information," *ACM SIGCOMM Computer Communication Review*, vol. 41, no. 1, 2011.

---

## 15. Authors

Group 03, Section G2, EEE 402, Department of Electrical and Electronic Engineering, Bangladesh
University of Engineering and Technology (BUET):

- Md. Rafid Zaman Khan Ridro
- Mustakim Billah Nafee
- Mushfiqul Amin
