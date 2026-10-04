"""
preprocess.py  -  turns raw WiFi CSI recordings into clean, fixed-size network inputs.

The raw .dat files are decoded once by decode.py and saved in  data/volunteers_csi.npz
(one matrix per recording: T packets x 90 streams, amplitude in dB).
This file does everything after that, following the paper (Fig. 4):

    raw (T x 90)
      -> 1. remove static background   (subtract each stream's average)
      -> 2. PCA subcarrier fusion      (90 streams -> a few principal components)
      -> 3. Savitzky-Golay smoothing   (remove noise, keep the motion shape)
      -> 4. segmentation (optional)    (cut out the part where the person moves)
      -> 5. resample to 384 points     (every recording gets the same length)

Step 6 of the paper, the wavelet transform (DWT), is done inside the network (model.py).
"""
import numpy as np
from scipy.signal import savgol_filter

ACTIVITIES = [
    "Horizontal Arm Wave", "High Arm Wave", "Two Hands Wave", "High Throw",
    "Draw X", "Draw Tick", "Toss Paper", "Forward Kick", "Side Kick", "Bend",
    "Hand Clap", "Walk", "Phone Call", "Drink Water", "Sit Down", "Squat",
]
ALL_VOLUNTEERS = ["A", "B", "C", "D", "E", "F", "7", "8", "9", "10"]
# distance / height data (one person): the condition takes the place of the volunteer
DISTANCES = ["1m", "3m", "6m"]
HEIGHTS = ["60cm", "90cm", "120cm"]
ALL_CONDITIONS = DISTANCES + HEIGHTS

LENGTH = 384          # every sample is resampled to this many time steps
SG_WINDOW = 5         # Savitzky-Golay window (packets)
SG_ORDER = 2          # Savitzky-Golay polynomial order


# ---------------------------------------------------------------- loading
def load_volunteers(cache_path, volunteers):
    """Load the recordings of the chosen volunteers (or conditions) from a decoded data file.

    Returns
        traces     : list of arrays, each (T, 90)  - the raw recordings
        labels     : array of activity numbers 0..15
        who        : array of volunteer names (e.g. "A", "8") or condition names ("3m")
    """
    d = np.load(cache_path, allow_pickle=True)
    vol = d["volunteer"].astype(str)
    keep = np.flatnonzero(np.isin(vol, volunteers))

    offsets = d["offsets"]
    data = d["data"]
    traces = [data[offsets[i]:offsets[i + 1]] for i in keep]
    labels = d["activity"][keep].astype(int) - 1      # 1..16  ->  0..15
    return traces, labels, vol[keep]


# ---------------------------------------------------------------- step 1 + 2: PCA
def remove_static(H):
    """Step 1: subtract each stream's long-term average (the static background)."""
    return H - H.mean(axis=0, keepdims=True)


def fit_pca_basis(train_traces, n_components=6):
    """Step 2a: find ONE set of PCA directions, shared by all recordings.

    Uses the TRAINING recordings only (using test data here would be cheating).
    Every recording is then projected onto the same directions, so "PC2" means
    the same thing for every sample.
    """
    R = np.zeros((90, 90))
    for H in train_traces:
        Hc = remove_static(H)
        R += Hc.T @ Hc                    # 90 x 90 correlation of the streams
    eigvals, eigvecs = np.linalg.eigh(R)             # ascending order
    eigvecs = eigvecs[:, ::-1][:, :n_components]     # biggest first
    # an eigenvector's sign is arbitrary; fix it so results are repeatable
    for k in range(eigvecs.shape[1]):
        if eigvecs[np.argmax(np.abs(eigvecs[:, k])), k] < 0:
            eigvecs[:, k] *= -1
    return eigvecs                                    # (90, n_components)


# ---------------------------------------------------------------- step 4: segmentation
def adaptive_segment(x, var_w=15, mean_w=8, t1=0.30, t2=0.50):
    """Step 4 (paper's adaptive method): find the time window with the most motion.

    variance in a sliding window -> smoothed by a sliding mean -> keep the part above
    a threshold p*max. p is chosen so the kept part is 30-50% of the recording.
    Returns (start, end).
    """
    T = len(x)
    win = np.lib.stride_tricks.sliding_window_view
    var = win(x, var_w, axis=0).var(axis=-1).mean(axis=1)   # average over the PCs
    M = win(var, mean_w).mean(axis=-1)
    offset = (var_w - 1) // 2 + (mean_w - 1) // 2

    best, closest = None, None
    for p in np.linspace(0.005, 0.995, 200):
        idx = np.flatnonzero(M >= p * M.max())
        if idx.size == 0:
            break
        lo, hi = idx[0], idx[-1]
        t = (hi - lo + 1) / T                               # fraction of recording kept
        if closest is None or abs(t - 0.4) < closest[0]:
            closest = (abs(t - 0.4), lo, hi)
        if t <= t1:
            break
        if t < t2:
            best = (lo, hi)                                 # smallest valid so far
    lo, hi = best if best else closest[1:]
    start = int(np.clip(lo + offset, 0, T - 1))
    end = int(np.clip(hi + offset + 1, start + 2, T))
    return start, end


# ---------------------------------------------------------------- all steps together
def preprocess(traces, basis, pcs=(2, 3, 4, 5), segmentation="none"):
    """Run steps 1-5 on every recording.

    traces       : list of (T, 90) raw recordings
    basis        : PCA directions from fit_pca_basis()
    pcs          : which principal components to keep (1-based, like the paper)
    segmentation : "none" or "adaptive"

    Returns an array (n_samples, n_pcs, 384), ready for the network.
    """
    cols = [p - 1 for p in pcs]
    out = np.zeros((len(traces), len(pcs), LENGTH), dtype=np.float32)
    for i, H in enumerate(traces):
        x = remove_static(H) @ basis[:, cols]                          # step 2b: (T, n_pcs)
        x = savgol_filter(x, SG_WINDOW, SG_ORDER, axis=0)              # step 3
        if segmentation == "adaptive":                                 # step 4
            s, e = adaptive_segment(x)
            x = x[s:e]
        src = np.linspace(0, 1, len(x))                                # step 5
        dst = np.linspace(0, 1, LENGTH)
        out[i] = np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])])
    return out
