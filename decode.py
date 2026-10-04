"""
decode.py  -  reads the raw WiAR .dat files and saves them as data files Python can load fast.

The WiFi card (Intel 5300) writes binary .dat files. This file turns each recording into
a matrix of  T packets x 90 streams  (3 receive antennas x 30 subcarriers), where every
value is the signal amplitude in dB. Nothing else is changed.

Two datasets, two output files:

  raw/          the 10 volunteers           ->  data/volunteers_csi.npz
  raw_factors/  distance & height data      ->  data/factors_csi.npz
                (one person, recorded at 1 m / 3 m / 6 m and at 60 / 90 / 120 cm)

In factors_csi.npz the condition ("1m", "3m", "6m", "60cm", "90cm", "120cm") is stored in
the field where volunteers_csi.npz stores the volunteer name, so the rest of the code
treats a condition exactly like a volunteer.

You normally never run this yourself: run.py calls it automatically the first time a file
is missing. It takes a few minutes, once. Run it by hand (e.g. after changing the raw
data):   python decode.py

This is a short port of the dataset's own MATLAB reader (read_bf_file.m, read_bfee.c,
get_scaled_csi.m), which the full project verified bit-exactly against the dataset
authors' exported values.
"""
import os
import re
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "raw")                 # where the volunteer folders are
OUT_FILE = os.path.join(HERE, "data", "volunteers_csi.npz")
FACTOR_DIR = os.path.join(HERE, "..", "raw_factors")      # where the distance/height folders are
FACTOR_FILE = os.path.join(HERE, "data", "factors_csi.npz")

# raw folder name -> volunteer name
FOLDERS = {
    "volunteer_a_all_data": "A", "volunteer_b_all_data": "B",
    "volunteer_c_all_data": "C", "volunteer_f_all_data": "F",
    "volunteerD_1": "D", "volunteerD_2": "D",
    "volunteerE_1": "E", "volunteerE_2": "E",
    "volunteer_7": "7", "volunteer_8": "8", "volunteer_9": "9", "volunteer_10": "10",
}
FILENAME = re.compile(r"^csi_a(\d+)_(\d+)(?:\.dat)?$", re.IGNORECASE)   # csi_a<activity>_<n>

# distance / height folder name -> condition name. The dataset calls them e.g.
# distance_factor_activity_data/3_alldata  and  height_factor_activity_data/90_rawdata;
# plain names like "3m" or "90cm" are accepted too. They can be nested at any depth.
DISTANCE_FOLDER = re.compile(r"^(1|3|6)(?:_alldata|m)$", re.IGNORECASE)
HEIGHT_FOLDER = re.compile(r"^(60|90|120)(?:_rawdata|cm)$", re.IGNORECASE)


def read_packet(rec):
    """Decode one CSI packet (a 0xBB record). Returns (csi, Nrx, Ntx).

    csi is complex, shape (Ntx, Nrx, 30). The values are packed as 8-bit signed
    integers at odd bit positions, so they are unpacked bit by bit.
    """
    Nrx, Ntx = int(rec[8]), int(rec[9])
    antenna_sel = int(rec[15])
    payload = rec[20:]
    n = Nrx * Ntx

    # bit position of every value: 3 skip bits per subcarrier, then 16 bits per entry
    sc = np.arange(30)[:, None]
    entry = np.arange(n)[None, :]
    bit = 3 * (sc + 1) + 16 * sc * n + 16 * entry                 # (30, n)
    byte, shift = bit // 8, (bit % 8).astype(np.uint16)
    b0, b1, b2 = (payload[byte + k].astype(np.uint16) for k in range(3))
    real = (((b0 >> shift) | (b1 << (8 - shift))) & 0xFF).astype(np.int16)
    imag = (((b1 >> shift) | (b2 << (8 - shift))) & 0xFF).astype(np.int16)
    real[real > 127] -= 256
    imag[imag > 127] -= 256
    flat = real + 1j * imag                                       # (30, n)

    csi = np.empty((Ntx, Nrx, 30), dtype=complex)
    for j in range(n):
        csi[j % Ntx, j // Ntx, :] = flat[:, j]

    # put the receive antennas in their physical order
    if Nrx > 1:
        perm = np.array([antenna_sel & 3, (antenna_sel >> 2) & 3, (antenna_sel >> 4) & 3])
        if perm[:Nrx].sum() + Nrx == (1, 3, 6)[Nrx - 1]:
            fixed = csi.copy()
            fixed[:, perm[:Nrx], :] = csi[:, :Nrx, :]
            csi = fixed
    return csi, Nrx, Ntx


def read_dat(path):
    """Read one .dat recording -> (T, 90) amplitude in dB."""
    data = np.fromfile(path, dtype=np.uint8)
    rows, cur = [], 0
    while cur < len(data) - 3:
        length = (int(data[cur]) << 8) | int(data[cur + 1])
        code = int(data[cur + 2])
        rec = data[cur + 3:cur + 2 + length]
        cur += 2 + length
        if code != 0xBB or len(rec) != length - 1:
            continue
        Nrx, Ntx = int(rec[8]), int(rec[9])
        if int(rec[16]) + (int(rec[17]) << 8) != (30 * (Nrx * Ntx * 16 + 3) + 7) // 8:
            continue                                              # damaged packet
        csi, Nrx, Ntx = read_packet(rec)

        # WiAR's get_scaled_csi.m: its noise term is switched off, so all the RSSI
        # scaling cancels out and what remains is  csi / sqrt(Nrx * Ntx)
        gain = 1 / np.sqrt(Nrx * Ntx) * {1: 1, 2: np.sqrt(2), 3: np.sqrt(10 ** 0.45)}[Ntx]
        with np.errstate(divide="ignore"):
            amp = 20 * np.log10(np.abs(csi[0]) * gain)            # first transmit antenna
        row = np.zeros((3, 30))
        row[:Nrx] = amp
        rows.append(row.reshape(90))
    H = np.array(rows)
    H[(H < -20) | (H > 70)] = 25.0                                # authors' outlier rule
    return H


def list_dat_files(folder, name):
    """(name, activity, sample, path) for every csi_a<activity>_<sample> file in folder."""
    items = []
    for fn in os.listdir(folder):
        m = FILENAME.match(fn)
        if m and int(m.group(2)) <= 30:              # volunteer A has a stray 31st sample
            items.append((name, int(m.group(1)), int(m.group(2)), os.path.join(folder, fn)))
    return items


def decode_and_save(items, out_file, source):
    """Decode a list of recordings and save them all in one .npz file."""
    items.sort(key=lambda r: (r[0], r[1], r[2]))
    print(f"Decoding {len(items)} raw recordings from {os.path.abspath(source)} "
          f"(one-time, a few minutes)...")
    t0, mats = time.time(), []
    for i, (_, _, _, path) in enumerate(items):
        mats.append(read_dat(path).astype(np.float32))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(items)}  ({time.time() - t0:.0f}s)")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    np.savez_compressed(
        out_file,
        data=np.concatenate(mats),
        offsets=np.cumsum([0] + [len(m) for m in mats]),
        volunteer=np.array([r[0] for r in items]),     # volunteer name, or condition name
        activity=np.array([r[1] for r in items], dtype=np.int16),
        sample=np.array([r[2] for r in items], dtype=np.int16),
    )
    print(f"Saved {out_file} ({os.path.getsize(out_file) / 1e6:.0f} MB) "
          f"in {time.time() - t0:.0f}s")
    return out_file


def build(raw_dir=RAW_DIR, out_file=OUT_FILE):
    """Volunteer recordings (raw/) -> data/volunteers_csi.npz"""
    items = []
    for folder, vol in FOLDERS.items():
        d = os.path.join(raw_dir, folder)
        if not os.path.isdir(d):
            print(f"  (skipping missing folder {folder})")
            continue
        items += list_dat_files(d, vol)
    if not items:
        raise SystemExit(f"No .dat files found in {os.path.abspath(raw_dir)}")
    return decode_and_save(items, out_file, raw_dir)


def build_factors(factor_dir=FACTOR_DIR, out_file=FACTOR_FILE):
    """Distance / height recordings (raw_factors/) -> data/factors_csi.npz"""
    if not os.path.isdir(factor_dir):
        raise SystemExit(f"Folder {os.path.abspath(factor_dir)} not found. Put the distance / "
                         f"height data there (see README), or set FACTOR_DIR in decode.py.")
    items = []
    for folder, _, _ in os.walk(factor_dir):        # search every sub-folder
        base = os.path.basename(folder)
        if DISTANCE_FOLDER.match(base):
            items += list_dat_files(folder, DISTANCE_FOLDER.match(base).group(1) + "m")
        elif HEIGHT_FOLDER.match(base):
            items += list_dat_files(folder, HEIGHT_FOLDER.match(base).group(1) + "cm")
    if not items:
        raise SystemExit(f"No distance / height .dat files found in "
                         f"{os.path.abspath(factor_dir)}")
    found = sorted(set(r[0] for r in items), key=lambda c: (c.endswith("cm"), int(c.rstrip("cm"))))
    print(f"Found conditions: {', '.join(found)}")
    return decode_and_save(items, out_file, factor_dir)


if __name__ == "__main__":
    if os.path.isdir(RAW_DIR):
        build()
    else:
        print(f"(no {os.path.abspath(RAW_DIR)} folder - skipping the volunteer data)")
    if os.path.isdir(FACTOR_DIR):
        build_factors()
    else:
        print(f"(no {os.path.abspath(FACTOR_DIR)} folder - skipping the distance/height data)")
