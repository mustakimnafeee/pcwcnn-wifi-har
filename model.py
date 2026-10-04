"""
model.py  -  the PCWCNN network (paper Fig. 6), plus a simple training loop.

The network has two paths that are joined together:

  MAIN PATH (trunk):   4 blocks of  Conv -> Conv(stride 2).  Each block halves the length.
                       384 -> 192 -> 96 -> 48 -> 24

  WAVELET PATHS:       the input is also passed through a Haar wavelet transform (DWT)
                       3 times, giving coarser versions of the signal:
                       384 -> 192 -> 96 -> 48
                       Each level goes through its own small conv branch and is
                       concatenated to the trunk at the matching length.

  END:                 average over time -> one fully-connected layer -> 16 activities

"width" scales the number of channels. width=1.0 is the paper's exact size (9.6 M
parameters, slow). width=0.25 has ~16x fewer parameters and trains many times faster
with very little loss in accuracy.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def haar_dwt(x):
    """One level of Haar wavelet transform, approximation part only.
    Averages each pair of neighbouring points (times sqrt 2): length L -> L/2."""
    return F.avg_pool1d(x, 2) * math.sqrt(2)


def conv(cin, cout, stride=1):
    """3x1 convolution + batch-norm + ReLU (the paper's basic building block)."""
    return nn.Sequential(
        nn.Conv1d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
    )


class PCWCNN(nn.Module):
    def __init__(self, n_inputs, n_classes=16, width=0.25):
        super().__init__()
        w = lambda c: max(8, int(c * width))          # scaled channel count
        self.n_inputs = n_inputs

        # wavelet branches: level 1 -> 1 conv, level 2 -> 2 convs, level 3 -> 3 convs.
        # Each principal component goes through the same branch separately.
        branch_ch = [w(128), w(256), w(512)]
        self.branches = nn.ModuleList()
        for level in range(3):
            layers, cin = [], 1
            for c in branch_ch[:level + 1]:
                layers.append(conv(cin, c))
                cin = c
            self.branches.append(nn.Sequential(*layers))

        # trunk blocks and their shortcuts
        mid = [w(64), w(128), w(256), w(512)]
        out = [w(128), w(256), w(512), w(1024)]
        self.blocks, self.shortcuts = nn.ModuleList(), nn.ModuleList()
        cin = n_inputs
        for i in range(4):
            self.blocks.append(nn.Sequential(conv(cin, mid[i]), conv(mid[i], out[i], stride=2)))
            self.shortcuts.append(nn.Sequential(
                nn.Conv1d(cin, out[i], kernel_size=1, stride=2, bias=False),
                nn.BatchNorm1d(out[i])))
            cin = out[i] * 2                                    # block + shortcut
            if i < 3:
                cin += branch_ch[i] * n_inputs                  # + wavelet branch
        self.fc = nn.Linear(cin, n_classes)

    def forward(self, x):                                       # x: (batch, n_pcs, 384)
        B, C, _ = x.shape
        a, h = x, x
        for i in range(4):
            parts = [self.blocks[i](h), self.shortcuts[i](h)]
            if i < 3:
                a = haar_dwt(a)                                 # next wavelet level
                L = a.shape[-1]
                # run every PC through the branch separately, then stack them back
                b = self.branches[i](a.reshape(B * C, 1, L)).reshape(B, -1, L)
                parts.append(b)
            h = torch.cat(parts, dim=1)                         # feature concatenation
        return self.fc(h.mean(dim=-1))                          # global average pooling


# ---------------------------------------------------------------- training
def augment(x, gen):
    """Light data augmentation: random time shift, amplitude scale and noise."""
    B, _, L = x.shape
    shift = torch.randint(-L // 10, L // 10 + 1, (1,), generator=gen).item()
    x = torch.roll(x, shift, dims=-1)
    # random numbers are drawn on the CPU (repeatable), then moved to x's device (GPU)
    scale = (1 + 0.1 * (2 * torch.rand(B, 1, 1, generator=gen) - 1)).to(x.device)
    noise = torch.randn(x.shape, generator=gen).to(x.device)
    return x * scale + 0.02 * x.std() * noise


def train_and_test(Xtr, ytr, Xte, yte, epochs=30, width=0.25, batch=32, lr=1e-3,
                   use_augment=True, seed=0, label="", make_model=None, verbose=True):
    """Train a fresh PCWCNN on (Xtr, ytr) and return predictions for Xte.

    make_model: optional function that builds a different network (used by baselines.py
    for the paper's reference 1D-CNN). By default a PCWCNN is built.

    Settings follow the paper: Adam, lr 1e-3 halved when the loss stops improving
    (down to 1e-5), batch 32, cross-entropy loss. The model from the LAST epoch is
    used - we never pick the epoch by looking at the test set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator().manual_seed(seed)

    xtr = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    xte = torch.tensor(Xte, device=device)

    model = (make_model() if make_model else PCWCNN(n_inputs=Xtr.shape[1], width=width))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5,
                                                       min_lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    def predict():
        model.eval()
        with torch.no_grad():
            return torch.cat([model(xte[i:i + 256]).argmax(1)
                              for i in range(0, len(xte), 256)]).cpu().numpy()

    history, t0 = [], time.time()
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(xtr), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            if len(idx) < 2:
                continue                                        # batch-norm needs >= 2
            xb = augment(xtr[idx], gen) if use_augment else xtr[idx]
            loss = loss_fn(model(xb), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / len(xtr)
        sched.step(train_loss)
        test_acc = float((predict() == yte).mean())
        history.append({"epoch": ep + 1, "train_loss": train_loss, "test_acc": test_acc})
        if verbose and ((ep + 1) % 5 == 0 or ep == 0 or ep + 1 == epochs):
            print(f"   {label} epoch {ep + 1:3d}/{epochs}  loss {train_loss:.3f}  "
                  f"test acc {100 * test_acc:5.1f}%   ({time.time() - t0:.0f}s)")
    n_params = sum(p.numel() for p in model.parameters())
    return predict(), history, n_params
