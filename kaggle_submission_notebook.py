#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════
#  IMAGE RESTORATION COMPETITION — KAGGLE NOTEBOOK  (T4 × 2)
# ═══════════════════════════════════════════════════════════════════════
#  Task : 2× super-resolution + denoising
#  Input: 128×128 float32 grayscale .npy  →  Output: 256×256 float32
#
#  Hardware target : Kaggle T4 × 2  (2 × 16 GB VRAM, 4 vCPUs)
#  Parallelism     : nn.DataParallel over both T4s, FP16 AMP, 4 workers
#
#  Paste each "# %% [markdown]" / "# %%" block as a cell in Kaggle,
#  or upload this .py directly as a Kaggle script notebook.
# ═══════════════════════════════════════════════════════════════════════

# %% [markdown]
# # Image Restoration — 2× SR + Denoising
# **Pipeline**: Train U-Net with PixelShuffle → Self-ensemble inference → Submission

# %% — Imports & Config
import os, glob, math, time, random
import numpy as np
import pandas as pd
import base64
from io import BytesIO

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark    = True
torch.backends.cudnn.deterministic = False  # faster, non-deterministic OK for training

# ── Paths  (auto-detect Kaggle vs local) ──
if os.path.exists("/kaggle/working"):
    _inp = "/kaggle/input"
    # auto-pick the dataset folder that contains train/GT
    _ds  = next(
        (d for d in sorted(os.listdir(_inp))
         if os.path.isdir(os.path.join(_inp, d, "train"))),
        None
    )
    DATA_ROOT = os.path.join(_inp, _ds) if _ds else _inp
    SUB_DIR   = "/kaggle/working/submission"
else:
    DATA_ROOT = "data"
    SUB_DIR   = "submission"

TRAIN_GT  = os.path.join(DATA_ROOT, "train", "GT")
TRAIN_LR  = os.path.join(DATA_ROOT, "train", "NoisyLR")
TEST_LR   = os.path.join(DATA_ROOT, "test",  "NoisyLR")
os.makedirs(SUB_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS  = torch.cuda.device_count()  # 2 on Kaggle T4×2


# %% — Hyperparameters (tune these!)
class CFG:
    # Architecture
    base_channels = 64          # U-Net base width
    n_res_blocks  = 4           # residual blocks per encoder/decoder level

    # Training
    epochs        = 200         # T4×2 is fast — more epochs → better convergence
    batch_size    = 64          # 32 per T4 × 2 GPUs; DataParallel splits automatically
    lr            = 2e-4
    weight_decay  = 1e-4
    freq_loss_w   = 0.1         # weight for frequency-domain loss

    # Validation
    val_frac      = 0.05        # 5% held out for monitoring

    # Inference
    use_tta       = True        # 8× self-ensemble (flip + rotate)
    clamp_output  = True        # clamp predictions to [0, 1]

    # Speed  (tuned for Kaggle T4×2: 2×16 GB VRAM, 4 vCPUs)
    use_amp       = True        # FP16 AMP — T4 Turing tensor cores (~1.5-2× faster)
    num_workers   = 4           # Kaggle T4 instance provides 4 logical CPUs


# %% — Dataset
class SRDataset(Dataset):
    """
    Loads .npy image pairs for training, or LR-only for test.
    Images: grayscale float32, LR=128×128, GT=256×256.
    """
    def __init__(self, lr_dir, gt_dir=None, augment=False):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
        self.gt_dir   = gt_dir
        self.augment  = augment and (gt_dir is not None)
        if gt_dir:
            self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
            assert len(self.lr_paths) == len(self.gt_paths), \
                f"Mismatch: {len(self.lr_paths)} LR vs {len(self.gt_paths)} GT"

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = np.load(self.lr_paths[idx]).astype(np.float32)

        if self.gt_dir is not None:
            gt = np.load(self.gt_paths[idx]).astype(np.float32)

            if self.augment:
                # Random horizontal flip
                if random.random() > 0.5:
                    lr = np.fliplr(lr).copy()
                    gt = np.fliplr(gt).copy()
                # Random vertical flip
                if random.random() > 0.5:
                    lr = np.flipud(lr).copy()
                    gt = np.flipud(gt).copy()
                # Random 90° rotation (k ∈ {0,1,2,3})
                k = random.randint(0, 3)
                lr = np.rot90(lr, k).copy()
                gt = np.rot90(gt, k).copy()

            # (H,W) → (1,H,W)
            lr_t = torch.from_numpy(lr).unsqueeze(0)
            gt_t = torch.from_numpy(gt).unsqueeze(0)
            return lr_t, gt_t
        else:
            fname = os.path.basename(self.lr_paths[idx])
            lr_t  = torch.from_numpy(lr).unsqueeze(0)
            return lr_t, fname


# %% — Model Architecture

class ResidualBlock(nn.Module):
    """Pre-activation residual block with scaled skip."""
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=True),
        )
        self.scale = 0.2  # scaled residual for stable training

    def forward(self, x):
        return x + self.body(x) * self.scale


class UNetSR(nn.Module):
    """
    U-Net with residual blocks and PixelShuffle ×2 upsampling.

    Input:  (B, 1, 128, 128)  — NoisyLR
    Output: (B, 1, 256, 256)  — Restored HR

    Global residual: output = net(x) + bicubic_upsample(x)
    """
    def __init__(self, ch=64, n_blocks=4):
        super().__init__()

        # ── Encoder ──
        self.head = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc1 = nn.Sequential(*[ResidualBlock(ch) for _ in range(n_blocks)])
        self.down1 = nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            *[ResidualBlock(ch * 2) for _ in range(n_blocks)]
        )
        self.down2 = nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1)

        # ── Bottleneck ──
        self.bottleneck = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            *[ResidualBlock(ch * 4) for _ in range(n_blocks)]
        )

        # ── Decoder ──
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1)
        self.skip_fuse2 = nn.Conv2d(ch * 4, ch * 2, 1)  # cat(up, skip) → fused
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            *[ResidualBlock(ch * 2) for _ in range(n_blocks)]
        )

        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1)
        self.skip_fuse1 = nn.Conv2d(ch * 2, ch, 1)
        self.dec1 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            *[ResidualBlock(ch) for _ in range(n_blocks)]
        )

        # ── 2× Upsample via PixelShuffle ──
        self.tail = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, 4, 3, padding=1),  # 4 = 1 channel × (2²)
            nn.PixelShuffle(2),               # (B, 1, 256, 256)
        )

    def forward(self, x):
        # Global residual: bicubic upsample of input
        x_bic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)

        # Encoder path
        e1 = self.enc1(self.head(x))              # (B, ch,   128, 128)
        e2 = self.enc2(self.down1(e1))             # (B, ch*2,  64,  64)

        # Bottleneck
        bn = self.bottleneck(self.down2(e2))       # (B, ch*4,  32,  32)

        # Decoder path with skip connections
        d2 = self.up2(bn)                          # (B, ch*2,  64,  64)
        d2 = self.skip_fuse2(torch.cat([d2, e2], dim=1))
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                          # (B, ch,   128, 128)
        d1 = self.skip_fuse1(torch.cat([d1, e1], dim=1))
        d1 = self.dec1(d1)

        # PixelShuffle 2× upsample
        out = self.tail(d1)                        # (B, 1, 256, 256)

        return out + x_bic                         # global residual


# %% — Loss Functions

class CharbonnierLoss(nn.Module):
    """Smooth L1 variant — less sensitive to outliers than MSE."""
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps_sq))


class CombinedLoss(nn.Module):
    """
    Spatial Charbonnier  +  Frequency-domain L1.
    The frequency term helps recover high-frequency detail lost in
    downsampling + noise degradation.
    """
    def __init__(self, freq_weight=0.1):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.fw    = freq_weight

    def forward(self, pred, target):
        loss = self.charb(pred, target)
        if self.fw > 0:
            pred_fft   = torch.fft.rfft2(pred)
            target_fft = torch.fft.rfft2(target)
            loss += self.fw * F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return loss


# %% — Training Loop

def compute_psnr(pred, gt):
    """Per-image PSNR using data_range = gt.max() - gt.min()."""
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12:
        return 60.0
    data_range = gt.max() - gt.min()
    if data_range < 1e-8:
        return 0.0
    return 10.0 * math.log10(data_range ** 2 / mse)


def train_model():
    print("=" * 60)
    print("  TRAINING")
    print("=" * 60)

    # ── Data ──
    full_ds = SRDataset(TRAIN_LR, TRAIN_GT, augment=True)
    n_val   = max(1, int(CFG.val_frac * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Disable augmentation for validation
    # (random_split returns Subsets that still use parent's __getitem__)

    _pin = (DEVICE.type == 'cuda')
    _nw  = CFG.num_workers
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=_nw, pin_memory=_pin,
                          drop_last=True, persistent_workers=_nw > 0,
                          prefetch_factor=2 if _nw > 0 else None)
    val_dl   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False,
                          num_workers=max(1, _nw // 2), pin_memory=_pin,
                          persistent_workers=_nw > 0,
                          prefetch_factor=2 if _nw > 0 else None)

    print(f"  Train: {n_train}  |  Val: {n_val}")

    # ── Model ──
    model = UNetSR(ch=CFG.base_channels, n_blocks=CFG.n_res_blocks).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    # torch.compile (PyTorch 2.0+) — fuses ops, ~15-30% throughput gain
    if hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: skipped ({e})")

    # ── Multi-GPU: DataParallel over both T4s ──
    if N_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))
        print(f"  DataParallel  : {N_GPUS} GPUs  "
              f"(effective batch = {CFG.batch_size}, {CFG.batch_size // N_GPUS} per GPU)")
    else:
        print(f"  DataParallel  : disabled (only {N_GPUS} GPU visible)")

    # ── Optimizer & Scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr,
                            weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs, eta_min=1e-6
    )
    criterion = CombinedLoss(freq_weight=CFG.freq_loss_w)

    # ── AMP scaler (no-op on CPU) ──
    _amp_enabled = CFG.use_amp and DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=_amp_enabled)
    print(f"  AMP: {'ON (FP16)' if _amp_enabled else 'OFF (FP32)'}")

    # ── Training ──
    best_psnr = 0.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, CFG.epochs + 1):
        # — Train phase —
        model.train()
        running_loss = 0.0

        for lr_img, gt_img in train_dl:
            lr_img = lr_img.to(DEVICE, non_blocking=True)
            gt_img = gt_img.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=_amp_enabled):
                pred = model(lr_img)
                loss = criterion(pred, gt_img)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_dl)

        # — Validation phase —
        model.eval()
        psnr_sum = 0.0
        psnr_cnt = 0

        with torch.no_grad():
            for lr_img, gt_img in val_dl:
                lr_img = lr_img.to(DEVICE, non_blocking=True)
                pred   = model(lr_img).cpu()

                for b in range(pred.shape[0]):
                    p = pred[b, 0].numpy()
                    g = gt_img[b, 0].numpy()
                    psnr_sum += compute_psnr(p, g)
                    psnr_cnt += 1

        val_psnr = psnr_sum / max(psnr_cnt, 1)

        # — Checkpointing (save from bare model, not from the DataParallel wrapper) —
        marker = ""
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            _bare = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in _bare.state_dict().items()}
            marker = " ★ new best"

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch <= 3 or marker:
            print(f"  Epoch {epoch:3d}/{CFG.epochs}  "
                  f"loss={avg_loss:.5f}  val_psnr={val_psnr:.2f} dB  "
                  f"lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"[{elapsed/60:.1f}m]{marker}")

    print(f"\n  Training complete in {elapsed/60:.1f} min")
    print(f"  Best validation PSNR: {best_psnr:.2f} dB")

    # Reload best weights into the bare model (strip DataParallel wrapper)
    bare_model = model.module if isinstance(model, nn.DataParallel) else model
    bare_model.load_state_dict(best_state)
    bare_model.to(DEVICE)
    bare_model.eval()
    return bare_model


# %% — Self-Ensemble Inference (TTA)

@torch.no_grad()
def predict_single(model, lr_tensor):
    """
    Predict a single image with optional 8× self-ensemble TTA.
    Input:  lr_tensor shape (1, 1, 128, 128)
    Output: numpy array (256, 256) float32
    """
    if not CFG.use_tta:
        pred = model(lr_tensor.to(DEVICE)).cpu()
        return pred.squeeze().numpy()

    # 8× self-ensemble: 4 rotations × 2 flips
    preds = []
    for do_flip in (False, True):
        for k in range(4):
            x = lr_tensor.clone()
            if do_flip:
                x = torch.flip(x, [-1])        # horizontal flip
            x = torch.rot90(x, k, [-2, -1])    # rotate by k×90°

            p = model(x.to(DEVICE)).cpu()

            # Undo transformations
            p = torch.rot90(p, -k, [-2, -1])
            if do_flip:
                p = torch.flip(p, [-1])
            preds.append(p)

    ensemble = torch.stack(preds).mean(dim=0)
    return ensemble.squeeze().numpy()


# %% — Generate Test Predictions

def generate_predictions(model):
    print("\n" + "=" * 60)
    print("  INFERENCE")
    print("=" * 60)

    model.eval()
    test_ds = SRDataset(TEST_LR, gt_dir=None, augment=False)

    tta_label = "ON (8×)" if CFG.use_tta else "OFF"
    print(f"  Test images: {len(test_ds)}")
    print(f"  Self-ensemble TTA: {tta_label}")

    t0 = time.time()
    n_total = len(test_ds)

    for i in range(n_total):
        lr_tensor, fname = test_ds[i]
        lr_tensor = lr_tensor.unsqueeze(0)  # (1, 1, 128, 128)

        pred = predict_single(model, lr_tensor)

        # ── Post-processing ──
        if CFG.clamp_output:
            pred = np.clip(pred, 0.0, 1.0)

        # Safety checks
        assert pred.shape == (256, 256), f"Bad shape: {pred.shape}"
        assert pred.dtype == np.float32, f"Bad dtype: {pred.dtype}"
        assert not np.isnan(pred).any(), "NaN detected!"
        assert not np.isinf(pred).any(), "Inf detected!"

        # Save
        out_path = os.path.join(SUB_DIR, fname)
        np.save(out_path, pred)

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            print(f"  {i+1:4d}/{n_total}  ({speed:.1f} img/s)")

    elapsed = time.time() - t0
    print(f"\n  Inference complete in {elapsed:.1f}s")
    return n_total


# %% — Create Submission CSV

def create_submission():
    print("\n" + "=" * 60)
    print("  CREATING SUBMISSION")
    print("=" * 60)

    rows = []
    files = sorted([f for f in os.listdir(SUB_DIR) if f.endswith(".npy")])

    for idx, file in enumerate(files, start=1):
        path = os.path.join(SUB_DIR, file)

        # Load and verify
        arr = np.load(path)
        assert arr.shape == (256, 256), f"{file}: shape {arr.shape}"
        assert arr.dtype == np.float32, f"{file}: dtype {arr.dtype}"

        # Encode to base64
        buffer = BytesIO()
        np.save(buffer, arr)
        encoded = base64.b64encode(buffer.getvalue()).decode()

        rows.append({
            "id": idx,
            "npy_base64": encoded,
        })

    df = pd.DataFrame(rows)
    csv_path = "submission.csv"
    df.to_csv(csv_path, index=False)

    print(f"  Files encoded: {len(df)}")
    print(f"  CSV saved:     {csv_path}")
    print(f"  CSV size:      {os.path.getsize(csv_path) / 1e6:.1f} MB")
    print(f"\n  Preview:")
    print(df.head())

    return df


# %% — Validation Sanity Check (optional)

def sanity_check():
    """Quick visual sanity check on a few predictions."""
    try:
        import matplotlib.pyplot as plt

        test_files = sorted(glob.glob(os.path.join(TEST_LR, "*.npy")))[:4]
        pred_files = sorted(glob.glob(os.path.join(SUB_DIR, "*.npy")))[:4]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, (tf, pf) in enumerate(zip(test_files, pred_files)):
            lr   = np.load(tf)
            pred = np.load(pf)

            axes[0, i].imshow(lr, cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f"Input LR [{i}]\n128×128")
            axes[0, i].axis('off')

            axes[1, i].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f"Prediction [{i}]\n256×256")
            axes[1, i].axis('off')

        plt.suptitle("Sanity Check: NoisyLR → Restored HR", fontsize=14)
        plt.tight_layout()
        plt.savefig("sanity_check.png", dpi=100)
        plt.show()
        print("  Sanity check plot saved.")
    except Exception as e:
        print(f"  (Sanity check skipped: {e})")


# %% — RUN EVERYTHING

if __name__ == "__main__":

    print(f"Device : {DEVICE}  |  GPUs: {N_GPUS}")
    print(f"Data   : {DATA_ROOT}")
    print(f"Train GT : {len(os.listdir(TRAIN_GT))} files")
    print(f"Train LR : {len(os.listdir(TRAIN_LR))} files")
    print(f"Test  LR : {len(os.listdir(TEST_LR))} files")

    # Step 1: Train
    model = train_model()

    # Step 2: Predict test set
    n_pred = generate_predictions(model)

    # Step 3: Create submission CSV
    df = create_submission()

    # Step 4: Visual check
    sanity_check()

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  ✓ ALL DONE")
    print("=" * 60)
    print(f"  Predictions saved : {SUB_DIR}/  ({n_pred} files)")
    _csv_path = os.path.join(os.path.dirname(SUB_DIR), "submission.csv") \
                if SUB_DIR != "submission" else "submission.csv"
    print(f"  Submission CSV    : {_csv_path}")
    print(f"  Each output       : 256×256 float32, clipped to [0, 1]")
    print("=" * 60)
