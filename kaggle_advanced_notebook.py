#!/usr/bin/env python
# ═══════════════════════════════════════════════════════════════════════
#  IMAGE RESTORATION — ADVANCED NOTEBOOK  (T4 × 2)
# ═══════════════════════════════════════════════════════════════════════
#  Improvements over baseline:
#    1. NAFNet-style blocks (SimpleGate + Simplified Channel Attention)
#    2. Multi-scale frequency loss (better HF recovery)
#    3. Progressive training: P1 cropped → P2 full-res fine-tune
#    4. Cosine scheduler + EMA weights
#    5. FP16 AMP — unified torch.amp API (no deprecated cuda.amp)
#    6. nn.DataParallel over both T4s (2 × 16 GB VRAM)
#    7. Auto-detect Kaggle paths vs local
#
#  Hardware target : Kaggle T4 × 2  (2 × 16 GB VRAM, 4 vCPUs)
# ═══════════════════════════════════════════════════════════════════════

# %% — Imports
import os, glob, math, time, random, copy
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark    = True
torch.backends.cudnn.deterministic = False  # faster, non-deterministic OK for training

# %% — Paths  (auto-detect Kaggle vs local)
if os.path.exists("/kaggle/working"):
    _inp = "/kaggle/input"
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

# %% — Config
class CFG:
    # Architecture
    base_channels = 64
    n_blocks      = 6          # more blocks = more capacity

    # Training — Phase 1 (main)
    epochs_p1     = 100         # T4×2 speed allows more epochs
    lr_p1         = 2e-4
    batch_size    = 64          # 32 per T4 × 2 GPUs; DataParallel splits automatically

    # Training — Phase 2 (fine-tune, full-res images)
    epochs_p2     = 40
    lr_p2         = 5e-5

    # DataLoader
    num_workers   = 4           # Kaggle T4 instance provides 4 logical CPUs

    # Loss
    freq_weight   = 0.1
    multiscale    = True        # apply loss at 256 + 128 scale

    # EMA
    use_ema       = True
    ema_decay     = 0.999

    # AMP
    use_amp       = True

    # Validation
    val_frac      = 0.05

    # Inference
    use_tta       = True
    clamp_output  = True


# %% — Dataset (same as baseline, with random crop option)
class SRDataset(Dataset):
    def __init__(self, lr_dir, gt_dir=None, augment=False, crop_size=None):
        self.lr_paths  = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
        self.gt_dir    = gt_dir
        self.augment   = augment and (gt_dir is not None)
        self.crop_size = crop_size  # LR crop size (GT crop = 2×)
        if gt_dir:
            self.gt_paths = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
            assert len(self.lr_paths) == len(self.gt_paths)

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = np.load(self.lr_paths[idx]).astype(np.float32)

        if self.gt_dir is not None:
            gt = np.load(self.gt_paths[idx]).astype(np.float32)

            # Random crop (helps regularization + allows larger batches)
            if self.crop_size and self.augment:
                cs = self.crop_size
                h, w = lr.shape
                if h > cs and w > cs:
                    y = random.randint(0, h - cs)
                    x = random.randint(0, w - cs)
                    lr = lr[y:y+cs, x:x+cs]
                    gt = gt[y*2:(y+cs)*2, x*2:(x+cs)*2]

            if self.augment:
                if random.random() > 0.5:
                    lr = np.fliplr(lr).copy(); gt = np.fliplr(gt).copy()
                if random.random() > 0.5:
                    lr = np.flipud(lr).copy(); gt = np.flipud(gt).copy()
                k = random.randint(0, 3)
                lr = np.rot90(lr, k).copy(); gt = np.rot90(gt, k).copy()

            return torch.from_numpy(lr).unsqueeze(0), torch.from_numpy(gt).unsqueeze(0)
        else:
            return torch.from_numpy(lr).unsqueeze(0), os.path.basename(self.lr_paths[idx])


# %% — NAFNet-Style Blocks

class SimpleGate(nn.Module):
    """Split channels in half, multiply element-wise. Cheaper than GELU."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global average pool → 1×1 conv → sigmoid → scale."""
    def __init__(self, ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x * self.fc(self.pool(x))


class NAFBlock(nn.Module):
    """NAFNet-style block: LayerNorm → Conv → SimpleGate → Conv → SCA → skip."""
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(1, ch)  # equivalent to LayerNorm for images
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, padding=1),   # expand for SimpleGate
            SimpleGate(),                            # halves channels
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.sca   = SimplifiedChannelAttention(ch)
        self.scale = nn.Parameter(torch.zeros(1))    # learnable residual scale

    def forward(self, x):
        res = self.body(self.norm(x))
        res = self.sca(res)
        return x + res * self.scale


# %% — Advanced U-Net Model

class AdvancedUNetSR(nn.Module):
    """
    U-Net with NAFNet-style blocks + PixelShuffle ×2.
    Stronger feature extraction than vanilla ResBlocks.
    """
    def __init__(self, ch=64, n_blocks=6):
        super().__init__()

        # Encoder
        self.head = nn.Conv2d(1, ch, 3, padding=1)
        self.enc1 = nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)])
        self.down1 = nn.Conv2d(ch, ch*2, 2, stride=2)  # strided conv downsample

        self.enc2 = nn.Sequential(*[NAFBlock(ch*2) for _ in range(n_blocks)])
        self.down2 = nn.Conv2d(ch*2, ch*4, 2, stride=2)

        # Bottleneck
        self.mid = nn.Sequential(*[NAFBlock(ch*4) for _ in range(n_blocks)])

        # Decoder
        self.up2 = nn.Sequential(
            nn.Conv2d(ch*4, ch*2 * 4, 1),
            nn.PixelShuffle(2),
        )
        self.fuse2 = nn.Conv2d(ch*4, ch*2, 1)
        self.dec2 = nn.Sequential(*[NAFBlock(ch*2) for _ in range(n_blocks)])

        self.up1 = nn.Sequential(
            nn.Conv2d(ch*2, ch * 4, 1),
            nn.PixelShuffle(2),
        )
        self.fuse1 = nn.Conv2d(ch*2, ch, 1)
        self.dec1 = nn.Sequential(*[NAFBlock(ch) for _ in range(n_blocks)])

        # 2× SR upsample
        self.tail = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, 4, 3, padding=1),  # 4 = 1 × 2²
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        x_bic = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)

        e1 = self.enc1(self.head(x))
        e2 = self.enc2(self.down1(e1))
        m  = self.mid(self.down2(e2))

        d2 = self.fuse2(torch.cat([self.up2(m), e2], 1))
        d2 = self.dec2(d2)

        d1 = self.fuse1(torch.cat([self.up1(d2), e1], 1))
        d1 = self.dec1(d1)

        return self.tail(d1) + x_bic


# %% — Losses

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps_sq = eps ** 2
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps_sq))


class MultiScaleFreqLoss(nn.Module):
    """
    Charbonnier + FFT L1 at original and half resolution.
    Multi-scale helps the model learn both coarse structure and fine detail.
    """
    def __init__(self, freq_w=0.1, multiscale=True):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.fw    = freq_w
        self.ms    = multiscale

    def forward(self, pred, target):
        # Full-resolution loss
        loss = self.charb(pred, target)
        if self.fw > 0:
            loss += self.fw * F.l1_loss(
                torch.abs(torch.fft.rfft2(pred)),
                torch.abs(torch.fft.rfft2(target)))

        # Half-resolution loss (optional)
        if self.ms:
            pred_ds   = F.interpolate(pred,   scale_factor=0.5, mode='area')
            target_ds = F.interpolate(target, scale_factor=0.5, mode='area')
            loss += 0.5 * self.charb(pred_ds, target_ds)

        return loss


# %% — EMA Helper

class ModelEMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)


# %% — PSNR
def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-12: return 60.0
    dr = gt.max() - gt.min()
    if dr < 1e-8: return 0.0
    return 10.0 * math.log10(dr ** 2 / mse)


# %% — Training (2-phase progressive)

def train_model():
    print("=" * 60)
    print("  TRAINING — ADVANCED PIPELINE")
    print("=" * 60)

    full_ds = SRDataset(TRAIN_LR, TRAIN_GT, augment=True, crop_size=96)
    n_val   = max(1, int(CFG.val_frac * len(full_ds)))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))

    _pin = (DEVICE.type == 'cuda')
    _nw  = CFG.num_workers
    train_dl = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=_nw, pin_memory=_pin, drop_last=True,
                          persistent_workers=_nw > 0,
                          prefetch_factor=2 if _nw > 0 else None)
    # Validation uses full images (no crop) — create separate dataset
    val_full = SRDataset(TRAIN_LR, TRAIN_GT, augment=False, crop_size=None)
    val_indices = val_ds.indices
    val_dl_items = [val_full[i] for i in val_indices]

    print(f"  Train: {n_train}  |  Val: {n_val}")

    model = AdvancedUNetSR(ch=CFG.base_channels, n_blocks=CFG.n_blocks).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    _bare = model  # bare model reference — valid before and after DataParallel wrapping
    if N_GPUS > 1:
        model = nn.DataParallel(model, device_ids=list(range(N_GPUS)))
        print(f"  DataParallel  : {N_GPUS} GPUs  "
              f"(effective batch = {CFG.batch_size}, {CFG.batch_size // N_GPUS} per GPU)")
    else:
        print(f"  DataParallel  : disabled (only {N_GPUS} GPU visible)")

    ema = ModelEMA(_bare, CFG.ema_decay) if CFG.use_ema else None
    _amp_enabled = CFG.use_amp and DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=_amp_enabled)
    print(f"  AMP: {'ON (FP16)' if _amp_enabled else 'OFF (FP32)'}")

    best_psnr = 0.0
    best_state = None
    t0 = time.time()

    # ── Phase 1: main training ──
    print(f"\n  Phase 1: {CFG.epochs_p1} epochs, lr={CFG.lr_p1}")
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr_p1,
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs_p1, eta_min=1e-6)
    criterion = MultiScaleFreqLoss(CFG.freq_weight, CFG.multiscale)

    for epoch in range(1, CFG.epochs_p1 + 1):
        model.train()
        rloss = 0.0

        for lr_img, gt_img in train_dl:
            lr_img = lr_img.to(DEVICE, non_blocking=True)
            gt_img = gt_img.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=_amp_enabled):
                pred = model(lr_img)
                loss = criterion(pred, gt_img)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            rloss += loss.item()
            if ema: ema.update(model)

        scheduler.step()
        avg_loss = rloss / len(train_dl)

        # Validate (using EMA model if available)
        eval_model = ema.ema if ema else _bare
        eval_model.eval()
        psnr_sum, psnr_cnt = 0.0, 0

        with torch.no_grad():
            for lr_t, gt_t in val_dl_items:
                lr_t = lr_t.unsqueeze(0).to(DEVICE)
                pred = eval_model(lr_t).cpu().squeeze().numpy()
                gt_np = gt_t.squeeze().numpy()
                psnr_sum += compute_psnr(pred, gt_np)
                psnr_cnt += 1

        vpsnr = psnr_sum / max(psnr_cnt, 1)
        marker = ""
        if vpsnr > best_psnr:
            best_psnr = vpsnr
            best_state = {k: v.cpu().clone()
                          for k, v in eval_model.state_dict().items()}
            marker = " ★"

        if epoch % 5 == 0 or epoch <= 3 or marker:
            elapsed = time.time() - t0
            print(f"  P1 Ep {epoch:3d}/{CFG.epochs_p1}  loss={avg_loss:.5f}  "
                  f"psnr={vpsnr:.2f}  lr={scheduler.get_last_lr()[0]:.1e}  "
                  f"[{elapsed/60:.1f}m]{marker}")

    # ── Phase 2: fine-tune with lower LR ──
    print(f"\n  Phase 2: {CFG.epochs_p2} epochs, lr={CFG.lr_p2}")
    # Load into the bare model; DataParallel wraps it in-place — no re-wrapping needed
    _bare.load_state_dict(best_state)
    if ema: ema = ModelEMA(_bare, CFG.ema_decay)
    scaler = torch.amp.GradScaler(enabled=_amp_enabled)  # fresh scaler for phase 2

    # Use full images (no crop) for fine-tuning
    ft_ds = SRDataset(TRAIN_LR, TRAIN_GT, augment=True, crop_size=None)
    ft_train, _ = random_split(ft_ds, [n_train, n_val],
                               generator=torch.Generator().manual_seed(SEED))
    ft_dl = DataLoader(ft_train, batch_size=max(1, CFG.batch_size // 2),
                       shuffle=True, num_workers=_nw, pin_memory=_pin, drop_last=True,
                       persistent_workers=_nw > 0,
                       prefetch_factor=2 if _nw > 0 else None)

    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr_p2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.epochs_p2, eta_min=1e-7)

    for epoch in range(1, CFG.epochs_p2 + 1):
        model.train()
        rloss = 0.0

        for lr_img, gt_img in ft_dl:
            lr_img = lr_img.to(DEVICE, non_blocking=True)
            gt_img = gt_img.to(DEVICE, non_blocking=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=_amp_enabled):
                pred = model(lr_img)
                loss = criterion(pred, gt_img)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            rloss += loss.item()
            if ema: ema.update(model)

        scheduler.step()

        eval_model = ema.ema if ema else _bare
        eval_model.eval()
        psnr_sum, psnr_cnt = 0.0, 0
        with torch.no_grad():
            for lr_t, gt_t in val_dl_items:
                lr_t = lr_t.unsqueeze(0).to(DEVICE)
                pred = eval_model(lr_t).cpu().squeeze().numpy()
                psnr_sum += compute_psnr(pred, gt_t.squeeze().numpy())
                psnr_cnt += 1
        vpsnr = psnr_sum / max(psnr_cnt, 1)
        marker = ""
        if vpsnr > best_psnr:
            best_psnr = vpsnr
            best_state = {k: v.cpu().clone()
                          for k, v in eval_model.state_dict().items()}
            marker = " ★"

        if epoch % 5 == 0 or epoch <= 2 or marker:
            elapsed = time.time() - t0
            print(f"  P2 Ep {epoch:3d}/{CFG.epochs_p2}  "
                  f"loss={rloss/len(ft_dl):.5f}  psnr={vpsnr:.2f}  "
                  f"[{elapsed/60:.1f}m]{marker}")

    print(f"\n  Best PSNR: {best_psnr:.2f} dB")
    # Return bare model with best weights (stripped of DataParallel wrapper)
    _bare.load_state_dict(best_state)
    _bare.to(DEVICE)
    _bare.eval()
    return _bare


# %% — Self-Ensemble TTA

@torch.no_grad()
def predict_single(model, lr_tensor):
    if not CFG.use_tta:
        pred = model(lr_tensor.to(DEVICE)).cpu()
        return pred.squeeze().numpy()

    preds = []
    for flip in (False, True):
        for k in range(4):
            x = lr_tensor.clone()
            if flip: x = torch.flip(x, [-1])
            x = torch.rot90(x, k, [-2, -1])
            p = model(x.to(DEVICE)).cpu()
            p = torch.rot90(p, -k, [-2, -1])
            if flip: p = torch.flip(p, [-1])
            preds.append(p)
    return torch.stack(preds).mean(0).squeeze().numpy()


# %% — Inference

def generate_predictions(model):
    print("\n" + "=" * 60)
    print("  INFERENCE")
    print("=" * 60)
    model.eval()
    test_ds = SRDataset(TEST_LR, gt_dir=None)
    print(f"  Test: {len(test_ds)}  |  TTA: {'ON (8×)' if CFG.use_tta else 'OFF'}")

    t0 = time.time()
    for i in range(len(test_ds)):
        lr_tensor, fname = test_ds[i]
        pred = predict_single(model, lr_tensor.unsqueeze(0))

        if CFG.clamp_output:
            pred = np.clip(pred, 0.0, 1.0)

        pred = pred.astype(np.float32)
        assert pred.shape == (256, 256)
        assert not np.isnan(pred).any()
        assert not np.isinf(pred).any()

        np.save(os.path.join(SUB_DIR, fname), pred)
        if (i+1) % 50 == 0 or (i+1) == len(test_ds):
            print(f"  {i+1}/{len(test_ds)}  ({(i+1)/(time.time()-t0):.1f} img/s)")

    print(f"  Done in {time.time()-t0:.1f}s")
    return len(test_ds)


# %% — Submission CSV

def create_submission():
    print("\n" + "=" * 60)
    print("  CREATING SUBMISSION")
    print("=" * 60)

    rows = []
    files = sorted([f for f in os.listdir(SUB_DIR) if f.endswith(".npy")])

    for idx, file in enumerate(files, start=1):
        arr = np.load(os.path.join(SUB_DIR, file))
        assert arr.shape == (256, 256) and arr.dtype == np.float32

        buffer = BytesIO()
        np.save(buffer, arr)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        rows.append({"id": idx, "npy_base64": encoded})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(os.path.dirname(SUB_DIR), "submission.csv") \
               if SUB_DIR != "submission" else "submission.csv"
    df.to_csv(csv_path, index=False)

    print(f"  {len(df)} files encoded")
    print(f"  CSV: {csv_path} ({os.path.getsize(csv_path)/1e6:.1f} MB)")
    print(df.head())
    return df


# %% — RUN

if __name__ == "__main__":
    print(f"Device : {DEVICE}  |  GPUs: {N_GPUS}")
    print(f"Data   : {DATA_ROOT}")
    print(f"Train GT : {len(os.listdir(TRAIN_GT))} files")
    print(f"Train LR : {len(os.listdir(TRAIN_LR))} files")
    print(f"Test  LR : {len(os.listdir(TEST_LR))} files")

    model = train_model()
    n_pred = generate_predictions(model)
    df = create_submission()

    print("\n" + "=" * 60)
    print("  ✓ SUBMISSION COMPLETE")
    print("=" * 60)
    print(f"  {n_pred} predictions → {SUB_DIR}/")
    _csv = os.path.join(os.path.dirname(SUB_DIR), "submission.csv") \
           if SUB_DIR != "submission" else "submission.csv"
    print(f"  submission.csv → {_csv}")
    print("=" * 60)
