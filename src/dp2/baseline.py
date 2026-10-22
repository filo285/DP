"""DP1 inference components extracted unchanged from baseline_inspection.py."""
import os
from dataclasses import dataclass, field
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy.io
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

@dataclass
class Config:
    csv_paths: dict = field(default_factory=lambda: {'origa': 'eda_outputs/origa_master_table.csv', 'rimone': 'eda_outputs/rimone_master_table.csv', 'drishti': 'eda_outputs/drishti_master_table.csv', 'refuge': 'eda_outputs/refuge_master_table.csv'})
    col_image: str = 'image_path'
    col_mask: str = 'mask_path'
    col_split: str = 'split'
    img_size: int = 512
    encoder: str = 'resnet34'
    encoder_weights: str = 'imagenet'
    dropout: float = 0.2
    batch_size: int = 8
    epochs: int = 60
    lr: float = 0.0001
    weight_decay: float = 0.0001
    early_stop_patience: int = 10
    num_workers: int = 0
    seed: int = 42
    out_dir: str = 'checkpoints'
    ckpt_name: str = 'baseline_unet.pth'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp: bool = True

def load_mask(path: str) -> 'np.ndarray | None':
    """Load a mask — handles image files and ORIGA-style .mat files."""
    if str(path).lower().endswith('.mat'):
        data = scipy.io.loadmat(path)
        return data['mask'].astype(np.uint8)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def _raw_to_unified(mask: np.ndarray, dataset: str) -> np.ndarray:
    """Map a raw mask array to {0:bg, 1:disc-rim, 2:cup}."""
    if mask is None:
        raise ValueError('load_mask returned None — file not found or unreadable')
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    if dataset == 'refuge':
        disc = m <= 200
        cup = m <= 50
    else:
        vals = np.unique(m)
        disc = m > 0
        cup = (m >= vals.max()) & (m > 0) if vals.max() > 0 else np.zeros_like(m, bool)
    unified = np.zeros(m.shape, dtype=np.uint8)
    unified[disc] = 1
    unified[cup] = 2
    return unified

class FundusSegDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform, cfg: Config):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row[self.cfg.col_image])
        if img is None:
            raise FileNotFoundError(row[self.cfg.col_image])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_mask = load_mask(row[self.cfg.col_mask])
        if raw_mask is None:
            raise FileNotFoundError(row[self.cfg.col_mask])
        mask = _raw_to_unified(raw_mask, row['dataset'])
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        aug = self.transform(image=img, mask=mask)
        img = aug['image']
        mask = aug['mask'].long()
        return (img, mask)

def build_transforms(cfg: Config):
    norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_tf = A.Compose([A.LongestMaxSize(cfg.img_size), A.PadIfNeeded(cfg.img_size, cfg.img_size, border_mode=cv2.BORDER_CONSTANT), A.HorizontalFlip(p=0.5), A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5), A.RandomBrightnessContrast(0.1, 0.1, p=0.3), norm, ToTensorV2()])
    eval_tf = A.Compose([A.LongestMaxSize(cfg.img_size), A.PadIfNeeded(cfg.img_size, cfg.img_size, border_mode=cv2.BORDER_CONSTANT), norm, ToTensorV2()])
    return (train_tf, eval_tf)

def load_master_tables(cfg: Config) -> pd.DataFrame:
    frames = []
    for name, path in cfg.csv_paths.items():
        if not os.path.exists(path):
            print(f'[warn] missing CSV for {name}: {path} — skipping')
            continue
        d = pd.read_csv(path)
        missing = [c for c in (cfg.col_image, cfg.col_mask) if c not in d.columns]
        if missing:
            print(f'[warn] {name}: no {missing} column — excluding from segmentation pool. Available: {list(d.columns)}')
            continue
        d = d.dropna(subset=[cfg.col_image, cfg.col_mask])
        d['dataset'] = name
        frames.append(d)
    if not frames:
        raise RuntimeError('No usable master tables found — check CFG.csv_paths.')
    return pd.concat(frames, ignore_index=True)

def make_splits(df: pd.DataFrame, cfg: Config):
    refuge = df[df.dataset == 'refuge']
    others = df[df.dataset != 'refuge']
    val_df = refuge[refuge[cfg.col_split].str.lower() == 'val']
    test_df = refuge[refuge[cfg.col_split].str.lower() == 'test']
    refuge_train = refuge[refuge[cfg.col_split].str.lower() == 'train']
    train_df = pd.concat([refuge_train, others], ignore_index=True)
    print(f'train pool: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}')
    return (train_df, val_df, test_df)

def build_model(cfg: Config) -> nn.Module:
    model = smp.Unet(encoder_name=cfg.encoder, encoder_weights=cfg.encoder_weights, in_channels=3, classes=3, activation=None, aux_params=None)
    if cfg.dropout > 0:
        head = model.segmentation_head
        model.segmentation_head = nn.Sequential(nn.Dropout2d(cfg.dropout), *head)
    return model

def _binary_disc_cup(pred: np.ndarray):
    """From a 3-class map return binary disc (rim+cup) and cup masks."""
    disc = pred >= 1
    cup = pred == 2
    return (disc, cup)

def dice_binary(a: np.ndarray, b: np.ndarray, eps=1e-06) -> float:
    inter = np.logical_and(a, b).sum()
    return (2 * inter + eps) / (a.sum() + b.sum() + eps)

def hausdorff_binary(a: np.ndarray, b: np.ndarray):
    """Symmetric Hausdorff between two binary mask boundaries (in pixels)."""

    def boundary_pts(m):
        if m.sum() == 0:
            return None
        edge = m & ~binary_erosion(m)
        return np.argwhere(edge)
    pa, pb = (boundary_pts(a), boundary_pts(b))
    if pa is None or pb is None:
        return np.nan
    return max(directed_hausdorff(pa, pb)[0], directed_hausdorff(pb, pa)[0])

def vcdr(disc: np.ndarray, cup: np.ndarray) -> float:
    """Vertical cup-to-disc ratio = vertical cup diameter / vertical disc diameter."""

    def vdiam(m):
        rows = np.where(m.any(axis=1))[0]
        return rows.max() - rows.min() + 1 if rows.size else 0
    dd = vdiam(disc)
    return vdiam(cup) / dd if dd > 0 else np.nan

@torch.no_grad()
def evaluate(model, loader, cfg: Config):
    model.eval()
    dscs_disc, dscs_cup, hd_disc, hd_cup, cdr_err = ([], [], [], [], [])
    for img, mask in loader:
        img = img.to(cfg.device)
        logits = model(img)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        gt = mask.numpy()
        for p, g in zip(pred, gt):
            pd_disc, pd_cup = _binary_disc_cup(p)
            gd_disc, gd_cup = _binary_disc_cup(g)
            dscs_disc.append(dice_binary(pd_disc, gd_disc))
            dscs_cup.append(dice_binary(pd_cup, gd_cup))
            hd_disc.append(hausdorff_binary(pd_disc, gd_disc))
            hd_cup.append(hausdorff_binary(pd_cup, gd_cup))
            ce = abs(vcdr(pd_disc, pd_cup) - vcdr(gd_disc, gd_cup))
            cdr_err.append(ce)
    return {'dice_disc': float(np.nanmean(dscs_disc)), 'dice_cup': float(np.nanmean(dscs_cup)), 'hd_disc': float(np.nanmean(hd_disc)), 'hd_cup': float(np.nanmean(hd_cup)), 'cdr_mae': float(np.nanmean(cdr_err))}
