"""Verify installed dependencies, GPU inference and existing DP1 data paths."""
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations
import cv2
import ipykernel
import matplotlib
import numpy
import openpyxl
import pandas
import scipy
import segmentation_models_pytorch as smp
import torch
import torchvision

paths = json.loads((ROOT / "configs/paths.json").read_text())
data_root = Path(paths["dp1_root"])
assert sys.prefix != sys.base_prefix, "Use the project .venv interpreter"
assert torch.cuda.is_available(), "CUDA is unavailable"
print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)

for key, expected in [("origa_table", 650), ("refuge_table", 1200)]:
    with (data_root / paths[key]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected, (key, len(rows))
    for row in rows:
        for column in ("image_path", "mask_path"):
            assert Path(row[column]).is_file(), row[column]
    print(key, ":", len(rows), "rows; all image/mask paths exist")
assert (data_root / paths["baseline_checkpoint"]).is_file()

# No pretrained-weight download, checkpoint mutation, or training.
model = smp.Unet(encoder_name="resnet34", encoder_weights=None, classes=3)
model = model.cuda().eval()
with torch.inference_mode():
    output = model(torch.zeros(1, 3, 128, 128, device="cuda"))
assert tuple(output.shape) == (1, 3, 128, 128)
assert torch.isfinite(output).all()
print("U-Net CUDA forward pass: PASS")
print("Environment check: PASS (not a reproduction of baseline metrics)")
