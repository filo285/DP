"""Pool inputs with no annotation dependency."""
import cv2
import numpy as np
from torch.utils.data import Dataset

from dp2.baseline import build_transforms
from .state import resolve_path, sha256, validate_public


class ImagePoolDataset(Dataset):
    def __init__(self, frame, cfg, data_root):
        validate_public(frame, allow_empty=True)
        self.frame = frame.reset_index(drop=True).copy()
        self.root = data_root
        _, self.transform = build_transforms(cfg)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = resolve_path(self.root, row.image_path)
        if "image_sha256" in row and sha256(path) != row.image_sha256:
            raise ValueError(f"Image content changed since preparation: {row.image_id}")
        raw = cv2.imread(str(path))
        if raw is None:
            raise FileNotFoundError(path)
        # This mask represents image geometry only; no reference annotation is read.
        value = self.transform(image=cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                               mask=np.ones(raw.shape[:2], dtype=np.uint8))
        return row.image_id, value["image"], value["mask"].bool()


def supervised_frame(public, annotations, root):
    validate_public(public)
    if (set(annotations.columns) != {"image_id", "mask_path"} or
            annotations.image_id.duplicated().any() or
            set(public.image_id) != set(annotations.image_id)):
        raise ValueError("Annotations must match the supervised subset exactly")
    merged = public.merge(annotations, on="image_id", validate="one_to_one", sort=False)
    if "image_sha256" in merged:
        for row in merged.itertuples():
            if sha256(resolve_path(root, row.image_path)) != row.image_sha256:
                raise ValueError(f"Supervised image changed since preparation: {row.image_id}")
    for column in ("image_path", "mask_path"):
        merged[column] = merged[column].map(lambda p: str(resolve_path(root, p)))
    return merged
