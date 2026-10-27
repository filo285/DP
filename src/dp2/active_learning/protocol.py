"""Freeze the ORIGA/REFUGE experiment and audit image/group overlap."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .annotation import AnnotationStore
from .state import (PUBLIC_COLUMNS, read_csv, relative_source_path, resolve_path,
                    save_json, seed_for, sha256, validate_public, write_state)


def validate_config(config, population=None):
    if config.get("version") != 1:
        raise ValueError("Unsupported active-learning protocol version")
    for key in ("initial_size", "acquisition_size", "acquisitions", "repetitions",
                "mc_passes", "ensemble_members"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    seed_for(config["seed"], "validate")
    if config["mc_passes"] < 2 or config["ensemble_members"] < 2:
        raise ValueError("At least two MC passes and ensemble members required")
    if config["methods"] != ["mc_dropout", "deep_ensembles"] or config["strategies"] != ["random", "bald", "vcdr"]:
        raise ValueError("Protocol requires both methods and the three separate strategies")
    if population is not None and config["initial_size"] + config["acquisition_size"] * config["acquisitions"] > population:
        raise ValueError("Requested annotation budget exceeds training population")


def sample_initial(train, count, seed):
    """Largest-remainder dataset allocation; seeded sampling within each dataset."""
    validate_public(train)
    if type(count) is not int or not 0 < count <= len(train):
        raise ValueError("Initial size must fit the training population")
    groups = [(name, frame.sort_values("image_id")) for name, frame in train.groupby("dataset", sort=True)]
    quotas = np.array([len(frame) * count / len(train) for _, frame in groups])
    sizes = np.floor(quotas).astype(int)
    for index in np.argsort(-(quotas - sizes), kind="stable")[:count - int(sizes.sum())]:
        sizes[index] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for (_, frame), size in zip(groups, sizes):
        selected.extend(rng.choice(frame.image_id.to_numpy(), int(size), replace=False).tolist())
    return train[train.image_id.isin(selected)].reset_index(drop=True)


class Experiment:
    """Frozen public inputs; private annotation tables are opened only on demand."""
    def __init__(self, directory, data_root=None):
        self.directory = Path(directory)
        self.protocol = json.loads((self.directory / "protocol.json").read_text())
        if self.protocol.get("status") != "complete":
            raise ValueError("Experiment preparation is incomplete")
        self.digest = sha256(self.directory / "protocol.json")
        self.config = self.protocol["config"]
        if sha256(self.directory / "audit.json") != self.protocol["audit_sha256"]:
            raise ValueError("Dataset audit changed since preparation")
        self.data_root = Path(data_root or self.protocol["data_root"])
        self.splits = {}
        for name in ("train", "val", "test"):
            path = self.directory / "splits" / f"{name}.csv"
            if sha256(path) != self.protocol["split_sha256"][f"{name}.csv"]:
                raise ValueError(f"Frozen split changed: {name}")
            self.splits[name] = read_csv(path)
            validate_public(self.splits[name])
        validate_config(self.config, len(self.splits["train"]))
        self._annotations = None

    def private_annotations(self, split):
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown annotation split")
        name = f"{split}_annotations.csv"
        path = self.directory / "private" / name
        if sha256(path) != self.protocol["annotation_sha256"][name]:
            raise ValueError("Private annotation manifest changed")
        return read_csv(path)

    @property
    def annotations(self):
        if self._annotations is None:
            self._annotations = AnnotationStore(self.private_annotations("train"), self.splits["train"].image_id)
        return self._annotations


def prepare_experiment(output, paths, config, data_root=None):
    """Audit source images without reading masks, then freeze versioned manifests."""
    validate_config(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(data_root or paths["dp1_root"])
    rows, annotations, source_hashes = [], {}, {}
    for dataset in ("origa", "refuge"):
        table_path = resolve_path(root, paths[f"{dataset}_table"])
        source_hashes[dataset] = sha256(table_path)
        table = read_csv(table_path)
        if not {"image_path", "mask_path"}.issubset(table.columns):
            raise ValueError(f"Missing image/mask columns in {table_path}")
        patient_columns = [c for c in ("patient_id", "subject_id", "person_id") if c in table.columns]
        for index, row in table.iterrows():
            split = row.get("split", "train").lower() if dataset == "refuge" else "train"
            if split not in ("train", "val", "test"):
                raise ValueError(f"Unknown {dataset} split: {split}")
            image_path = relative_source_path(row.image_path, paths["dp1_root"])
            mask_path = relative_source_path(row.mask_path, paths["dp1_root"])
            image_id = f"{dataset}:{row.get('image_id') or Path(image_path).stem}"
            absolute = resolve_path(root, image_path)
            if not resolve_path(root, mask_path).is_file():
                raise FileNotFoundError(f"Missing annotation for {image_id}")
            raw = cv2.imread(str(absolute))
            if raw is None:
                raise ValueError(f"Unreadable image: {absolute}")
            pixel_hash = hashlib.sha256(str(raw.shape).encode() + raw.tobytes()).hexdigest()
            patient = next((row[c] for c in patient_columns if row[c]), "")
            rows.append({"image_id": image_id, "dataset": dataset, "image_path": image_path,
                         "image_sha256": sha256(absolute), "pixel_sha256": pixel_hash,
                         "patient_id": f"{dataset}:{patient}" if patient else "",
                         "split": split, "resolved_path": str(absolute).casefold()})
            annotations.setdefault(split, []).append({"image_id": image_id, "mask_path": mask_path})
            if (index + 1) % 100 == 0:
                print(f"Audit {dataset}: {index + 1}/{len(table)}", flush=True)
    frame = pd.DataFrame(rows)
    failures = {}
    for column in ("image_id", "resolved_path", "image_sha256", "pixel_sha256"):
        duplicate = frame[frame[column].duplicated(keep=False)]
        if not duplicate.empty:
            failures[column] = duplicate[["image_id", "split", column]].loc[:, lambda f: ~f.columns.duplicated()].to_dict("records")
    known = frame[frame.patient_id != ""]
    overlapping = known.groupby("patient_id").split.nunique()
    overlapping = overlapping[overlapping > 1].index.tolist()
    if overlapping:
        failures["patient_overlap"] = overlapping
    audit = {"status": "failed" if failures else "passed", "images": len(frame),
             "split_counts": frame.split.value_counts().to_dict(), "failures": failures,
             "patient_ids_available": len(known),
             "patient_independence": "verified_for_available_ids" if len(known) else "not_verified_no_patient_identifiers",
             "patient_scope": "dataset-qualified source patient_id/subject_id/person_id; cross-dataset identity not established",
             "near_duplicate_audit": "not performed; hashes detect byte-identical or identical decoded images only",
             "annotation_pixels_read": False}
    save_json(output / "audit.json", audit)
    if failures:
        raise ValueError(f"Dataset overlap audit failed; inspect {output / 'audit.json'}")
    splits = {name: frame[frame.split == name][list(PUBLIC_COLUMNS)].reset_index(drop=True)
              for name in ("train", "val", "test")}
    for split in splits.values():
        validate_public(split)
    validate_config(config, len(splits["train"]))
    (output / "splits").mkdir()
    (output / "private").mkdir()
    for name, split in splits.items():
        split.to_csv(output / "splits" / f"{name}.csv", index=False)
        pd.DataFrame(annotations[name]).to_csv(output / "private" / f"{name}_annotations.csv", index=False)
    protocol = {"version": 1, "status": "complete", "config": config,
                "data_root": str(root.resolve()), "source_table_sha256": source_hashes,
                "split_sha256": {p.name: sha256(p) for p in (output / "splits").glob("*.csv")},
                "annotation_sha256": {p.name: sha256(p) for p in (output / "private").glob("*.csv")},
                "audit_sha256": sha256(output / "audit.json")}
    save_json(output / "protocol.json", protocol)
    experiment = Experiment(output)
    for repetition in range(config["repetitions"]):
        labeled = sample_initial(splits["train"], config["initial_size"],
                                 seed_for(config["seed"], "initial", repetition))
        pool = splits["train"][~splits["train"].image_id.isin(labeled.image_id)].reset_index(drop=True)
        write_state(output / f"repeat_{repetition:02d}" / "initial", experiment,
                    labeled, pool, experiment.annotations.reveal(labeled.image_id), repetition, 0)
    return experiment
