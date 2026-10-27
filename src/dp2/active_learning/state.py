"""Strict public manifests, portable paths and reproducible identities."""
import hashlib
import json
import os
import uuid
from pathlib import Path, PureWindowsPath

import pandas as pd

PUBLIC_COLUMNS = ("image_id", "dataset", "image_path", "image_sha256", "pixel_sha256", "patient_id")
REQUIRED_COLUMNS = ("image_id", "dataset", "image_path")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_public(frame, allow_empty=False):
    extra = set(frame.columns) - set(PUBLIC_COLUMNS)
    if extra or not set(REQUIRED_COLUMNS).issubset(frame.columns):
        raise ValueError(f"Invalid public manifest columns (annotation fields forbidden): {sorted(extra)}")
    if frame.empty and not allow_empty:
        raise ValueError("Empty image manifest")
    if frame.image_id.isna().any() or frame.image_id.astype(str).str.strip().eq("").any() or frame.image_id.duplicated().any():
        raise ValueError("Missing or duplicate image IDs")
    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {column}")


def ids_hash(ids):
    values = list(ids)
    if len(values) != len(set(values)):
        raise ValueError("Duplicate IDs")
    return hashlib.sha256(json.dumps(sorted(values), ensure_ascii=True).encode()).hexdigest()


def seed_for(base, *parts):
    if isinstance(base, bool) or not isinstance(base, int) or not 0 <= base < 2**32:
        raise ValueError("Seed must be a nonnegative 32-bit integer")
    raw = json.dumps([base, *parts], ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def relative_source_path(value, source_root):
    """Convert legacy Windows paths even when preparing on Linux."""
    value, source_root = str(value), str(source_root)
    if PureWindowsPath(value).is_absolute():
        return PureWindowsPath(value).relative_to(PureWindowsPath(source_root)).as_posix()
    path = Path(value)
    if path.is_absolute():
        return path.relative_to(Path(source_root)).as_posix()
    return value.replace("\\", "/")


def resolve_path(root, relative):
    if PureWindowsPath(str(relative)).is_absolute() or Path(relative).is_absolute():
        raise ValueError("Expected a path relative to the dataset root")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset path escapes its configured root")
    return path


def validate_partition(train, labeled, pool, val, test):
    frames = (train, labeled, pool, val, test)
    for frame in frames:
        validate_public(frame, allow_empty=frame is pool)
    t, l, u, v, x = (set(f.image_id) for f in frames)
    if l & u or l | u != t or t & (v | x) or v & x:
        raise ValueError("Invalid or overlapping train/labeled/pool/validation/test partition")
    reference = train.set_index("image_id")
    for frame in (labeled, pool):
        expected = reference.loc[frame.image_id].reset_index()
        if list(frame.columns) != list(train.columns) or not expected.equals(frame.reset_index(drop=True)):
            raise ValueError("Subset metadata differs from frozen training population")


def write_state(output, experiment, labeled, pool, annotations, repetition, round_index,
                smoke=False, acquisition=None):
    output = Path(output)
    validate_partition(experiment.splits["train"], labeled, pool,
                       experiment.splits["val"], experiment.splits["test"])
    expected = experiment.annotations.reveal(labeled.image_id)
    if not expected.equals(annotations.reset_index(drop=True)):
        raise ValueError("Revealed annotations do not match labeled IDs")
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in (("labeled", labeled), ("unlabeled", pool), ("labeled_annotations", annotations)):
        frame.to_csv(output / f"{name}.csv", index=False)
    state = {"version": 1, "experiment_sha256": experiment.digest,
             "repetition": repetition, "round": round_index,
             "smoke": bool(smoke or experiment.config.get("smoke", False)),
             "acquisition": acquisition,
             "labeled_ids_sha256": ids_hash(labeled.image_id),
             "pool_ids_sha256": ids_hash(pool.image_id),
             "labeled_count": len(labeled), "pool_count": len(pool),
             "files": {p.name: sha256(p) for p in output.glob("*.csv")}}
    save_json(output / "state.json", state)
    return state


def load_state(experiment, directory):
    directory = Path(directory)
    state = json.loads((directory / "state.json").read_text())
    if state["experiment_sha256"] != experiment.digest:
        raise ValueError("State belongs to a different experiment")
    if (type(state["repetition"]) is not int or not 0 <= state["repetition"] < experiment.config["repetitions"]
            or type(state["round"]) is not int or state["round"] < 0
            or type(state.get("smoke", False)) is not bool):
        raise ValueError("Invalid state repetition, round or smoke flag")
    required = {"labeled.csv", "unlabeled.csv", "labeled_annotations.csv"}
    if set(state["files"]) != required:
        raise ValueError("Incomplete state manifest")
    for name, digest in state["files"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"State file changed: {name}")
    labeled, pool, annotations = (read_csv(directory / name) for name in
                                  ("labeled.csv", "unlabeled.csv", "labeled_annotations.csv"))
    validate_partition(experiment.splits["train"], labeled, pool,
                       experiment.splits["val"], experiment.splits["test"])
    if (ids_hash(labeled.image_id) != state["labeled_ids_sha256"] or
            ids_hash(pool.image_id) != state["pool_ids_sha256"] or
            len(labeled) != state["labeled_count"] or len(pool) != state["pool_count"]):
        raise ValueError("State membership hash/count mismatch")
    if not experiment.annotations.reveal(labeled.image_id).equals(annotations):
        raise ValueError("Annotation records do not match labeled membership")
    return state, labeled, pool, annotations
