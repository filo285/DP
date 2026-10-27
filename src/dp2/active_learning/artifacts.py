"""Atomic, verified stages and an OS-released single-writer study lock."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import uuid
import time

from .state import save_json, sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_path(root, value):
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact reference escapes study directory")
    return path


def verify_stage(path):
    path = Path(path)
    seal = read_json(path / "_complete.json")
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and p.name != "_complete.json"}
    if actual != set(seal["files"]):
        raise ValueError(f"Stage file set changed: {path}")
    for name, digest in seal["files"].items():
        if sha256(local_path(path, name)) != digest:
            raise ValueError(f"Completed artifact changed: {path / name}")
    return seal


def seal_stage(path):
    path = Path(path)
    save_json(path / "_complete.json", {"version": 1, "files": {
        p.relative_to(path).as_posix(): sha256(p) for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "_complete.json"}})


def archive_partial(path):
    """Retain interrupted work alongside its original location; never delete it."""
    path = Path(path).resolve()
    destination = path.with_name(path.name + ".interrupted-" + uuid.uuid4().hex)
    if destination.parent != path.parent:
        raise ValueError("Archive must remain in the same directory")
    rename_directory(path, destination)


def rename_directory(source, destination):
    """Windows scanners can briefly retain directory handles after file writes."""
    for attempt in range(10):
        try:
            return Path(source).rename(destination)
        except PermissionError as error:
            if os.name != "nt" or getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 9:
                raise
            time.sleep(min(.05 * 2**attempt, 1.))


def stage(path, build, resumable=False):
    """Publish complete directories by rename; never replay a published selection."""
    path = Path(path)
    if path.exists():
        verify_stage(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial")
    if pending.exists() and (pending / "_complete.json").exists():
        verify_stage(pending)
    else:
        if pending.exists() and not resumable:
            archive_partial(pending)
        build(pending)
        seal_stage(pending)
    rename_directory(pending, path)
    return path


@contextmanager
def study_lock(root):
    """OS advisory lock automatically releases even when a process is terminated."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / ".writer.lock").open("a+b")
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        stream.close()
        raise RuntimeError("Another process owns this study; use a separate output directory") from error
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()
