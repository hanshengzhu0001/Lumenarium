#!/usr/bin/env python3
"""Safely compact legacy ``processed_masks.pt`` tensor storage.

Older Imaginarium runs serialized ``cropped_data["images"][:, 3]`` directly.
That tensor is a strided alpha-channel view, so ``torch.save`` retained the
entire four-channel RGBA backing storage.  This migration writes a contiguous
tensor with identical shape, dtype and values.

Each file is written and verified beside the original, then atomically replaces
it.  An interruption before ``os.replace`` leaves the original untouched.
Already-compact files are detected from tensor storage size and skipped, making
the command safe to resume.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any

import torch


def storage_nbytes(tensor: torch.Tensor) -> int:
    """Return the serialized backing-storage size used by a tensor."""
    if hasattr(tensor, "untyped_storage"):
        return int(tensor.untyped_storage().nbytes())
    return int(tensor.storage().size() * tensor.element_size())


def logical_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def tensors_identical(first: torch.Tensor, second: torch.Tensor) -> bool:
    if first.shape != second.shape or first.dtype != second.dtype:
        return False
    if torch.equal(first, second):
        return True
    if first.is_floating_point() or first.is_complex():
        equal_or_nan = torch.eq(first, second) | (
            torch.isnan(first) & torch.isnan(second)
        )
        return bool(torch.all(equal_or_nan).item())
    return False


def load_tensor(path: Path) -> torch.Tensor:
    value: Any = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{path} contains {type(value).__name__}, expected torch.Tensor"
        )
    if value.layout != torch.strided:
        raise TypeError(f"{path} uses unsupported tensor layout {value.layout}")
    return value


def compact_one(path: Path, *, verify: bool = True) -> dict[str, Any]:
    before_stat = path.stat()
    before_file_bytes = int(before_stat.st_size)
    original = load_tensor(path)
    original_storage_bytes = storage_nbytes(original)
    original_logical_bytes = logical_nbytes(original)

    result: dict[str, Any] = {
        "path": str(path),
        "shape": list(original.shape),
        "dtype": str(original.dtype),
        "before_file_bytes": before_file_bytes,
        "before_storage_bytes": original_storage_bytes,
        "logical_bytes": original_logical_bytes,
    }
    if original_storage_bytes <= original_logical_bytes:
        result.update(
            {
                "status": "already_compact",
                "after_file_bytes": before_file_bytes,
                "reclaimed_bytes": 0,
            }
        )
        return result

    # ``contiguous()`` may return the same tensor when a contiguous slice has
    # a non-zero offset into a larger storage.  ``clone`` guarantees ownership
    # of a new logical-size storage in both contiguous and strided-view cases.
    compact = original.detach().clone(
        memory_format=torch.contiguous_format
    )
    if storage_nbytes(compact) != logical_nbytes(compact):
        raise RuntimeError(f"failed to create compact storage for {path}")
    if not tensors_identical(original, compact):
        raise RuntimeError(f"in-memory value check failed for {path}")

    temporary = path.with_name(
        f".{path.name}.compact-{os.getpid()}-{time.time_ns()}.tmp"
    )
    try:
        torch.save(compact, temporary)
        os.chmod(temporary, stat.S_IMODE(before_stat.st_mode))
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())

        if verify:
            reloaded = load_tensor(temporary)
            if storage_nbytes(reloaded) != logical_nbytes(reloaded):
                raise RuntimeError(
                    f"temporary file still has aliased storage: {temporary}"
                )
            if not tensors_identical(original, reloaded):
                raise RuntimeError(
                    f"round-trip value check failed for {temporary}"
                )
            del reloaded

        after_file_bytes = int(temporary.stat().st_size)
        os.replace(temporary, path)
        result.update(
            {
                "status": "compacted",
                "after_file_bytes": after_file_bytes,
                "reclaimed_bytes": before_file_bytes - after_file_bytes,
            }
        )
        return result
    finally:
        if temporary.exists():
            temporary.unlink()


def discover(root: Path) -> list[Path]:
    return sorted(root.rglob("processed_masks.pt"))


def format_gib(value: int) -> str:
    return f"{value / (1024 ** 3):.3f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Atomically compact legacy Imaginarium processed mask tensors."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root containing per-asset processed_masks.pt files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect tensors and report reclaimable storage without writing.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Process at most this many discovered files; zero means all.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip reloading each temporary file before replacement.",
    )
    parser.add_argument(
        "--jsonl-log",
        type=Path,
        help="Append one machine-readable result per inspected file.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"root is not a directory: {root}")
    if args.max_files < 0:
        parser.error("--max-files must be non-negative")

    paths = discover(root)
    if args.max_files:
        paths = paths[: args.max_files]
    print(
        f"DISCOVERED={len(paths)} ROOT={root} "
        f"MODE={'dry-run' if args.dry_run else 'compact'}",
        flush=True,
    )

    log_handle = None
    if args.jsonl_log:
        args.jsonl_log.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.jsonl_log.open("a", encoding="utf-8", buffering=1)

    total_before = 0
    total_after = 0
    total_reclaimed = 0
    compacted = 0
    skipped = 0
    failed = 0
    started = time.monotonic()
    try:
        for index, path in enumerate(paths, start=1):
            try:
                if args.dry_run:
                    tensor = load_tensor(path)
                    before = int(path.stat().st_size)
                    storage = storage_nbytes(tensor)
                    logical = logical_nbytes(tensor)
                    estimated_after = (
                        before
                        if storage <= logical
                        else max(0, int(before * logical / storage))
                    )
                    result = {
                        "path": str(path),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "before_file_bytes": before,
                        "before_storage_bytes": storage,
                        "logical_bytes": logical,
                        "status": (
                            "already_compact" if storage <= logical else "reclaimable"
                        ),
                        "after_file_bytes": estimated_after,
                        "reclaimed_bytes": before - estimated_after,
                    }
                else:
                    result = compact_one(path, verify=not args.no_verify)

                total_before += int(result["before_file_bytes"])
                total_after += int(result["after_file_bytes"])
                total_reclaimed += int(result["reclaimed_bytes"])
                if result["status"] == "compacted":
                    compacted += 1
                else:
                    skipped += 1
                print(
                    f"[{index}/{len(paths)}] {result['status'].upper()} "
                    f"saved={format_gib(int(result['reclaimed_bytes']))} "
                    f"total_saved={format_gib(total_reclaimed)} "
                    f"{path.parent.name}",
                    flush=True,
                )
            except Exception as error:
                failed += 1
                result = {
                    "path": str(path),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(
                    f"[{index}/{len(paths)}] FAILED {path}: {result['error']}",
                    file=sys.stderr,
                    flush=True,
                )

            if log_handle:
                log_handle.write(
                    json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                )
    except KeyboardInterrupt:
        print("\nINTERRUPTED: current original remains intact; rerun to resume.", flush=True)
        return 130
    finally:
        if log_handle:
            log_handle.close()

    elapsed = time.monotonic() - started
    print(
        "SUMMARY "
        f"files={len(paths)} compacted={compacted} skipped={skipped} "
        f"failed={failed} before={format_gib(total_before)} "
        f"after={format_gib(total_after)} reclaimed={format_gib(total_reclaimed)} "
        f"elapsed_seconds={elapsed:.1f}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
