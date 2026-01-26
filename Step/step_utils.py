"""Shared helpers for chaining step scripts."""
from __future__ import annotations

import os
import shutil
from typing import Optional


def resolve_input_path(
    input_npz: Optional[str],
    fallback_path: str,
) -> str:
    if input_npz:
        return input_npz
    if not os.path.exists(fallback_path):
        raise FileNotFoundError(f"Expected input not found: {fallback_path}")
    return fallback_path


def copy_as_input(source_path: str, output_dir: str, input_name: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    dest_path = os.path.join(output_dir, input_name)
    shutil.copyfile(source_path, dest_path)
    return dest_path


def copy_output_to_next(output_path: str, next_dir: Optional[str], next_input_name: str) -> Optional[str]:
    if not next_dir:
        return None
    os.makedirs(next_dir, exist_ok=True)
    dest_path = os.path.join(next_dir, next_input_name)
    shutil.copyfile(output_path, dest_path)
    return dest_path
