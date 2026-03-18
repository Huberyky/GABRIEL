from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)


def strip_reasoning_tags(payload: Any) -> Any:
    """Remove vendor reasoning tags such as ``<think>...</think>`` recursively."""

    if isinstance(payload, str):
        return THINK_TAG_RE.sub("", payload).strip()
    if isinstance(payload, list):
        return [strip_reasoning_tags(item) for item in payload]
    if isinstance(payload, dict):
        return {key: strip_reasoning_tags(value) for key, value in payload.items()}
    return payload


def prompt_hash(*parts: Optional[str], extra: Optional[Mapping[str, Any]] = None) -> str:
    payload = {
        "parts": [str(part or "") for part in parts],
        "extra": dict(extra or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_run_metadata(
    save_dir: str,
    *,
    task_name: str,
    model: str,
    prompt_hash_value: str,
    prompt_language: str,
    additional_instructions: Optional[str],
    incremental: bool,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    path = Path(save_dir) / "run_metadata.json"
    metadata = {
        "task": task_name,
        "model": model,
        "prompt_hash": prompt_hash_value,
        "prompt_language": prompt_language,
        "additional_instructions": additional_instructions or "",
        "incremental": bool(incremental),
    }
    if extra:
        metadata.update(extra)
    warning = None
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prior = None
        if isinstance(prior, dict) and prior.get("prompt_hash") not in {None, prompt_hash_value}:
            warning = (
                f"[{task_name}] WARNING: prompt hash changed from {prior.get('prompt_hash')} "
                f"to {prompt_hash_value}. Existing cached files may not match the current prompt configuration."
            )
            metadata["previous_prompt_hash"] = prior.get("prompt_hash")
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if warning:
        print(warning)
    return metadata


def load_incremental_cache(save_dir: str, file_name: str) -> Optional[Path]:
    base = Path(save_dir) / f"{Path(file_name).stem}_cleaned.csv"
    return base if base.exists() else None
