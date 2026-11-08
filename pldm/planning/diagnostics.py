"""Diagnostics plumbing for the execution-aware HWM research line.

Step 0 only ships the recorder: it is **not** wired into the planning loop yet.
``TwoLvlPlanner`` already stores the ``ExecRepairConfig`` and the chunk offset,
so Step 1 can call this recorder right after the low-level plan is produced.

Design constraints:

* never write full prediction tensors to disk (they are large); only scalars
  and per-task vectors are serialised;
* one JSONL line per planning decision so partial / crashed runs stay readable;
* no dependency beyond ``torch`` for turning tensors into python floats.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

ScalarLike = Union[int, float, bool, torch.Tensor]


def to_jsonable(value: Any) -> Any:
    """Convert tensors / containers into JSON-serialisable python objects."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        if tensor.dtype == torch.bool:
            return [bool(x) for x in tensor.flatten().tolist()]
        if tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            return [int(x) for x in tensor.flatten().tolist()]
        return [float(x) for x in tensor.flatten().tolist()]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if hasattr(value, "tolist"):  # numpy scalars / arrays
        return to_jsonable(value.tolist())
    return str(value)


class DiagnosticsRecorder:
    """Append-only JSONL recorder for planning diagnostics.

    Parameters
    ----------
    exec_cfg:
        ``pldm.planning.d4rl.enums.ExecRepairConfig`` instance. Typed as
        ``Any`` on purpose to keep this module free of import cycles.
    output_dir:
        Where the JSONL file lives. Falls back to ``exec_cfg.output_dir`` and
        finally to "disabled" (``path`` is ``None``).
    chunk_offset:
        Offset of the current chunk inside ``MPCEvaluator.envs``. Global task
        ids are ``exec_cfg.task_id_offset + chunk_offset + env_index``.
    enabled:
        Explicit override used by unit tests; ``None`` follows the config
        (``enabled and log_diagnostics``).
    """

    def __init__(
        self,
        exec_cfg: Any = None,
        output_dir: Optional[Union[str, Path]] = None,
        chunk_offset: int = 0,
        filename: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.exec_cfg = exec_cfg
        self.chunk_offset = int(chunk_offset)
        self.task_id_offset = int(getattr(exec_cfg, "task_id_offset", 0) or 0)
        self._enabled_override = enabled

        if output_dir is None:
            cfg_dir = getattr(exec_cfg, "output_dir", "") if exec_cfg else ""
            output_dir = cfg_dir or None
        self.output_dir = Path(output_dir) if output_dir else None

        self.filename = (
            filename
            or getattr(exec_cfg, "diagnostics_filename", None)
            or "diagnostics.jsonl"
        )
        self._handle = None

    # ------------------------------------------------------------------ config
    @property
    def enabled(self) -> bool:
        if self._enabled_override is not None:
            return bool(self._enabled_override)
        cfg = self.exec_cfg
        if cfg is None:
            return False
        return bool(
            getattr(cfg, "enabled", False) and getattr(cfg, "log_diagnostics", False)
        )

    @property
    def path(self) -> Optional[Path]:
        if self.output_dir is None:
            return None
        return self.output_dir / self.filename

    def task_ids(self, batch_size: int) -> List[int]:
        """Global task ids covered by the current chunk."""
        base = self.task_id_offset + self.chunk_offset
        return [base + i for i in range(int(batch_size))]

    # ----------------------------------------------------------------- writing
    def record(
        self,
        event: str,
        *,
        batch_size: Optional[int] = None,
        task_ids: Optional[Sequence[int]] = None,
        step: Optional[int] = None,
        level: Optional[Union[int, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one diagnostics record. No-op when disabled."""
        if not self.enabled:
            return

        payload: Dict[str, Any] = {"time": round(time.time(), 3), "event": str(event)}
        if level is not None:
            payload["level"] = level
        if step is not None:
            payload["step"] = int(step)
        if task_ids is not None:
            payload["task_ids"] = [int(t) for t in task_ids]
        elif batch_size is not None:
            payload["task_ids"] = self.task_ids(batch_size)
        if metrics:
            payload.update({str(k): to_jsonable(v) for k, v in metrics.items()})

        handle = self._ensure_open()
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def _ensure_open(self):
        if self._handle is not None:
            return self._handle
        path = self.path
        if path is None:
            raise RuntimeError(
                "DiagnosticsRecorder has no output directory; set "
                "exec_repair.output_dir or pass output_dir explicitly."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        return self._handle

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # ---------------------------------------------------------------- context
    def __enter__(self) -> "DiagnosticsRecorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"DiagnosticsRecorder(enabled={self.enabled}, "
            f"path={self.path}, task_id_offset={self.task_id_offset}, "
            f"chunk_offset={self.chunk_offset})"
        )
