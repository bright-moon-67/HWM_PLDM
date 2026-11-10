"""Environment state snapshot / restore for counterfactual candidate replay.

Stage 2 of the research plan compares several high-level candidates *from the
same real state*.  That requires more than resetting the environment to the same
start position: velocities, simulator internals and the environment's own random
state all have to be restored, otherwise the candidates are not paired and the
comparison is confounded.

The wrappers used here are dm_control / gym-mujoco environments where the full
simulator state is available through ``physics.get_state()`` (dm_control) or
``sim.get_state()`` (gym).  This module hides those differences behind
``snapshot_env`` / ``restore_env`` and provides ``verify_snapshot_roundtrip`` so
a run can check *once* that state restoration actually works before trusting a
whole counterfactual experiment.

If no snapshot mechanism is found, :func:`snapshot_env` raises
``NotImplementedError`` with an explicit message: silently falling back to
"reset to the start" would produce unpaired candidates that look like data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class EnvSnapshot:
    """Opaque, restorable image of one environment."""

    kind: str
    state: Any
    numpy_rng: Optional[Tuple[Any, ...]] = None
    gym_rng: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Plain container (``torch.save`` friendly, no dataclass identity)."""
        return {
            "kind": self.kind,
            "state": self.state,
            "numpy_rng": self.numpy_rng,
            "gym_rng": self.gym_rng,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EnvSnapshot":
        return cls(
            kind=payload["kind"],
            state=payload["state"],
            numpy_rng=payload.get("numpy_rng"),
            gym_rng=payload.get("gym_rng"),
            extra=payload.get("extra") or {},
        )


def _iter_env_chain(env: Any) -> Iterator[Any]:
    """Yield ``env`` and every wrapped environment reachable from it."""
    seen = set()
    stack = [env]
    while stack:
        obj = stack.pop()
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for attr in ("env", "unwrapped", "_env"):
            child = getattr(obj, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)


def _find_physics_like(env: Any) -> Optional[Tuple[str, Any]]:
    """Locate a simulator object with ``get_state`` / ``set_state``."""
    for obj in _iter_env_chain(env):
        physics = getattr(obj, "physics", None)
        if physics is not None and callable(
            getattr(physics, "get_state", None)
        ) and callable(getattr(physics, "set_state", None)):
            return "dm_control", physics
        sim = getattr(obj, "sim", None)
        if sim is not None and callable(getattr(sim, "get_state", None)) and callable(
            getattr(sim, "set_state", None)
        ):
            return "gym_mujoco", sim
    return None


def _find_gym_rng(env: Any) -> Optional[Any]:
    for obj in _iter_env_chain(env):
        rng = getattr(obj, "np_random", None)
        if rng is not None and hasattr(rng, "bit_generator"):
            return rng
    return None


def supports_snapshot(env: Any) -> bool:
    """Whether :func:`snapshot_env` can produce a restorable image of ``env``."""
    return _find_physics_like(env) is not None


def snapshot_env(env: Any) -> EnvSnapshot:
    """Capture the full simulator + RNG state of ``env``."""
    found = _find_physics_like(env)
    if found is None:
        raise NotImplementedError(
            "environment exposes neither physics.get_state/set_state nor "
            "sim.get_state/set_state; counterfactual replay needs full state "
            "snapshots (resetting to the start position is not equivalent). "
            "Add snapshot support to the wrapper before collecting candidates."
        )
    kind, simulator = found
    state = simulator.get_state()
    if isinstance(state, np.ndarray):
        state = state.copy()
    rng = _find_gym_rng(env)
    return EnvSnapshot(
        kind=kind,
        state=state,
        numpy_rng=np.random.get_state(),
        gym_rng=(rng.bit_generator.state if rng is not None else None),
    )


def restore_env(env: Any, snapshot: Any) -> None:
    """Restore a state produced by :func:`snapshot_env`."""
    if isinstance(snapshot, dict):
        snapshot = EnvSnapshot.from_dict(snapshot)
    if not isinstance(snapshot, EnvSnapshot):
        raise TypeError(f"unsupported snapshot type {type(snapshot)}")

    found = _find_physics_like(env)
    if found is None:
        raise NotImplementedError("environment does not support state restoration")
    kind, simulator = found
    simulator.set_state(snapshot.state)

    if snapshot.numpy_rng is not None:
        np.random.set_state(snapshot.numpy_rng)
    if snapshot.gym_rng is not None:
        rng = _find_gym_rng(env)
        if rng is not None:
            rng.bit_generator.state = snapshot.gym_rng


def verify_snapshot_roundtrip(
    env: Any,
    *,
    n_probe_steps: int = 1,
    atol: float = 1e-8,
) -> bool:
    """Check that ``snapshot -> random steps -> restore`` is lossless.

    Returns ``True`` when the observation after restoring differs from the
    original by at most ``atol``.  Intended as a one-off validation before a
    collection run; the candidates are worthless if this returns ``False``.
    """
    if not supports_snapshot(env):
        return False
    before = np.asarray(env.get_obs(), dtype=np.float64).copy()
    snapshot = snapshot_env(env)
    try:
        for _ in range(max(int(n_probe_steps), 1)):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            env.step(action)
        restore_env(env, snapshot)
        after = np.asarray(env.get_obs(), dtype=np.float64)
    finally:
        # Even if the probe failed, leave the environment at the snapshot.
        try:
            restore_env(env, snapshot)
        except Exception:  # pragma: no cover - defensive
            pass

    if before.shape != after.shape:
        return False
    return bool(np.allclose(before, after, atol=atol, equal_nan=True))
