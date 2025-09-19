# SPDX-License-Identifier: MIT
from dataclasses import dataclass, field
from typing import Dict, Callable, Iterable, Optional, Set
from pxr import Usd, UsdGeom, Tf, Gf, Sdf
import omni.usd
import math

# ---- Change detection tolerances (tune if needed) ----
POS_EPS = 1e-1       # meters: ignore smaller position jitter
ANG_EPS_DEG = 0.5   # degrees: ignore tiny rotation jitter

def _world_pose(prim: Usd.Prim):
    x = UsdGeom.Xformable(prim)
    if not x:
        return Gf.Vec3d(0,0,0), Gf.Quatd(1,0,0,0)
    M = x.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return M.ExtractTranslation(), M.ExtractRotation().GetQuat()

def _quat_dot(a: Gf.Quatd, b: Gf.Quatd) -> float:
    """Dot product between two (assumed unit) quaternions."""
    ar = a.GetReal()
    ai = a.GetImaginary()  # Gf.Vec3d
    br = b.GetReal()
    bi = b.GetImaginary()
    return ar * br + ai[0] * bi[0] + ai[1] * bi[1] + ai[2] * bi[2]

def _pose_changed(old: dict | None, new_pos: Gf.Vec3d, new_q: Gf.Quatd) -> bool:
    """Return True if pose changed beyond thresholds."""
    if not old:
        return True

    op: Gf.Vec3d = old["pos"]
    oq: Gf.Quatd = old["quat"]

    # position delta
    if (new_pos - op).GetLength() > POS_EPS:
        return True

    # rotation delta (via quaternion dot)
    dot = abs(_quat_dot(oq, new_q))          # handle q vs -q equivalence
    dot = max(-1.0, min(1.0, dot))           # clamp for numeric safety
    ang_deg = 2.0 * math.degrees(math.acos(dot))
    return ang_deg > ANG_EPS_DEG

@dataclass
class PoseCache:
    poses: Dict[str, Dict[str, object]] = field(default_factory=dict)
    def set_pose(self, path: str, pos: Gf.Vec3d, quat: Gf.Quatd):
        self.poses[path] = {"pos": pos, "quat": quat}
    def get(self, path: str):
        return self.poses.get(path)

def _is_under_or_equal(ancestor_prim_path: str, descendant_prim_path: str) -> bool:
    if descendant_prim_path == ancestor_prim_path:
        return True
    return descendant_prim_path.startswith(ancestor_prim_path.rstrip("/") + "/")

class RealtimeTracker:
    """
    Polls each physics tick and listens to USD change notices.
    Calls on_update(path, pose_dict) **only when the pose actually changed**.
    """
    def __init__(self, paths_to_track: Iterable[str],
                 on_update: Optional[Callable[[str, dict], None]] = None,
                 verbose: bool = False):
        self.paths: Set[str] = set(paths_to_track)
        self.on_update = on_update
        self.verbose = verbose
        self.cache = PoseCache()
        self._stage: Optional[Usd.Stage] = None
        self._usd_notice_reg = None
        self._usd_callback_fn = None
        self._poll_enabled = False
        self._poll_name = None

    # ---- lifecycle ----
    def start(self, add_physics_callback_fn: Callable, name: str = "realtime_tracker"):
        if self._poll_enabled:
            return
        self._stage = omni.usd.get_context().get_stage()
        if self._stage is None:
            raise RuntimeError("USD stage not available.")
        self._poll_name = name
        self._refresh_all()
        self._register_usd_notice()
        add_physics_callback_fn(self._poll_name, callback_fn=self._on_sim_step)
        self._poll_enabled = True
        if self.verbose:
            print("[RealtimeTracker] started")

    def stop(self, remove_physics_callback_fn: Callable, name: Optional[str] = None):
        poll_name = name or self._poll_name
        if self._poll_enabled and poll_name:
            try:
                remove_physics_callback_fn(poll_name)
            except Exception:
                pass
        self._poll_enabled = False
        self._poll_name = None
        self._unregister_usd_notice()
        if self.verbose:
            print("[RealtimeTracker] stopped")

    # ---- polling (physics) ----
    def _on_sim_step(self, dt: float):
        stage = self._stage or omni.usd.get_context().get_stage()
        touched = 0
        for pth in list(self.paths):
            prim = stage.GetPrimAtPath(pth)
            if not prim or not prim.IsValid():
                continue
            pos, quat = _world_pose(prim)
            prev = self.cache.get(pth)
            if _pose_changed(prev, pos, quat):
                self.cache.set_pose(pth, pos, quat)
                touched += 1
                if self.on_update:
                    self.on_update(pth, self.cache.get(pth))
        if self.verbose and touched:
            print(f"[RealtimeTracker] poll updated {touched} prim(s)")

    def _refresh_all(self):
        stage = self._stage or omni.usd.get_context().get_stage()
        for pth in list(self.paths):
            prim = stage.GetPrimAtPath(pth)
            if prim and prim.IsValid():
                pos, quat = _world_pose(prim)
                self.cache.set_pose(pth, pos, quat)

    # ---- USD notices (UI edits) ----
    def _register_usd_notice(self):
        stage = self._stage
        if not stage:
            return
        self._usd_callback_fn = self._on_usd_changed  # strong ref
        self._usd_notice_reg = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._usd_callback_fn, stage)

    def _unregister_usd_notice(self):
        if self._usd_notice_reg:
            Tf.Notice.Revoke(self._usd_notice_reg)
            self._usd_notice_reg = None
        self._usd_callback_fn = None

    def _on_usd_changed(self, notice: Usd.Notice.ObjectsChanged, sender):
        if not self._stage:
            return
        # Normalize property paths to prim paths
        changed_prim_paths: Set[str] = set()
        for p in notice.GetChangedInfoOnlyPaths():
            changed_prim_paths.add(str(Sdf.Path(p).GetPrimPath()))
        for p in notice.GetResyncedPaths():
            changed_prim_paths.add(str(Sdf.Path(p).GetPrimPath()))
        if not changed_prim_paths:
            return

        stage = self._stage
        touched = 0
        for tracked in list(self.paths):
            if any(_is_under_or_equal(ch, tracked) for ch in changed_prim_paths):
                prim = stage.GetPrimAtPath(tracked)
                if prim and prim.IsValid():
                    pos, quat = _world_pose(prim)
                    prev = self.cache.get(tracked)
                    if _pose_changed(prev, pos, quat):
                        self.cache.set_pose(tracked, pos, quat)
                        touched += 1
                        if self.on_update:
                            self.on_update(tracked, self.cache.get(tracked))
        if self.verbose and touched:
            print(f"[RealtimeTracker] notice updated {touched} prim(s)")
