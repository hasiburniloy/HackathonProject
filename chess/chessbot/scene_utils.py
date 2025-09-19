# SPDX-License-Identifier: MIT
import os
from pxr import Usd, UsdGeom, Gf
import omni.usd

def find_chess_usd(path_dir: str, candidates) -> str:
    if not os.path.isdir(path_dir):
        raise FileNotFoundError(f"ASSET_DIR not found: {path_dir}")
    for name in candidates:
        full = os.path.join(path_dir, name)
        if os.path.isfile(full):
            return full.replace("\\", "/")
    for f in os.listdir(path_dir):
        if f.lower().endswith(".usd"):
            return os.path.join(path_dir, f).replace("\\", "/")
    raise FileNotFoundError(f"No .usd found in {path_dir}")

def world_bbox_center(stage: Usd.Stage, prim: Usd.Prim) -> Gf.Vec3d:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
    return cache.ComputeWorldBound(prim).ComputeCentroid()

def set_camera_pose_matrix(cam_path: str, eye: Gf.Vec3d, target: Gf.Vec3d, up: Gf.Vec3d = Gf.Vec3d(0, 1, 0)):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(cam_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Invalid camera prim: {cam_path}")

    fwd = (target - eye)
    if fwd.GetLength() < 1e-9:
        fwd = Gf.Vec3d(0, 0, -1)
    fwd = fwd.GetNormalized()

    right = Gf.Cross(fwd, up)
    if right.GetLength() < 1e-9:
        up = Gf.Vec3d(0, 0, 1)
        right = Gf.Cross(fwd, up)
    right = right.GetNormalized()
    true_up = Gf.Cross(right, fwd)
    back = -fwd

    rot3 = Gf.Matrix3d(
        right[0],  true_up[0],  back[0],
        right[1],  true_up[1],  back[1],
        right[2],  true_up[2],  back[2],
    )
    xf = Gf.Matrix4d(1.0)
    xf.SetRotateOnly(rot3)
    xf.SetTranslateOnly(eye)

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xop = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    xop.Set(xf)

def analyze_prims_under(stage: Usd.Stage, root_path: str):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    def walk(p):
        yield p
        for c in p.GetChildren():
            yield from walk(c)
    center = world_bbox_center(stage, root)
    out = []
    for prim in walk(root):
        x = UsdGeom.Xformable(prim) if prim else None
        if x:
            M = x.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            pos = M.ExtractTranslation()
        else:
            pos = Gf.Vec3d(0,0,0)
        rel = pos - center
        out.append({
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "world_position": pos,
            "relative_position": rel,
            "distance_from_center": rel.GetLength()
        })
    return out
