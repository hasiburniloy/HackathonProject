# SPDX-License-Identifier: MIT
import os
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import omni.usd

from .scene_utils import world_bbox_center
from .config import A1_CORNER, DEFAULT_PICK_LIFT

FILE_NAMES_FOR_BOARD = {"board", "chessboard", "squares", "tiles", "base", "frame"}

# ---------- Color extraction helpers (GetAttr()-safe) ----------
def _get_display_color(prim: Usd.Prim):
    try:
        img = UsdGeom.Imageable(prim)
        if not img:
            return None
        pv = img.GetPrimvar("displayColor")
        if not pv:
            return None
        attr = pv.GetAttr()
        if attr and attr.HasAuthoredValueOpinion():
            vals = pv.Get()
            if vals and len(vals) > 0:
                c = vals[0]
                return (float(c[0]), float(c[1]), float(c[2]))
    except Exception:
        pass
    return None

def _get_bound_material(prim: Usd.Prim):
    try:
        return UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    except Exception:
        return None

def _get_basecolor_from_shader(shader: UsdShade.Shader):
    if not shader:
        return None, None

    def _as_path(val):
        try:
            if isinstance(val, Sdf.AssetPath):
                return val.path
        except Exception:
            pass
        return str(val) if val is not None else None

    def _probe(names):
        for n in names:
            inp = shader.GetInput(n)
            if not inp:
                continue
            if inp.HasConnectedSource():
                src, _, _ = inp.GetConnectedSource()
                if isinstance(src, UsdShade.Shader):
                    for tex_name in ("file", "inputs:file", "inputs:filename", "texture:file"):
                        ti = src.GetInput(tex_name)
                        if ti:
                            attr = ti.GetAttr()
                            if attr and attr.HasAuthoredValueOpinion():
                                return None, _as_path(ti.Get())
            attr = inp.GetAttr()
            if attr and attr.HasAuthoredValueOpinion():
                v = inp.Get()
                try:
                    return (float(v[0]), float(v[1]), float(v[2])), None
                except Exception:
                    try:
                        f = float(v); return (f, f, f), None
                    except Exception:
                        pass
        return None, None

    return _probe([
        "baseColor", "diffuseColor",
        "albedo_color",
        "inputs:base_color_constant", "inputs:diffuse_color_constant",
        "tint_color", "color"
    ])

def _average_texture_color(path: str):
    try:
        if not path or str(path).startswith("omniverse://") or (not os.path.isfile(path)):
            return None
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((256, 256))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = arr.reshape(-1, 3).mean(axis=0)
        return (float(mean[0]), float(mean[1]), float(mean[2]))
    except Exception:
        return None

def get_approx_base_color(prim: Usd.Prim):
    mat = _get_bound_material(prim)
    if mat:
        surf_out = mat.GetSurfaceOutput()
        shader = surf_out.GetConnectedSource()[0] if (surf_out and surf_out.HasConnectedSource()) else None
        rgb, tex = _get_basecolor_from_shader(shader)
        if rgb is not None:
            return {"rgb": rgb, "source": "material-const", "texture": None, "material_path": str(mat.GetPath())}
        if tex:
            trgb = _average_texture_color(tex)
            if trgb is not None:
                return {"rgb": trgb, "source": "material-texture", "texture": tex, "material_path": str(mat.GetPath())}
    dc = _get_display_color(prim)
    if dc is not None:
        return {"rgb": dc, "source": "displayColor", "texture": None, "material_path": str(mat.GetPath()) if mat else None}
    return {"rgb": None, "source": "unknown", "texture": None, "material_path": str(mat.GetPath()) if mat else None}

# ---------- Board & Piece mapping ----------
def _get_world_pose(stage: Usd.Stage, prim: Usd.Prim):
    x = UsdGeom.Xformable(prim)
    if not x:
        return Gf.Vec3d(0,0,0), Gf.Quatd(1,0,0,0)
    M = x.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p = M.ExtractTranslation()
    R = M.ExtractRotation().GetQuat()
    return p, R

def _name_hint_color(prim: Usd.Prim):
    n = prim.GetName().lower()
    if any(k in n for k in ["white", "w_","w-","_w","-w"]): return "white"
    if any(k in n for k in ["black", "b_","b-","_b","-b"]): return "black"
    return None

def _piece_type_from_name(prim: Usd.Prim):
    n = prim.GetName().lower()
    for t in ["king","queen","rook","bishop","knight","pawn","k","q","r","b","n","p"]:
        if f"_{t}" in n or n.startswith(t) or n.endswith(t):
            m = {"k":"king","q":"queen","r":"rook","b":"bishop","n":"knight","p":"pawn"}
            return m.get(t, t)
    return "unknown"

def _label_from_rgb(rgb, fallback="unknown", threshold=0.5):
    if rgb is None:
        return fallback
    r,g,b = rgb
    Y = 0.2126*r + 0.7152*g + 0.0722*b
    return "white" if Y >= threshold else "black"

class BoardGrid:
    """Builds a1..h8 world poses and assigns pieces to nearest squares; also extracts color info."""
    def __init__(self, stage: Usd.Stage, chess_root_path: str,
                 a1_corner: str = A1_CORNER, lift_above_square: float = DEFAULT_PICK_LIFT):
        self.stage = stage
        self.root_path = chess_root_path
        self.a1_corner = a1_corner
        self.lift = float(lift_above_square)

        self.root_prim = stage.GetPrimAtPath(chess_root_path)
        if not self.root_prim or not self.root_prim.IsValid():
            raise RuntimeError(f"Invalid chess root: {chess_root_path}")

        self.board_center = world_bbox_center(stage, self.root_prim)

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
        box = cache.ComputeWorldBound(self.root_prim).GetBox()
        self.min = box.GetMin(); self.max = box.GetMax()

        dx = self.max[0] - self.min[0]
        dy = self.max[1] - self.min[1]
        self.size_x = dx; self.size_y = dy
        self.step_x = self.size_x / 8.0
        self.step_y = self.size_y / 8.0

        z = self.board_center[2]
        self.corners = {
            "lower_left":  Gf.Vec3d(self.min[0] + self.step_x*0.5, self.min[1] + self.step_y*0.5, z),
            "lower_right": Gf.Vec3d(self.max[0] - self.step_x*0.5, self.min[1] + self.step_y*0.5, z),
            "upper_left":  Gf.Vec3d(self.min[0] + self.step_x*0.5, self.max[1] - self.step_y*0.5, z),
            "upper_right": Gf.Vec3d(self.max[0] - self.step_x*0.5, self.max[1] - self.step_y*0.5, z),
        }
        if a1_corner not in self.corners:
            raise ValueError("a1_corner must be one of lower_left/lower_right/upper_left/upper_right")
        self.a1_world = self.corners[a1_corner]

        if a1_corner == "lower_left":
            self.dir_file = Gf.Vec3d(self.step_x, 0, 0); self.dir_rank = Gf.Vec3d(0, self.step_y, 0)
        elif a1_corner == "lower_right":
            self.dir_file = Gf.Vec3d(-self.step_x, 0, 0); self.dir_rank = Gf.Vec3d(0, self.step_y, 0)
        elif a1_corner == "upper_left":
            self.dir_file = Gf.Vec3d(self.step_x, 0, 0); self.dir_rank = Gf.Vec3d(0, -self.step_y, 0)
        else:
            self.dir_file = Gf.Vec3d(-self.step_x, 0, 0); self.dir_rank = Gf.Vec3d(0, -self.step_y, 0)

        self.square_poses = self._build_square_poses()

    def _build_square_poses(self):
        squares = {}
        files = "abcdefgh"
        z = self.board_center[2]
        up_quat = Gf.Quatd(1, 0, 0, 0)
        for f_idx, f in enumerate(files):
            for r in range(8):
                center = self.a1_world + self.dir_file * f_idx + self.dir_rank * r
                sq = f"{f}{r+1}"
                squares[sq] = {"pos": Gf.Vec3d(center[0], center[1], z), "quat": up_quat}
        return squares

    def square_pick_pose(self, square: str, hover: float = None):
        if square not in self.square_poses:
            raise KeyError(square)
        p = self.square_poses[square]["pos"]
        q = self.square_poses[square]["quat"]
        h = self.lift if hover is None else hover
        return Gf.Vec3d(p[0], p[1], p[2] + h), q

    def _closest_square(self, pos: Gf.Vec3d):
        files = "abcdefgh"
        rel = pos - self.a1_world
        i = round(rel[0] / self.dir_file[0]) if abs(self.dir_file[0]) > 1e-9 else round(rel[1] / self.dir_file[1])
        j = round(rel[1] / self.dir_rank[1]) if abs(self.dir_rank[1]) > 1e-9 else round(rel[0] / self.dir_rank[0])
        i = max(0, min(7, int(i))); j = max(0, min(7, int(j)))
        return f"{files[i]}{j+1}"

    def assign_pieces(self):
        stage = omni.usd.get_context().get_stage()
        pieces = []
        occupied = {}
        for prim in stage.Traverse():
            pth = str(prim.GetPath())
            if not pth.startswith(self.root_path + "/"):
                continue
            if prim.GetTypeName() != "Mesh":
                continue
            n = prim.GetName().lower()
            if any(k in n for k in FILE_NAMES_FOR_BOARD):
                continue

            pos, quat = _get_world_pose(stage, prim)
            col_info = get_approx_base_color(prim)
            name_hint = _name_hint_color(prim)
            color_label = name_hint or _label_from_rgb(col_info["rgb"], fallback="unknown")
            ptype = _piece_type_from_name(prim)
            square = self._closest_square(pos)

            info = {
                "path": str(prim.GetPath()),
                "type": ptype,
                "color": color_label,
                "rgb": col_info["rgb"],
                "color_source": col_info["source"],
                "square": square,
                "world_pos": pos,
                "world_quat": quat,
            }
            pieces.append(info)
            occupied[square] = {"color": color_label, "type": ptype, "path": info["path"], "rgb": col_info["rgb"]}
        return pieces, occupied
