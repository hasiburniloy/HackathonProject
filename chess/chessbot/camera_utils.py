# SPDX-License-Identifier: MIT
import numpy as np
from pxr import Usd, UsdGeom, Gf
import omni.usd
from omni.isaac.core.utils import prims as core_prims
import omni.replicator.core as rep

from .config import (
    CAM_PATH, CAMERA_RESOLUTION,
    FOCUS_DISTANCE, FOCAL_LENGTH, CLIPPING_NEAR, CLIPPING_FAR,
    CAMERA_Z, USE_EXACT_UI_POS, CAM_UI_POS
)
from .scene_utils import world_bbox_center, set_camera_pose_matrix

class RGBDCameraManager:
    def __init__(self, chess_root: str):
        self.chess_root = chess_root
        self._get_camera_data_fn = None

    def create_or_reset(self):
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(CAM_PATH).IsValid():
            stage.RemovePrim(CAM_PATH)

        core_prims.create_prim(
            prim_path=CAM_PATH,
            prim_type="Camera",
            attributes={
                "focusDistance": float(FOCUS_DISTANCE),
                "focalLength": float(FOCAL_LENGTH),
                "horizontalAperture": 20.955,
                "verticalAperture": 15.2908,
                "clippingRange": (float(CLIPPING_NEAR), float(CLIPPING_FAR)),
            },
        )

        board_ctr = world_bbox_center(stage, stage.GetPrimAtPath(self.chess_root))
        eye = CAM_UI_POS if USE_EXACT_UI_POS else Gf.Vec3d(board_ctr[0], board_ctr[1], CAMERA_Z)
        set_camera_pose_matrix(CAM_PATH, eye=eye, target=board_ctr)

        # Replicator
        rp = rep.create.render_product(CAM_PATH, resolution=CAMERA_RESOLUTION)
        rgb   = rep.AnnotatorRegistry.get_annotator("rgb")
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
        norms = rep.AnnotatorRegistry.get_annotator("normals")
        rgb.attach([rp]); depth.attach([rp]); norms.attach([rp])

        def _grab():
            rep.orchestrator.step()
            return rgb.get_data(), depth.get_data(), norms.get_data()

        self._get_camera_data_fn = _grab

    def get_camera_data(self):
        if not self._get_camera_data_fn:
            return None, None, None
        try:
            return self._get_camera_data_fn()
        except Exception as e:
            print("[ERROR] camera grab:", e)
            return None, None, None

    def save_images(self, base_filename="chess_camera"):
        rgb, depth, normals = self.get_camera_data()
        try:
            import cv2
            if rgb is not None:
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 1); rgb = (rgb * 255).astype(np.uint8)
                cv2.imwrite(f"{base_filename}_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if depth is not None:
                depth = np.nan_to_num(depth.astype(np.float32), nan=0, posinf=0, neginf=0)
                dmin, dmax = float(depth.min()), float(depth.max())
                depth_norm = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8) if dmax > dmin else np.zeros_like(depth, np.uint8)
                cv2.imwrite(f"{base_filename}_depth.png", depth_norm)
            if normals is not None:
                normals = np.nan_to_num(normals.astype(np.float32), nan=0, posinf=0, neginf=0)
                normals_vis = np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)
                cv2.imwrite(f"{base_filename}_normals.png", normals_vis)
        except Exception as e:
            print("[ERROR] save images:", e)

    def get_camera_info(self):
        stage = omni.usd.get_context().get_stage()
        cam = stage.GetPrimAtPath(CAM_PATH)
        if not cam or not cam.IsValid():
            return {}
        def _get(attr, default=None):
            a = cam.GetAttribute(attr)
            v = a.Get() if a and a.HasAuthoredValueOpinion() else None
            return v if v is not None else default
        return {
            "height_above_board": CAMERA_Z,
            "focal_length": _get("focalLength", FOCAL_LENGTH),
            "f_stop": _get("fStop", None),  # not always authored
            "focus_distance": _get("focusDistance", FOCUS_DISTANCE),
            "resolution": CAMERA_RESOLUTION,
            "clipping_range": tuple(_get("clippingRange", (CLIPPING_NEAR, CLIPPING_FAR))),
        }
