# # SPDX-License-Identifier: MIT
# """
# Standalone Chessboard Detector + Realtime Tracker (Isaac Sim)

# Usage (Windows):
#   <ISAAC_SIM_INSTALL>\python.bat path\to\standalone_chess.py --ui
#   <ISAAC_SIM_INSTALL>\python.bat path\to\standalone_chess.py --headless --asset-dir C:\path\to\chess\usd

# Common ISAAC locations:
#   - Launcher install:   %LOCALAPPDATA%\ov\pkg\isaac_sim-<version>\python.bat
#   - Pip package:        <your_venv>\Scripts\python.exe  (works too)

# Args:
#   --ui / --headless     UI on/off (default: UI on)
#   --asset-dir           Folder containing your chessboard USD(s)
#   --print-updates       Print live board snapshots on changes (default: on)
#   --pos-eps             Jitter threshold (meters) for pose change (default: 1e-5)
#   --ang-eps-deg         Jitter threshold (degrees) for rotation change (default: 0.25)
# """

import os, sys, argparse, time

# --- Isaac Sim bootstrap FIRST ---
from isaacsim import SimulationApp

def parse_args():
    p = argparse.ArgumentParser()
    ui = p.add_mutually_exclusive_group()
    ui.add_argument("--ui", action="store_true", help="Run with UI (default)")
    ui.add_argument("--headless", action="store_true", help="Run headless")
    p.add_argument("--asset-dir", type=str, default=None, help="Folder with chessboard USD(s)")
    p.add_argument("--print-updates", action="store_true", default=True, help="Print live board snapshots on changes")
    p.add_argument("--no-print-updates", action="store_false", dest="print_updates")
    p.add_argument("--pos-eps", type=float, default=1e-5, help="Position epsilon (m) for change detection")
    p.add_argument("--ang-eps-deg", type=float, default=0.25, help="Angular epsilon (deg) for change detection")
    return p.parse_args()

args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})  # must be created before any Isaac/pxr import

# --- Now the rest of Isaac imports ---
from omni.isaac.core import World
import omni.usd
from pxr import Usd, UsdGeom, Gf

# Make local package 'chessbot' importable (assumes chessbot/ next to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ---- Import your modular helpers (from previous refactoring) ----
# chessbot/config.py should define sane defaults; we override asset dir via CLI if provided.
from chessbot import (
    ASSET_DIR as CFG_ASSET_DIR,
    CHESS_USD_NAME_CANDIDATES,
    CHESS_ROOT,
    A1_CORNER,
    DEFAULT_PICK_LIFT,
)
from chessbot.scene_utils import find_chess_usd, analyze_prims_under
from chessbot.camera_utils import RGBDCameraManager
from chessbot.board_mapping import BoardGrid
from chessbot import realtime as rt

# We still reuse the PickPlace task just to spawn Franka/table
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace


# ---------------------------
# Small utilities (printing)
# ---------------------------
def print_board_state(title, state):
    pieces = state.get("pieces", [])
    print(f"{title} {len(pieces)} pieces")
    # sort by rank then file for readability
    def _key(p):
        sq = p["square"]
        return (sq[1], sq[0]) if len(sq) == 2 else (sq, sq)
    for p in sorted(pieces, key=_key):
        pos = p["world_pos"]
        rgb = p["rgb"]
        rgb_s = "None" if rgb is None else f"({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})"
        print(f"  {p['color']:5} {p['type']:6} {p['square']:>2}  "
              f"→ ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  col={rgb_s}  path={p['path']}")


# ---------------------------
# Build scene
# ---------------------------
def build_scene(world: World, asset_dir: str, pos_eps: float, ang_eps_deg: float, print_updates: bool):
    stage: Usd.Stage = omni.usd.get_context().get_stage()

    # 1) Spawn Franka/Table via PickPlace
    world.add_task(PickPlace(name="awesome_task"))

    # NOTE: World requires a reset to materialize tasks before we add more content.
    world.reset()
    stage = omni.usd.get_context().get_stage()
    # Find and remove the cube prim
    for prim in stage.Traverse():
        if prim.GetName().lower() in ["cube", "target", "pick_target"]:
            stage.RemovePrim(prim.GetPath())
            break

    # 2) Load chessboard USD under /World/Chess
    chess_usd = find_chess_usd(asset_dir, CHESS_USD_NAME_CANDIDATES)
    if stage.GetPrimAtPath(CHESS_ROOT).IsValid():
        stage.RemovePrim(CHESS_ROOT)

    chess_xf = UsdGeom.Xform.Define(stage, CHESS_ROOT)
    chess_prim = stage.GetPrimAtPath(CHESS_ROOT)
    chess_prim.GetReferences().ClearReferences()
    chess_prim.GetReferences().AddReference(chess_usd)

    # Position board (adjust to your table)
    UsdGeom.XformCommonAPI(chess_xf).SetTranslate((0.68959, 0.0, 0.0123))
    stage.Load(CHESS_ROOT)

    # Optional debug dump
    _ = analyze_prims_under(stage, CHESS_ROOT)

    # 3) Create overhead RGB-D camera (Replicator annotators attached)
    camera = RGBDCameraManager(chess_root=CHESS_ROOT)
    camera.create_or_reset()

    # 4) Build board grid (a1..h8), detect/assign pieces
    board_map = BoardGrid(stage, CHESS_ROOT, a1_corner=A1_CORNER, lift_above_square=DEFAULT_PICK_LIFT)
    pieces_info, occupied = board_map.assign_pieces()

    # Print initial snapshot
    initial_state = {
        "pieces": pieces_info,
        "occupied": occupied,
        "squares": board_map.square_poses
    }
    print_board_state("[BOARD SNAPSHOT: initial]", initial_state)

    # 5) Realtime tracker: only emits when pose actually changes (thresholds configurable)
    rt.POS_EPS = float(pos_eps)
    rt.ANG_EPS_DEG = float(ang_eps_deg)

    def on_pose_change(prim_path: str, pose: dict):
        # Update our cached list entry and occupancy; mark dirty for per-frame print
        rec = next((p for p in pieces_info if p["path"] == prim_path), None)
        if not rec:
            return
        rec["world_pos"] = pose["pos"]
        rec["world_quat"] = pose["quat"]
        # recompute nearest square and adjust occupied
        new_sq = board_map._closest_square(rec["world_pos"])
        if new_sq != rec["square"]:
            old_sq = rec["square"]
            rec["square"] = new_sq
            occupied.pop(old_sq, None)
            occupied[new_sq] = {"color": rec["color"], "type": rec["type"], "path": rec["path"], "rgb": rec["rgb"]}
        # flag for printing once this frame
        on_pose_change._dirty = True

    on_pose_change._dirty = False  # attach attribute for per-frame coalescing

    tracker = rt.RealtimeTracker(
        paths_to_track=[p["path"] for p in pieces_info],
        on_update=on_pose_change,
        verbose=False
    )
    tracker.start(world.add_physics_callback, name="rt_tracker")

    # 6) Per-frame emitter: print a single snapshot if anything changed
    def on_after_step(dt: float):
        if print_updates and on_pose_change._dirty:
            state = {
                "pieces": pieces_info,
                "occupied": occupied,
                "squares": board_map.square_poses
            }
            print_board_state("[BOARD UPDATE]", state)
        on_pose_change._dirty = False

    world.add_physics_callback("board_emit", callback_fn=on_after_step)

    # Return handles in case a caller wants to query
    return {
        "camera": camera,
        "board_map": board_map,
        "pieces_info": pieces_info,
        "occupied": occupied,
        "tracker": tracker,
    }


def main():
    # Decide asset dir (CLI overrides config)
    asset_dir = args.asset_dir or CFG_ASSET_DIR
    if not asset_dir or not os.path.isdir(asset_dir):
        raise FileNotFoundError(f"Asset dir not found: {asset_dir!r}. Pass --asset-dir")

    world = World(stage_units_in_meters=1.0)
    handles = build_scene(
        world=world,
        asset_dir=asset_dir,
        pos_eps=args.pos_eps,
        ang_eps_deg=args.ang_eps_deg,
        print_updates=args.print_updates,
    )

    # Run the sim
    world.play()
    try:
        while simulation_app.is_running():
            world.step(render=not args.headless)
    finally:
        # Clean shutdown
        try:
            world.remove_physics_callback("board_emit")
        except Exception:
            pass
        if handles.get("tracker"):
            handles["tracker"].stop(world.remove_physics_callback, name="rt_tracker")
        simulation_app.close()


if __name__ == "__main__":
    main()
