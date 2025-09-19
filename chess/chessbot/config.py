# SPDX-License-Identifier: MIT
from pxr import Gf

# Assets
ASSET_DIR = r"C:/PhD_Project/Twin/Tutorials/Assest/chess/chess"
CHESS_USD_NAME_CANDIDATES = ["chessboard.usd", "board.usd", "chess - Copy.usd", "ChessBoard.usd"]

# Board orientation and picking
A1_CORNER = "lower_left"   # lower_left | lower_right | upper_left | upper_right
DEFAULT_PICK_LIFT = 0.06

# Camera placement & params
CAMERA_Z = 0.95
USE_EXACT_UI_POS = False
CAM_UI_POS = Gf.Vec3d(0.0, -0.97756, 1.02452)

FOCAL_LENGTH = 24.0
F_STOP = 2.8
FOCUS_DISTANCE = float(CAMERA_Z)
CAMERA_RESOLUTION = (1024, 1024)
CLIPPING_NEAR = 0.01
CLIPPING_FAR = 10000.0

# Misc
REMOVE_PICKPLACE_CUBE = True

# Paths
CHESS_ROOT = "/World/Chess"
CAM_PATH   = "/World/RGBD_Camera"
