import os

##################################################

MEDIAPIPE_POINTS = os.path.join(
    'data',
    'data-parsed',
)
UNDISTORTED = os.path.join(
    'data',
    'undistorted',
)

PATH = {
    'mediapipe_points': MEDIAPIPE_POINTS,
    'undistorted': UNDISTORTED,
}

##################################################

CONFIG = {
    'path': PATH,
}
