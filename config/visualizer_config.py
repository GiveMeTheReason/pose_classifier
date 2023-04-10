from config.base_config import ConfigBaseClass

##################################################

scene = {
    'xaxis': {'range': [-1, 1], 'autorange': False},
    'yaxis': {'range': [-1, 1], 'autorange': False},
    'zaxis': {'range': [0, 2], 'autorange': False},
    'aspectratio': {'x': 1, 'y': 1, 'z': 1},
}
scene_camera = {
    'up': {'x': 0, 'y': -1, 'z': 0},
    'center': {'x': -0.8, 'y': 0.7, 'z': 1},
    'eye': {'x': 0.8, 'y': -0.3, 'z': -0.5},
}

##################################################

play_button = {
    'label': 'Play',
    'method': 'animate',
    'args': [
        None,
        {
            'frame': {
                'duration': 1000/30,
                'redraw': True,
                'mode': 'immediate',
            },
            'fromcurrent': True,
            'transition': {
                'duration': 1000/30,
                'easing': 'quadratic-in-out',
            },
        }
    ],
}
pause_button = {
    'label': 'Pause',
    'method': 'animate',
    'args': [
        [None],
        {
            'frame': {
                'duration': 0,
                'redraw': False,
            },
            'mode': 'immediate',
            'transition': {
                'duration': 0,
            },
        },
    ],
}
update_buttons = {
    'type': 'buttons',
    'buttons': [
        play_button,
        pause_button,
    ],
}

##################################################

class VISUALIZER_CONFIG(ConfigBaseClass):
    scene: dict = scene
    scene_camera: dict = scene_camera
    update_buttons: dict = update_buttons
