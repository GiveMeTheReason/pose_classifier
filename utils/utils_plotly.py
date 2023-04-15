import typing as tp

import numpy as np
import plotly.graph_objs as go


def update_nested_dict(old_dict: tp.Dict, new_dict: tp.Dict) -> tp.Dict:
    for key, value in new_dict.items():
        if isinstance(value, dict):
            old_dict[key] = update_nested_dict(old_dict.get(key, {}), value)
        else:
            old_dict[key] = value
    return old_dict


def get_empty_figure_3d(
    traces: int = 1,
    with_axis: bool = True,
    **kwargs,
) -> go.Figure:
    empty_scatter = get_scatter_3d(np.zeros((1, 3)))
    params = dict(
        data=[empty_scatter] * traces,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=with_axis),
                yaxis=dict(visible=with_axis),
                zaxis=dict(visible=with_axis),
            ),
        ),
    )
    params = update_nested_dict(params, kwargs)
    return go.Figure(**params)


def get_frame(
    data: tp.List[tp.Any],
    frame_num: int,
    **kwargs,
) -> go.Frame:
    params = dict(
        data=data,
        name=f'frame{frame_num}',
    )
    params = update_nested_dict(params, kwargs)
    return go.Frame(**params)


def get_scatter_3d(
    points: np.ndarray,
    colors: tp.Optional[np.ndarray] = None,
    size: int = 1,
    step: int = 1,
    **kwargs,
) -> go.Scatter3d:
    marker: tp.Dict[str, tp.Any] = dict(size=size)
    if colors is not None:
        marker['color'] = colors[::step]

    params = dict(
        x=points[::step, 0],
        y=points[::step, 1],
        z=points[::step, 2],
        mode='markers',
        marker=marker,
    )
    params = update_nested_dict(params, kwargs)
    return go.Scatter3d(**params)


def visualize_data(
    data: tp.List[tp.Any],
    with_axis: bool = True,
):
    fig = go.Figure(
        data=data,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=with_axis),
                yaxis=dict(visible=with_axis),
                zaxis=dict(visible=with_axis),
            ),
        ),
    )
    fig.update_scenes(aspectmode='data')
    fig.show()
