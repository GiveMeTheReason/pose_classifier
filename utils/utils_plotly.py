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


def create_figure(
    title: str = '',
    groups: tp.Optional[tp.List[str]] = None,
    **kwargs,
) -> go.Figure:
    fig = go.Figure()

    if groups is None:
        groups = []
    for group in groups:
        fig.add_trace(go.Scatter(
            y=[None],
            mode='none',
            name=group,
            legendgroup=group
        ))

    set_default_layout(fig, title, **kwargs)
    return fig


def create_figure_3d(
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


def set_default_layout(
    fig: go.Figure,
    title: str = '',
    **kwargs,
) -> None:
    params = dict(
        title=title,
        xaxis_title='Frame',
        hoverlabel=dict(
            namelength=-1,
        ),
        margin=dict(
            l=0,
            r=0,
            t=50,
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )
    params = update_nested_dict(params, kwargs)
    fig.update_layout(**params)


def plot_line(
    data: tp.Any,
    fig: tp.Optional[go.Figure] = None,
    **kwargs,
):
    if fig is None:
        fig = go.Figure()
        set_default_layout(fig)
    params = dict(
        y=data,
        mode='lines',
    )
    params = update_nested_dict(params, kwargs)
    fig.add_trace(go.Scatter(**params))
    return fig


def visualize_data(
    data: tp.List[tp.Any],
    with_axis: bool = True,
    **kwargs,
):
    params = dict(
        data=data,
        layout=dict(
            scene=dict(
                xaxis=dict(visible=with_axis),
                yaxis=dict(visible=with_axis),
                zaxis=dict(visible=with_axis),
            ),
        ),
    )
    params = update_nested_dict(params, kwargs)
    fig = go.Figure(**params)
    fig.update_scenes(aspectmode='data')
    fig.show()
