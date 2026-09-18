"""Live trajectory-replay figure, adapted from Brain Storming/test_plotly.py
to work directly off Simulation.trajectory (a live in-memory list of rows)
instead of a static CSV, and to show the mission goal."""
from __future__ import annotations
from collections import defaultdict

import plotly.graph_objects as go

from app.missions import Mission

_COLORS = {"own_ship": "#1f77b4"}
_PALETTE = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


def _color_for(vehicle: str, targets_order: list[str]) -> str:
    if vehicle in _COLORS:
        return _COLORS[vehicle]
    idx = targets_order.index(vehicle) if vehicle in targets_order else 0
    return _PALETTE[idx % len(_PALETTE)]


def trajectory_bounds(trajectory: list[dict], mission: Mission,
                       pad_frac: float = 0.12) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fixed x/y axis range spanning the WHOLE trajectory (+ goal), so playback frames
    don't rescale/jump as points accumulate."""
    xs = [row["x"] for row in trajectory] + [mission.goal[0]]
    ys = [row["y"] for row in trajectory] + [mission.goal[1]]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_pad = (x_hi - x_lo) * pad_frac or 50.0
    y_pad = (y_hi - y_lo) * pad_frac or 50.0
    return (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)


def trajectory_figure(trajectory: list[dict], mission: Mission,
                       title: str | None = None,
                       x_range: tuple[float, float] | None = None,
                       y_range: tuple[float, float] | None = None,
                       current_time: float | None = None,
                       total_time: float | None = None) -> go.Figure:
    data: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for row in trajectory:
        data[row["vehicle"]].append((row["time"], row["x"], row["y"]))
    for v in data:
        data[v].sort(key=lambda p: p[0])

    targets_order = [t.name for t in mission.targets]
    all_times = sorted({t for series in data.values() for t, _, _ in series})

    fig = go.Figure()

    # faint full path per vehicle
    for name, series in data.items():
        xs, ys = [p[1] for p in series], [p[2] for p in series]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{name} (path)",
            line=dict(width=1.5, dash="dot", color=_color_for(name, targets_order)),
            opacity=0.45, showlegend=False,
        ))

    # goal marker
    gx, gy = mission.goal
    fig.add_trace(go.Scatter(
        x=[gx], y=[gy], mode="markers+text", text=["goal"], textposition="top center",
        marker=dict(size=16, symbol="star", color="#ffd700",
                    line=dict(width=1, color="#7a6200")),
        name="goal",
    ))

    # current-position markers
    for name, series in data.items():
        last = series[-1]
        fig.add_trace(go.Scatter(
            x=[last[1]], y=[last[2]], mode="markers+text", text=[name],
            textposition="top center", marker=dict(size=15, color=_color_for(name, targets_order)),
            name=name,
        ))

    if current_time is None and all_times:
        current_time = all_times[-1]
    if total_time is None:
        total_time = current_time
    time_label = None
    if current_time is not None:
        time_label = f"t = {current_time:.0f}s / {total_time:.0f}s"

    xaxis: dict = dict(title="x (m)")
    yaxis: dict = dict(title="y (m)", scaleanchor="x", scaleratio=1)
    if x_range is not None:
        xaxis["range"] = list(x_range)
        xaxis["autorange"] = False
    if y_range is not None:
        yaxis["range"] = list(y_range)
        yaxis["autorange"] = False

    fig.update_layout(
        title=title or f"{mission.name} ({mission.id})",
        xaxis=xaxis, yaxis=yaxis,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
        height=580,
        annotations=[dict(
            text=f"<b>{time_label}</b>", x=0.01, y=0.99, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False, font=dict(size=15, color="#0b3d63"),
            bgcolor="rgba(255,255,255,0.75)", bordercolor="#0b3d63", borderwidth=1, borderpad=4,
        )] if time_label else [],
    )
    return fig

