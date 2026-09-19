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


def _arrow_size(speed: float, base: float = 14.0, scale: float = 4.0,
               min_size: float = 12.0, max_size: float = 34.0) -> float:
    """Marker size for a vessel's direction arrow -- grows with speed so a glance at the
    plot shows relative speed, not just heading."""
    return max(min_size, min(max_size, base + speed * scale))


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
                       total_time: float | None = None,
                       collision: dict | None = None) -> go.Figure:
    data: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    for row in trajectory:
        data[row["vehicle"]].append((row["time"], row["x"], row["y"], row["heading"], row["speed"]))
    for v in data:
        data[v].sort(key=lambda p: p[0])

    targets_order = [t.name for t in mission.targets]
    all_times = sorted({p[0] for series in data.values() for p in series})

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

    # current-position markers -- arrow points along heading (0=north/+y, clockwise, matching
    # the sim's compass-bearing convention), sized by speed so relative speed reads at a glance
    for name, series in data.items():
        last = series[-1]
        _, lx, ly, lhdg, lspd = last
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers+text", text=[name],
            textposition="top center",
            marker=dict(size=_arrow_size(lspd), symbol="arrow", angle=lhdg,
                       color=_color_for(name, targets_order),
                       line=dict(width=1, color="rgba(0,0,0,0.35)")),
            name=name,
            hovertext=[f"{name}: heading {lhdg:.0f}\u00b0, speed {lspd:.2f} m/s"],
            hoverinfo="text",
        ))

    if collision is not None:
        fig.add_trace(go.Scatter(
            x=[collision["x"]], y=[collision["y"]], mode="markers+text",
            text=["\U0001F4A5 COLLISION"], textposition="bottom center",
            textfont=dict(color="#d32f2f", size=14),
            marker=dict(size=24, symbol="x", color="#ff1744", line=dict(width=3, color="#7a0000")),
            name="collision",
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

    annotations = []
    if time_label:
        annotations.append(dict(
            text=f"<b>{time_label}</b>", x=0.01, y=0.99, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False, font=dict(size=15, color="#0b3d63"),
            bgcolor="rgba(255,255,255,0.75)", bordercolor="#0b3d63", borderwidth=1, borderpad=4,
        ))
    if collision is not None:
        annotations.append(dict(
            text=f"<b>\U0001F4A5 COLLISION with {collision['vehicle']} at t={collision['time']:.0f}s "
                 f"(range {collision['range_m']:.0f}m)</b>",
            x=0.5, y=1.06, xref="paper", yref="paper", xanchor="center", yanchor="bottom",
            showarrow=False, font=dict(size=13, color="#ffffff"),
            bgcolor="#d32f2f", bordercolor="#7a0000", borderwidth=1, borderpad=6,
        ))

    fig.update_layout(
        title=title or f"{mission.name} ({mission.id})",
        xaxis=xaxis, yaxis=yaxis,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
        height=580,
        annotations=annotations,
    )
    return fig

