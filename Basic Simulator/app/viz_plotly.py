"""Live trajectory-replay figure, adapted from Brain Storming/test_plotly.py
to work directly off Simulation.trajectory (a live in-memory list of rows)
instead of a static CSV, and to show the mission goal."""
from __future__ import annotations
import math
from collections import defaultdict

import plotly.graph_objects as go

from app.missions import Mission
from app.simulation import _segment_min_range
from app.units import mps_to_kn, m_to_nm

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


def _actual_range_stats(trajectory: list[dict], target_name: str,
                        up_to_time: float) -> tuple[float, float] | tuple[None, None]:
    """ACTUAL (not predicted -- unlike cpa_tcpa()) range from own-ship to `target_name` right
    now, and the minimum such range recorded up to `up_to_time` -- interpolating the closest
    approach WITHIN each consecutive pair of samples (same _segment_min_range() find_collision()
    already uses), not just the sampled endpoints, so a fast pass-by between two recorded steps
    isn't missed. Returns (now_range_m, min_range_m), or (None, None) if `target_name` has no
    recorded rows yet at/before up_to_time."""
    own_rows = sorted((r for r in trajectory if r["vehicle"] == "own_ship" and r["time"] <= up_to_time),
                      key=lambda r: r["time"])
    tgt_rows = sorted((r for r in trajectory if r["vehicle"] == target_name and r["time"] <= up_to_time),
                      key=lambda r: r["time"])
    if not own_rows or not tgt_rows:
        return None, None
    by_own = {r["time"]: r for r in own_rows}
    by_tgt = {r["time"]: r for r in tgt_rows}
    common = sorted(set(by_own) & set(by_tgt))
    if not common:
        return None, None
    now_o, now_tg = by_own[common[-1]], by_tgt[common[-1]]
    now_range = math.hypot(now_o["x"] - now_tg["x"], now_o["y"] - now_tg["y"])
    min_range = now_range
    prev_t, prev_o, prev_tg = None, None, None
    for t in common:
        o, tg = by_own[t], by_tgt[t]
        if prev_t is not None:
            rng, _ = _segment_min_range(prev_t, (prev_o["x"], prev_o["y"]), (prev_tg["x"], prev_tg["y"]),
                                        t, (o["x"], o["y"]), (tg["x"], tg["y"]))
            min_range = min(min_range, rng)
        prev_t, prev_o, prev_tg = t, o, tg
    return now_range, min_range


def _distance_box_text(trajectory: list[dict], targets_order: list[str], up_to_time: float) -> str | None:
    """Multi-line 'now/min actual range' readout for every target, for the new bottom-right
    plot annotation -- distinct from the existing CPA/TCPA box, which is a PREDICTED future
    closest approach, not what has actually happened in the run so far."""
    lines = ["\U0001F4CF Actual range"]
    for name in targets_order:
        now_r, min_r = _actual_range_stats(trajectory, name, up_to_time)
        if now_r is None:
            continue
        lines.append(f"{name}: now {m_to_nm(now_r):.3f} NM \u2022 min {m_to_nm(min_r):.3f} NM")
    return "<br>".join(lines) if len(lines) > 1 else None


def _distance_annotation(text: str) -> dict:
    return dict(
        text=text, x=0.99, y=0.01, xref="paper", yref="paper",
        xanchor="right", yanchor="bottom", showarrow=False, align="left",
        font=dict(size=12, color="#0b3d63"),
        bgcolor="rgba(255,255,255,0.85)", bordercolor="#0b3d63", borderwidth=1, borderpad=6,
    )

def trajectory_bounds(trajectory: list[dict], mission: Mission,
                       pad_frac: float = 0.12,
                       target_margin_factor: float = 2.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fixed x/y axis range spanning own-ship's WHOLE trajectory (+ goal), so playback
    frames don't rescale/jump as points accumulate. Target positions are included too, but
    ONLY within `target_margin_factor` times own-ship's own span -- a target that just
    sails off in a straight line for a long, non-terminating run (own-ship successfully
    avoided it, then the run continues to max_steps) can travel far further than own-ship
    ever does, which would otherwise balloon the axes and squash the actually-relevant
    encounter into an illegible sliver. Confirmed: Imazu01/v3_rag_cot -- ts1 at 20 m/s over
    2000s travelled 40km while own-ship moved under 1km on the same run."""
    own_rows = [row for row in trajectory if row["vehicle"] == "own_ship"]
    xs = [row["x"] for row in own_rows] + [mission.goal[0]]
    ys = [row["y"] for row in own_rows] + [mission.goal[1]]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    own_span = max(x_hi - x_lo, y_hi - y_lo, 1.0)
    margin = own_span * target_margin_factor
    for row in trajectory:
        if row["vehicle"] == "own_ship":
            continue
        if x_lo - margin <= row["x"] <= x_hi + margin and y_lo - margin <= row["y"] <= y_hi + margin:
            xs.append(row["x"])
            ys.append(row["y"])
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
            hovertext=[f"{name}: heading {lhdg:.0f}\u00b0, speed {mps_to_kn(lspd):.2f} kt"],
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
    if targets_order and current_time is not None:
        dist_text = _distance_box_text(trajectory, targets_order, current_time)
        if dist_text:
            annotations.append(_distance_annotation(dist_text))
    if collision is not None:
        annotations.append(dict(
            text=f"<b>\U0001F4A5 COLLISION with {collision['vehicle']} at t={collision['time']:.0f}s "
                 f"(range {m_to_nm(collision['range_m']):.3f} NM)</b>",
            x=0.5, y=1.06, xref="paper", yref="paper", xanchor="center", yanchor="bottom",
            showarrow=False, font=dict(size=13, color="#ffffff"),
            bgcolor="#d32f2f", bordercolor="#7a0000", borderwidth=1, borderpad=6,
        ))

    fig.update_layout(
        title=title or "",
        xaxis=xaxis, yaxis=yaxis,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=10),
        height=580,
        annotations=annotations,
    )
    return fig


def animated_trajectory_figure(trajectory: list[dict], mission: Mission,
                               title: str | None = None,
                               x_range: tuple[float, float] | None = None,
                               y_range: tuple[float, float] | None = None,
                               frame_times: list[float] | None = None,
                               collision: dict | None = None,
                               frame_duration_ms: int = 400,
                               metrics_by_time: dict[float, str] | None = None,
                               decision_by_time: dict[float, str] | None = None) -> go.Figure:
    """Same visuals as trajectory_figure(), but as ONE figure with native Plotly go.Frame
    animation (Play/Pause button + slider) instead of many separate st.plotly_chart() calls.
    Streamlit requires a unique `key` per plotly_chart() call within a single script run, so
    looping placeholder.plotly_chart() to animate forces a full component remount every frame
    -- that remount is what caused the flicker/blank-screen-until-the-end bug. Native frames
    animate entirely client-side (zero Streamlit round-trips), so there's nothing to remount.
    `metrics_by_time`/`decision_by_time` (keyed by exact frame time) render as extra in-plot
    annotation boxes per frame -- since the native slider/Play button never talks back to
    Streamlit, this is the only way to keep the goal-bearing/CPA/decision text in sync with
    whatever frame the user is currently looking at (a server-side panel has no way to know
    which frame the client-side animation is on)."""
    data: dict[str, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    for row in trajectory:
        data[row["vehicle"]].append((row["time"], row["x"], row["y"], row["heading"], row["speed"]))
    for v in data:
        data[v].sort(key=lambda p: p[0])
    vehicles = list(data.keys())
    targets_order = [t.name for t in mission.targets]
    all_times = sorted({p[0] for series in data.values() for p in series})
    frame_times = frame_times or all_times
    total_time = all_times[-1] if all_times else 0.0

    def _pos_at(series, t):
        pts = [p for p in series if p[0] <= t]
        return pts[-1] if pts else series[0]

    fig = go.Figure()

    for name in vehicles:
        xs, ys = [p[1] for p in data[name]], [p[2] for p in data[name]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=f"{name} (path)",
            line=dict(width=1.5, dash="dot", color=_color_for(name, targets_order)),
            opacity=0.45, showlegend=False,
        ))

    gx, gy = mission.goal
    fig.add_trace(go.Scatter(
        x=[gx], y=[gy], mode="markers+text", text=["goal"], textposition="top center",
        marker=dict(size=16, symbol="star", color="#ffd700", line=dict(width=1, color="#7a6200")),
        name="goal",
    ))
    n_static = len(vehicles) + 1

    t0 = frame_times[0] if frame_times else (all_times[0] if all_times else 0.0)
    for name in vehicles:
        _, lx, ly, lhdg, lspd = _pos_at(data[name], t0)
        fig.add_trace(go.Scatter(
            x=[lx], y=[ly], mode="markers+text", text=[name], textposition="top center",
            marker=dict(size=_arrow_size(lspd), symbol="arrow", angle=lhdg,
                       color=_color_for(name, targets_order),
                       line=dict(width=1, color="rgba(0,0,0,0.35)")),
            name=name,
            hovertext=[f"{name}: heading {lhdg:.0f}\u00b0, speed {mps_to_kn(lspd):.2f} kt"],
            hoverinfo="text",
        ))
    marker_idx = list(range(n_static, n_static + len(vehicles)))

    def _time_annotation(t):
        return dict(
            text=f"<b>t = {t:.0f}s / {total_time:.0f}s</b>", x=0.01, y=0.99,
            xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
            font=dict(size=15, color="#0b3d63"),
            bgcolor="rgba(255,255,255,0.75)", bordercolor="#0b3d63", borderwidth=1, borderpad=4,
        )

    def _collision_annotation():
        return dict(
            text=f"<b>\U0001F4A5 COLLISION with {collision['vehicle']} at t={collision['time']:.0f}s "
                 f"(range {m_to_nm(collision['range_m']):.3f} NM)</b>",
            x=0.5, y=1.06, xref="paper", yref="paper", xanchor="center", yanchor="bottom",
            showarrow=False, font=dict(size=13, color="#ffffff"),
            bgcolor="#d32f2f", bordercolor="#7a0000", borderwidth=1, borderpad=6,
        )

    def _metrics_annotation(text):
        return dict(
            text=text, x=0.99, y=0.99, xref="paper", yref="paper",
            xanchor="right", yanchor="top", showarrow=False, align="left",
            font=dict(size=12, color="#0b3d63"),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#0b3d63", borderwidth=1, borderpad=6,
        )

    def _decision_annotation(text):
        return dict(
            text=text, x=0.01, y=0.90, xref="paper", yref="paper",
            xanchor="left", yanchor="top", showarrow=False, align="left",
            font=dict(size=12, color="#5a3d00"),
            bgcolor="rgba(255,247,224,0.9)", bordercolor="#a67c00", borderwidth=1, borderpad=6,
        )

    def _extra_annotations(t):
        anns = []
        if metrics_by_time and t in metrics_by_time:
            anns.append(_metrics_annotation(metrics_by_time[t]))
        if decision_by_time and t in decision_by_time:
            anns.append(_decision_annotation(decision_by_time[t]))
        if targets_order:
            dist_text = _distance_box_text(trajectory, targets_order, t)
            if dist_text:
                anns.append(_distance_annotation(dist_text))
        return anns

    frames = []
    for t in frame_times:
        frame_data = []
        for name in vehicles:
            _, lx, ly, lhdg, lspd = _pos_at(data[name], t)
            frame_data.append(go.Scatter(
                x=[lx], y=[ly], marker=dict(size=_arrow_size(lspd), angle=lhdg),
                hovertext=[f"{name}: heading {lhdg:.0f}\u00b0, speed {mps_to_kn(lspd):.2f} kt"],
            ))
        anns = [_time_annotation(t)] + _extra_annotations(t)
        if collision is not None and t >= collision["time"]:
            anns.append(_collision_annotation())
        frames.append(go.Frame(name=f"{t:.0f}", data=frame_data, traces=marker_idx,
                               layout=go.Layout(annotations=anns)))
    fig.frames = frames

    xaxis: dict = dict(title="x (m)")
    yaxis: dict = dict(title="y (m)", scaleanchor="x", scaleratio=1)
    if x_range is not None:
        xaxis["range"] = list(x_range)
        xaxis["autorange"] = False
    if y_range is not None:
        yaxis["range"] = list(y_range)
        yaxis["autorange"] = False

    init_annotations = [_time_annotation(t0)] + _extra_annotations(t0)
    if collision is not None and t0 >= collision["time"]:
        init_annotations.append(_collision_annotation())

    slider_steps = [
        dict(method="animate", label=f"{t:.0f}s",
            args=[[f"{t:.0f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                     transition=dict(duration=0))])
        for t in frame_times
    ]

    fig.update_layout(
        title=title or "",
        xaxis=xaxis, yaxis=yaxis,
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=60, b=60),
        height=620,
        annotations=init_annotations,
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=-0.22, xanchor="left", yanchor="top",
            buttons=[
                dict(label="\u25b6 Play", method="animate",
                    args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True),
                                     fromcurrent=True, transition=dict(duration=0))]),
                dict(label="\u23f8 Pause", method="animate",
                    args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False),
                                       transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.12, y=-0.22, len=0.85, xanchor="left", yanchor="top",
            currentvalue=dict(prefix="t=", suffix="s", visible=True),
            steps=slider_steps,
        )],
    )
    return fig

