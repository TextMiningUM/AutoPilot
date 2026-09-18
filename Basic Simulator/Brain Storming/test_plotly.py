#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build + sanity-check a Plotly trajectory-replay figure using scenario_D
(the collision case) from the evaluate_run.py test set -- reusing real data
rather than inventing new numbers."""
import csv
import plotly.graph_objects as go
from collections import defaultdict

def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            data[row["vehicle"]].append((float(row["time"]), float(row["x"]), float(row["y"])))
    for v in data:
        data[v].sort()
    return data

def trajectory_figure(csv_path, title="Trajectory replay"):
    data = load_csv(csv_path)
    all_times = sorted({t for series in data.values() for t, x, y in series})

    fig = go.Figure()
    # full path (faint) per vehicle, drawn once
    for name, series in data.items():
        xs = [p[1] for p in series]
        ys = [p[2] for p in series]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{name} (path)",
                                  line=dict(width=1, dash="dot"), opacity=0.4))

    # animation frames: one marker per vehicle per time step
    frames = []
    for t in all_times:
        frame_data = []
        for name, series in data.items():
            pt = min(series, key=lambda p: abs(p[0] - t))
            frame_data.append(go.Scatter(x=[pt[1]], y=[pt[2]], mode="markers+text",
                                          text=[name], textposition="top center",
                                          marker=dict(size=14), name=name))
        frames.append(go.Frame(data=frame_data, name=str(t)))

    for name, series in data.items():
        fig.add_trace(go.Scatter(x=[series[0][1]], y=[series[0][2]], mode="markers+text",
                                  text=[name], textposition="top center",
                                  marker=dict(size=14), name=name))

    fig.frames = frames
    fig.update_layout(
        title=title,
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        updatemenus=[dict(type="buttons", buttons=[
            dict(label="Play", method="animate",
                 args=[None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True}]),
            dict(label="Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
        ])],
        sliders=[dict(steps=[dict(method="animate", args=[[str(t)],
                 {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                 label=f"{t:.0f}s") for t in all_times])],
    )
    return fig

if __name__ == "__main__":
    fig = trajectory_figure("/home/claude/eval_function/scenario_D_collision.csv",
                             "Scenario D: collision case (own ship vs TS1)")
    fig.write_html("/home/claude/viz_demo/plotly_test.html")
    print("Frames built:", len(fig.frames))
    print("Traces built:", len(fig.data))
    print("Wrote plotly_test.html -- open it to confirm it actually renders/animates.")
