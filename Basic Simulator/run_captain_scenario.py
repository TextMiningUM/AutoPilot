"""Standalone Captain walking-skeleton visualizer -- runs one (or all) scenario file(s)
under Data/Captain/Scenarios/ to completion via CaptainSkeleton's own debug control set
(design_captain_missions.md Sec 15.3), prints the Mission Progress Report + evaluation,
and plots speed/fuel over mission time with brown-envelope decision markers.

No Streamlit involved -- this is the quick, non-UI way to actually SEE the walking
skeleton run (Sec 15.4's full UI spec is still deferred). Local-safe: pure deterministic
simulation, no GPU/API key needed.

Run with:
    python run_captain_scenario.py                                # all 4 scenarios
    python run_captain_scenario.py --scenario skeleton_fog_whale_zone_v1
    python run_captain_scenario.py --no-plot                      # reports only, no matplotlib
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.captain_skeleton import CaptainSkeleton  # noqa: E402

SCENARIOS_DIR = ROOT.parent / "Data" / "Captain" / "Scenarios"
# Generous fixed cap rather than a budget derived from the ORIGINAL speed-of-advance --
# a brown envelope (e.g. engine_failure) can drop the effective speed well below that for
# the rest of the mission, under-budgeting a pre-computed step count (the same bug once
# found in test_step_mission_advances_the_clock_and_stops_at_the_destination, see repo
# memory) -- cheap pure-Python stepping, so a large cap costs nothing in practice.
MAX_STEPS = 20_000


def run_scenario(path: Path) -> tuple[CaptainSkeleton, list[dict]]:
    """Steps a fresh skeleton to completion, sampling speed/fuel/distance every step."""
    skeleton = CaptainSkeleton.from_scenario_file(path)
    history: list[dict] = []
    for _ in range(MAX_STEPS):
        if skeleton.sim.reached_destination():
            break
        skeleton.step_mission(1)
        history.append({
            "t_h": skeleton.sim.state.elapsed_s / 3600.0,
            "speed_kn": skeleton.sim.state.current_speed_kn,
            "fuel_t": skeleton.sim.state.fuel_tonnes,
        })
    return skeleton, history


def decision_markers(skeleton: CaptainSkeleton) -> list[tuple[float, str]]:
    """(t_h, label) for every logged Captain decision -- the moments worth annotating."""
    return [(d["t"] / 3600.0, f"{d['cause']}: {d['new']['tool']}")
            for d in skeleton.state.event_log if d["field_path"] == "captain_decision"]


def plot_run(name: str, skeleton: CaptainSkeleton, history: list[dict], out_path: Path) -> None:
    """Saves a 2-panel speed/fuel-over-time PNG with red dashed lines at each decision."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = [h["t_h"] for h in history]
    ax1.plot(t, [h["speed_kn"] for h in history], label="speed (kn)")
    ax1.axhline(skeleton.order.speed_of_advance_kn, color="gray", linestyle="--", label="original SOA")
    ax1.set_ylabel("Speed (kn)")
    ax1.legend(loc="upper right")
    ax2.plot(t, [h["fuel_t"] for h in history], color="tab:orange")
    ax2.set_ylabel("Fuel remaining (t)")
    ax2.set_xlabel("Mission time (h)")
    for t_h, label in decision_markers(skeleton):
        for ax in (ax1, ax2):
            ax.axvline(t_h, color="red", linestyle=":", alpha=0.6)
        ax1.annotate(label, (t_h, skeleton.order.speed_of_advance_kn), rotation=90,
                    fontsize=7, color="red", va="bottom")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run and visualize a Captain walking-skeleton scenario.")
    ap.add_argument("--scenario", type=str, default=None,
                    help="scenario filename stem under Data/Captain/Scenarios/ (default: run all)")
    ap.add_argument("--no-plot", action="store_true", help="skip matplotlib, print reports only")
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "_captain_runs"),
                    help="directory to save plots to (default: Basic Simulator/_captain_runs/)")
    args = ap.parse_args()

    paths = ([SCENARIOS_DIR / f"{args.scenario}.json"] if args.scenario
            else sorted(SCENARIOS_DIR.glob("*.json")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        print(f"\n=== {path.stem} ===")
        skeleton, history = run_scenario(path)
        print(skeleton.show_mpr())
        result = skeleton.evaluate()
        print(f"\nEvaluation: verdict={result.verdict} composite={result.composite_score:.3f} "
              f"safety_passed={result.safety_passed} mission_outcome={result.mission_outcome_score:.2f} "
              f"resource_efficiency={result.resource_efficiency['score']:.2f}")
        if not args.no_plot:
            plot_run(path.stem, skeleton, history, out_dir / f"{path.stem}.png")


if __name__ == "__main__":
    main()
