import json, glob, math, sys
sys.path.insert(0, 'Basic Simulator')
from app.narrate import cpa_tcpa

files = sorted(glob.glob('Basic Simulator/Data/missions/_llm_runs/Imazu01__*screening_v2_uncapped.json'))
for f in files:
    d = json.loads(open(f, encoding='utf-8').read())
    traj = d['trajectory']
    by_t = {}
    for row in traj:
        by_t.setdefault(row['time'], {})[row['vehicle']] = row
    for cp in d.get('checkpoints', []):
        t = cp['time']
        if not (1400 <= t <= 2200):
            continue
        dec = cp.get('decision') or {}
        reasoning = (dec.get('reasoning') or '')
        if 'passed' not in reasoning.lower() and 'past' not in reasoning.lower():
            continue
        row = by_t.get(t, {})
        own, ts1 = row.get('own_ship'), row.get('ts1')
        if not own or not ts1:
            continue
        cpa, tcpa = cpa_tcpa(own['x'], own['y'], own['heading'], own['speed'],
                             ts1['x'], ts1['y'], ts1['heading'], ts1['speed'])
        print(f.split('\\')[-1], f"t={t}")
        print(f"  ground truth (recomputed): CPA={cpa:.1f}m TCPA={tcpa:.1f}s")
        print(f"  action={dec.get('action')} reasoning={reasoning}")
        print()
