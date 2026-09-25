import json, glob, math, sys
sys.path.insert(0, 'Basic Simulator')
from app.narrate import cpa_tcpa

def closing_flag(ox, oy, ohdg, ospd, tx, ty, thdg, tspd):
    oh, th = math.radians(ohdg), math.radians(thdg)
    vox, voy = ospd * math.sin(oh), ospd * math.cos(oh)
    vtx, vty = tspd * math.sin(th), tspd * math.cos(th)
    dx, dy = tx - ox, ty - oy
    dvx, dvy = vtx - vox, vty - voy
    rel_sq = dvx ** 2 + dvy ** 2
    return rel_sq >= 1e-6 and -(dx * dvx + dy * dvy) > 0

files = sorted(glob.glob('Basic Simulator/Data/missions/_llm_runs/Imazu*screening_v2_uncapped.json'))
hits = 0
for f in files:
    d = json.loads(open(f, encoding='utf-8').read())
    traj = d['trajectory']
    by_t = {}
    for row in traj:
        by_t.setdefault(row['time'], {})[row['vehicle']] = row
    for cp in d.get('checkpoints', []):
        t = cp['time']
        row = by_t.get(t, {})
        own = row.get('own_ship')
        for name, tgt in row.items():
            if name == 'own_ship' or own is None:
                continue
            cpa, tcpa = cpa_tcpa(own['x'], own['y'], own['heading'], own['speed'],
                                 tgt['x'], tgt['y'], tgt['heading'], tgt['speed'])
            closing = closing_flag(own['x'], own['y'], own['heading'], own['speed'],
                                   tgt['x'], tgt['y'], tgt['heading'], tgt['speed'])
            # The dangerous case: TCPA rounds to ~0 while STILL closing (report would NOT
            # show "already past"), i.e. right at/just before the actual closest approach.
            if closing and tcpa < 5.0 and cpa < 300.0:
                dec = cp.get('decision') or {}
                hits += 1
                print(f.split('\\')[-1], f"t={t} contact={name}")
                print(f"  ground truth: CPA={cpa:.1f}m TCPA={tcpa:.1f}s closing={closing}")
                print(f"  action={dec.get('action')} rule={dec.get('encounter_rule')}")
                print(f"  reasoning={dec.get('reasoning')}")
                print()
print(f"\n{hits} checkpoint(s) found with a genuinely live (closing) near-zero-TCPA encounter")
