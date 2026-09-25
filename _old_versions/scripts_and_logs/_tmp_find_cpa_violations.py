import json, glob

files = [f for f in glob.glob('Basic Simulator/Data/missions/_llm_runs/*.json')
         if not f.endswith(('_sweep_status.json', '_sweep_summary.json'))]

rows = []
for f in files:
    try:
        d = json.loads(open(f, encoding='utf-8').read())
    except Exception:
        continue
    ev = d.get('evaluation') or {}
    safety = ev.get('safety') or {}
    min_cpa = safety.get('min_cpa_m')
    if min_cpa is None:
        continue
    rows.append({
        'file': f.split('\\')[-1], 'mission': d.get('mission_id'), 'config': d.get('config'),
        'tag': d.get('tag'), 'min_cpa_m': min_cpa, 'passed': safety.get('passed'),
        'verdict': ev.get('verdict'),
    })

rows.sort(key=lambda r: r['min_cpa_m'])
print(f"{len(rows)} run files with a computed min CPA (safe distance = 500m for all)\n")

collisions = [r for r in rows if not r['passed']]
severe = [r for r in rows if r['passed'] and r['min_cpa_m'] < 50]
moderate = [r for r in rows if r['passed'] and 50 <= r['min_cpa_m'] < 500]

print(f"=== ACTUAL COLLISIONS (min_cpa < 15m hull-to-hull) -- {len(collisions)} ===")
for r in collisions:
    print(f"  {r['min_cpa_m']:7.1f}m  {r['mission']:16s} {r['config']:26s} {r['tag']:22s} {r['file']}")

print(f"\n=== CPA COMPLETELY IGNORED (<50m, i.e. <10% of the 500m safe distance) -- {len(severe)} ===")
for r in severe:
    print(f"  {r['min_cpa_m']:7.1f}m  {r['mission']:16s} {r['config']:26s} {r['tag']:22s} {r['file']}")

print(f"\n=== moderate violation (50-500m) -- {len(moderate)} (showing worst 15) ===")
for r in moderate[:15]:
    print(f"  {r['min_cpa_m']:7.1f}m  {r['mission']:16s} {r['config']:26s} {r['tag']:22s} {r['file']}")

n_ok = sum(1 for r in rows if r['passed'] and r['min_cpa_m'] >= 500)
print(f"\nfully respected safe distance (>=500m): {n_ok}/{len(rows)}")
