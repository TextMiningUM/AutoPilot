import json, glob, re

pat = re.compile(r'^Imazu\d\d__(v0_base|v10_super_colreg_rag|v11_super_colreg_rag_cot)__W0_base__screening_v2_uncapped\.json$')
files = [f for f in glob.glob('Basic Simulator/Data/missions/_llm_runs/*.json') if pat.match(f.split('\\')[-1])]

need_redo = []
for f in sorted(files):
    d = json.loads(open(f, encoding='utf-8').read())
    ev = d.get('evaluation') or {}
    min_cpa = (ev.get('safety') or {}).get('min_cpa_m')
    mission, config = d.get('mission_id'), d.get('config')
    if min_cpa is not None and min_cpa < 450.0:
        need_redo.append((mission, config, min_cpa))

print(f"{len(files)} total v0/v10/v11 Imazu files, {len(need_redo)} need redoing (min_cpa < 450m)\n")
for mission, config, cpa in need_redo:
    print(f"  {mission:10s} {config:26s} min_cpa={cpa:.1f}m")

by_config = {}
for mission, config, _ in need_redo:
    by_config.setdefault(config, []).append(mission)
print()
for config, missions in by_config.items():
    print(f"{config}: {' '.join(sorted(set(missions)))}")
