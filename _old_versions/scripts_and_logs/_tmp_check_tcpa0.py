import json, glob

files = sorted(glob.glob('Basic Simulator/Data/missions/_llm_runs/Imazu01__*screening_v2_uncapped.json'))
for f in files:
    d = json.loads(open(f, encoding='utf-8').read())
    for cp in d.get('checkpoints', []):
        if cp['time'] in (1600.0, 2000.0):
            dec = cp.get('decision') or {}
            print(f.split('\\')[-1], 't=', cp['time'])
            print('  action:', dec.get('action'), dec.get('degrees'), 'encounter_rule=', dec.get('encounter_rule'))
            print('  reasoning:', dec.get('reasoning'))
            m = cp.get('measurement') or {}
            print('  checks_fired:', m.get('checks_fired'))
            print()
