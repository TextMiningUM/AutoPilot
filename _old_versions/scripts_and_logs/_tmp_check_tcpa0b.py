import json, glob

files = sorted(glob.glob('Basic Simulator/Data/missions/_llm_runs/Imazu01__*screening_v2_uncapped.json'))
for f in files:
    d = json.loads(open(f, encoding='utf-8').read())
    for cp in d.get('checkpoints', []):
        dec = cp.get('decision') or {}
        reasoning = (dec.get('reasoning') or '').lower()
        if 'already' in reasoning or 'passed' in reasoning or 'past' in reasoning:
            print(f.split('\\')[-1], 't=', cp['time'], 'action=', dec.get('action'), dec.get('degrees'))
            print('  reasoning:', dec.get('reasoning'))
            m = cp.get('measurement') or {}
            det = m.get('details') or {}
            for k in ('A','B','C','D'):
                if k in det:
                    print('  ', k, det[k])
            print()
