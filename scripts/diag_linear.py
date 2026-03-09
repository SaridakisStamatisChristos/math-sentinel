import json, re
from pathlib import Path

hc = json.loads(Path('memory/hard_cases.json').read_text(encoding='utf-8'))
replay = []
if Path('memory/replay.jsonl').exists():
    replay = [json.loads(l) for l in Path('memory/replay.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

lin_hc = [c for c in hc if c.get('domain')=='linear_equation']
count = len(lin_hc)

pattern = re.compile(r"Solve:\s*([+-]?\d+)x\s*\+\s*([+-]?\d+)\s*=\s*([+-]?\d+)")

mismatches = []
for c in lin_hc:
    m = pattern.search(c['task'])
    stored = c.get('answer','').strip()
    if not m:
        continue
    a = int(m.group(1)); b=int(m.group(2)); cc=int(m.group(3))
    try:
        x = (cc - b)/a
    except Exception:
        continue
    correct = f"x={x}"
    if stored=='' or (stored.startswith('x=') and abs(float(stored.split('=')[1]) - x) > 1e-6) or (not stored.startswith('x=')):
        mismatches.append({'task':c['task'],'stored':stored,'correct':correct})

lin_replay = [r for r in replay if r.get('domain')=='linear_equation']
replay_ok = sum(1 for r in lin_replay if r.get('ok'))

print(f'linear hard_cases total={count}, mismatches={len(mismatches)}')
print('\nFirst 10 mismatches:')
for m in mismatches[:10]:
    print(m)

print('\nreplay linear total=', len(lin_replay), 'ok=', replay_ok)
print('\nSample replay lines (first 10):')
for r in lin_replay[:10]:
    print(r)
