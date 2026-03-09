import json
from pathlib import Path

hc_path = Path('memory/hard_cases.json')
replay_path = Path('memory/replay.jsonl')

hc = json.loads(hc_path.read_text(encoding='utf-8')) if hc_path.exists() else []
replay = [json.loads(l) for l in replay_path.read_text(encoding='utf-8').splitlines() if l.strip()] if replay_path.exists() else []

from collections import Counter, defaultdict

hc_domains = Counter(c.get('domain') for c in hc)
replay_domains = Counter(r.get('domain') for r in replay)

replay_ok = defaultdict(int)
for r in replay:
    if r.get('ok'):
        replay_ok[r.get('domain')] += 1

print('hard_cases per domain:')
for d, cnt in hc_domains.most_common():
    print(f'  {d}: {cnt}')

print('\nreplay per domain (total / ok):')
for d, cnt in replay_domains.most_common():
    ok = replay_ok.get(d, 0)
    print(f'  {d}: {cnt} / {ok} ok')

print('\nNotes:')
print('  - hard_cases stores only cases where model result was not verified (ok==False).')
print('  - If a domain is absent from hard_cases, it may have been solved in replay (ok==True), or not sampled during memory refresh.')
