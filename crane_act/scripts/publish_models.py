#!/usr/bin/env python3
"""Copy the final checkpoints to /workspace/a_crane/checkpoints/<skill> and (re)write skills.json.
usage: python3 publish_models.py bow=bow_n50_phase point_left=point_left_n40_phase ..."""
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skills import DESCRIPTIONS  # noqa: E402

A = '/workspace/a_crane'
HOME = os.path.expanduser('~')
path = f'{A}/skills.json'
d = json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}
for arg in sys.argv[1:]:
    skill, name = arg.split('=')
    src = f'{HOME}/checkpoints/{name}'
    dst = f'{A}/checkpoints/{skill}'
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # default duration = longest training episode (sim seconds) + 0.5 s margin
    frames = 0
    log = f'{HOME}/logs/convert_{name}.log'
    if os.path.exists(log):
        m = re.findall(r'padded to (\d+)', open(log, encoding='utf-8', errors='replace').read())
        frames = int(m[-1]) if m else 0
    d[skill] = {'model_path': dst, 'description': DESCRIPTIONS[skill],
                'default_seconds': round(frames / 20 + 0.5, 1) if frames else 8.0,
                'trained_from': name}
json.dump(d, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(json.dumps(d, ensure_ascii=False, indent=1))
