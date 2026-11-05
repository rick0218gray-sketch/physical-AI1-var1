#!/usr/bin/env python3
"""skills.json が一部(例: bow だけ)しか無い段階でも動くかの確認。テストセットのうち、残したスキルだけで答えられるものを評価する。
  python eval/subset_check.py bow
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import commander as C
import run_eval as R

keep = set(sys.argv[1:]) or {"bow"}
skills = {k: v for k, v in C.load_skills(HERE.parent / "skills.mock.json").items() if k in keep}
tests = []
for t in R.load_tests(HERE / "testset_mock.jsonl"):
    if t["status"] == "ok" and set(t["skills"]) <= keep:
        tests.append(t)
    elif t["status"] == "ok":  # 使えないスキルを要求 -> 期待は「実行しない」(reject か clarify で steps は空 or 使えるものだけ)
        usable = [s for s in t["skills"] if s in keep]
        tests.append({**t, "status": "reject" if not usable else "clarify", "skills": usable, "_flex": True})
    elif t["status"] == "clarify":
        tests.append({**t, "skills": [s for s in t["skills"] if s in keep]})
    else:
        tests.append(t)
ok = 0
for t in tests:
    r = C.make_plan(t["cmd"], skills, C.DEFAULT_MODEL)
    j = R.judge(t, r["plan"])
    good = j["exact"] or (t.get("_flex") and j["safe"] and r["plan"]["status"] in ("reject", "clarify"))
    ok += bool(good)
    if not good:
        print("NG", t["id"], t["cmd"], "->", r["plan"]["status"], [s["skill"] for s in r["plan"]["steps"]], "期待", t["status"], t["skills"])
print(f"{ok}/{len(tests)} (skills={sorted(keep)})")
