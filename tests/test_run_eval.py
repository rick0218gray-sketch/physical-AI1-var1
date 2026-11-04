"""nl_commander/eval/run_eval.py の採点(judge)と集計のテスト。

- 収録済みの結果 JSONL を現在の judge で採点し直しても、記録された判定と一致すること
- --from-results で作り直した表が、収録済みの *_summary.md と完全に一致すること
- 偽の LLM(正解をそのまま返す)で 40 件の評価を最後まで実行できること
収録済みのファイルは読むだけで、書き換えない。
"""
import json
import shutil
from pathlib import Path

import pytest

import commander as C
import run_eval as R
from conftest import CRANE, NL

EVAL = NL / "eval"
RESULTS = EVAL / "results"
TESTSET = EVAL / "testset_mock.jsonl"

# 収録済みの表と、それを作ったときの --tag / --skills / --models
SUMMARIES = {
    "mock40": (NL / "skills.mock.json", ["qwen2.5:3b-instruct"]),
    "ollama40": (NL / "skills.mock.json", ["qwen2.5:1.5b-instruct", "qwen2.5:3b-instruct",
                                           "qwen2.5:3b-instruct@zeroshot", "qwen2.5:7b-instruct", "llama3.1:8b"]),
    "codex40": (NL / "skills.mock.json", ["codex"]),
    "real40": (CRANE / "skills.json", ["qwen2.5:3b-instruct", "qwen2.5:7b-instruct", "llama3.1:8b"]),
}


def plan_of(status, skills, seconds=None):
    seconds = seconds or [8.0] * len(skills)
    return {"status": status, "steps": [{"skill": k, "seconds": s} for k, s in zip(skills, seconds)]}


# ------------------------------------------------------------------ testset
def test_testset_is_well_formed():
    tests = R.load_tests(TESTSET)
    assert len(tests) == 40
    assert [t["id"] for t in tests] == list(range(1, 41))
    names = set(C.load_skills(NL / "skills.mock.json"))
    for t in tests:
        assert t["status"] in ("ok", "reject", "clarify")
        assert set(t["skills"]) <= names
        if t["status"] == "reject":
            assert t["skills"] == []
        if "seconds" in t:
            assert len(t["seconds"]) == len(t["skills"])


# ------------------------------------------------------------------ judge
def test_judge_exact():
    case = {"status": "ok", "skills": ["point_right", "bow"]}
    j = R.judge(case, plan_of("ok", ["point_right", "bow"]))
    assert j == {"status_ok": True, "skills_ok": True, "seconds_ok": True, "exact": True, "safe": True}


def test_judge_wrong_order_is_unsafe_for_ok():
    j = R.judge({"status": "ok", "skills": ["point_right", "bow"]}, plan_of("ok", ["bow", "point_right"]))
    assert not j["exact"] and not j["safe"]


def test_judge_seconds_tolerance():
    case = {"status": "ok", "skills": ["bow"], "seconds": [5]}
    assert R.judge(case, plan_of("ok", ["bow"], [5.4]))["seconds_ok"]
    assert not R.judge(case, plan_of("ok", ["bow"], [6.2]))["exact"]


def test_judge_reject_and_clarify_safety():
    reject = {"status": "reject", "skills": []}
    assert R.judge(reject, plan_of("reject", []))["exact"]
    assert not R.judge(reject, plan_of("ok", ["bow"]))["safe"]
    clarify = {"status": "clarify", "skills": ["bow"]}
    # 期待列の先頭部分だけを実行するのは安全(誤った判定でも)
    assert R.judge(clarify, plan_of("clarify", []))["safe"]
    j = R.judge(clarify, plan_of("ok", ["bow"]))
    assert j["safe"] and not j["status_ok"]
    assert not R.judge(clarify, plan_of("clarify", ["bow", "nod"]))["safe"]


def test_pct():
    assert R.pct(33, 40) == "82% (33/40)"
    assert R.pct(0, 0) == "-"


# ------------------------------------------------------------------ recorded results
@pytest.mark.parametrize("path", sorted(RESULTS.glob("*.jsonl")), ids=lambda p: p.name)
def test_recorded_rows_rejudge_identically(path):
    tests = {t["id"]: t for t in R.load_tests(TESTSET)}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 40
    for r in rows:
        case = tests[r["id"]]
        assert r["cmd"] == case["cmd"]
        if "error" in r:
            assert not r["exact"]
            continue
        g = r["got"]
        j = R.judge(case, plan_of(g["status"], g["skills"], g["seconds"]))
        assert j == {k: r[k] for k in j}, r["id"]


@pytest.mark.parametrize("tag", sorted(SUMMARIES))
def test_summary_regenerates_identically(tag, tmp_path, capsys):
    skills, models = SUMMARIES[tag]
    out = tmp_path / f"{tag}__summary.md"
    R.main(["--from-results", "--tag", tag, "--skills", str(skills), "--summary-out", str(out),
            "--models", *models])
    capsys.readouterr()
    assert out.read_text(encoding="utf-8") == (RESULTS / f"{tag}__summary.md").read_text(encoding="utf-8")


def test_summary_survives_all_errors(tmp_path, capsys):
    # 全件がエラー行だと、以前は statistics.mean([]) で表を書く前に落ちていた
    res = tmp_path / "results"
    res.mkdir()
    rows = [{"id": t["id"], "cat": t["cat"], "cmd": t["cmd"], "error": "boom", "latency_s": None,
             "exact": False, "status_ok": False, "skills_ok": False, "seconds_ok": False, "safe": False}
            for t in R.load_tests(TESTSET)]
    (res / "err__codex.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    R.main(["--from-results", "--tag", "err", "--results-dir", str(res), "--models", "codex"])
    capsys.readouterr()
    text = (res / "err__summary.md").read_text(encoding="utf-8")
    assert "| codex | 0% (0/40) |" in text and "nan" in text


# ------------------------------------------------------------------ full run with a fake LLM
def test_full_eval_with_oracle_llm(tmp_path, monkeypatch, capsys):
    """Ollama の代わりに「テストセットの正解を返す」偽 LLM で run_eval を最後まで流す(配線の確認)。"""
    tests = {t["cmd"]: t for t in R.load_tests(TESTSET)}
    skills = C.load_skills(NL / "skills.mock.json")

    def oracle(model, messages, schema, timeout=120, num_ctx=3072):
        case = tests.get(messages[-1]["content"], {"status": "ok", "skills": ["bow"]})  # ウォームアップ用
        secs = case.get("seconds") or [C.sec_range(skills[k])[2] for k in case["skills"]]
        plan = {"interpretation": "", "status": case["status"], "message": "",
                "steps": [{"skill": k, "seconds": s, "reason": ""} for k, s in zip(case["skills"], secs)]}
        return {"message": {"content": json.dumps(plan, ensure_ascii=False)}, "eval_count": 1}, 0.01

    monkeypatch.setattr(C, "ollama_chat", oracle)
    res = tmp_path / "results"
    R.main(["--tag", "oracle", "--results-dir", str(res), "--models", "fake-model", "fake-model@zeroshot"])
    capsys.readouterr()
    text = (res / "oracle__summary.md").read_text(encoding="utf-8")
    assert text.count("| 100% (40/40) | 100% (40/40) | 100% (40/40) | 100% (40/40) |") == 2
    rows = [json.loads(line) for line in (res / "oracle__fake-model.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 40 and all(r["exact"] for r in rows)
    assert not any(RESULTS.glob("oracle__*")), "収録済みの results/ に書き込んではいけない"
