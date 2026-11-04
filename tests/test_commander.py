"""nl_commander/commander.py の純粋なロジック(プロンプト・スキーマ・計画の検証・字幕)と、
偽の LLM + mock_run_skill.sh による逐次実行のテスト。"""
import json
import math

import pytest

import commander as C
from conftest import CRANE, NL

MOCK_SKILLS = NL / "skills.mock.json"
REAL_SKILLS = CRANE / "skills.json"
SKILL_NAMES = {"bow", "nod", "wave", "point_left", "point_right", "grip"}


@pytest.fixture
def skills():
    return C.load_skills(MOCK_SKILLS)


# ------------------------------------------------------------------ skills / prompt
@pytest.mark.parametrize("path", [MOCK_SKILLS, REAL_SKILLS])
def test_load_skills_and_hints(path):
    sk = C.load_skills(path)
    assert set(sk) == SKILL_NAMES
    # skill_hints.json の別名・例文がマージされる("_comment" キーは無視される)
    assert "礼" in sk["bow"]["aliases"]
    assert sk["wave"]["examples"] == ["ハローって手を振って"]


def test_real_and_mock_skills_share_descriptions():
    real = json.load(open(REAL_SKILLS, encoding="utf-8"))
    mock = json.load(open(MOCK_SKILLS, encoding="utf-8"))
    assert set(real) == set(mock)
    for k in real:
        assert real[k]["description"] == mock[k]["description"]


def test_find_skills_path_explicit_mock():
    p, is_real = C.find_skills_path(str(MOCK_SKILLS))
    assert p == MOCK_SKILLS and is_real is False


def test_sec_range_defaults():
    assert C.sec_range({}) == (C.SEC_MIN, C.SEC_MAX, C.SEC_DEFAULT)
    assert C.sec_range({"default_seconds": 6.2, "min_seconds": 2, "max_seconds": 9}) == (2.0, 9.0, 6.2)


def test_build_schema(skills):
    sc = C.build_schema(skills)
    assert sc["properties"]["status"]["enum"] == ["ok", "reject", "clarify"]
    assert sc["properties"]["steps"]["maxItems"] == C.MAX_STEPS
    assert set(sc["properties"]["steps"]["items"]["properties"]["skill"]["enum"]) == SKILL_NAMES
    # 生成順: 解釈・判定を先に書かせる
    assert list(sc["properties"])[:2] == ["interpretation", "status"]


def test_build_messages_fewshot_and_zeroshot(skills):
    few = C.build_messages("右を指して", skills, fewshot=True)
    zero = C.build_messages("右を指して", skills, fewshot=False)
    assert few[0]["role"] == zero[0]["role"] == "system"
    assert few[-1] == zero[-1] == {"role": "user", "content": "右を指して"}
    assert len(zero) == 2
    # few-shot は user/assistant の組。assistant 側はスキーマどおりの JSON
    pairs = few[1:-1]
    assert len(pairs) % 2 == 0 and len(pairs) >= 4
    for u, a in zip(pairs[::2], pairs[1::2]):
        assert u["role"] == "user" and a["role"] == "assistant"
        plan = json.loads(a["content"])
        assert plan["status"] in ("ok", "reject", "clarify")
        assert all(s["skill"] in skills for s in plan["steps"])
    for name in skills:
        assert f"- {name}:" in few[0]["content"]


def test_build_messages_env_switch(skills, monkeypatch):
    monkeypatch.setenv("COMMANDER_FEWSHOT", "0")
    assert len(C.build_messages("x", skills)) == 2


def test_build_text_prompt_contains_schema_and_examples(skills):
    system, user = C.build_text_prompt("右を指して", skills)
    assert '"enum": ["ok", "reject", "clarify"]' in system
    assert user.startswith("# 例") and user.endswith("# 本番\n指令: 右を指して\n出力:")


# ------------------------------------------------------------------ extract_json
def test_extract_json_with_preamble_and_fence():
    text = '考えます {"x": 1}\n```json\n{"status": "ok", "steps": [{"skill": "bow"}]}\n```'
    assert C.extract_json(text) == {"status": "ok", "steps": [{"skill": "bow"}]}


def test_extract_json_takes_last_plan():
    text = '{"status": "reject", "steps": []} 訂正: {"status": "ok", "steps": []}'
    assert C.extract_json(text)["status"] == "ok"


def test_extract_json_broken_brace_then_plan():
    assert C.extract_json('{ oops {"status": "clarify"}')["status"] == "clarify"


def test_extract_json_none():
    with pytest.raises(json.JSONDecodeError):
        C.extract_json("no json here {1: 2}")


# ------------------------------------------------------------------ normalize_plan
def _plan(status, *steps, **kw):
    return {"status": status, "interpretation": kw.get("i", "解釈"), "message": kw.get("m", "msg"),
            "steps": [dict(s) for s in steps]}


def test_normalize_ok(skills):
    plan, warns = C.normalize_plan(_plan("ok", {"skill": "bow", "seconds": 8, "reason": "r"}), skills)
    assert warns == []
    assert plan == {"status": "ok", "interpretation": "解釈", "message": "msg",
                    "steps": [{"skill": "bow", "seconds": 8.0, "reason": "r"}]}


def test_normalize_unknown_skill_dropped(skills):
    plan, warns = C.normalize_plan(_plan("ok", {"skill": "jump", "seconds": 3}, {"skill": "nod", "seconds": 3}), skills)
    assert [s["skill"] for s in plan["steps"]] == ["nod"]
    assert any("jump" in w for w in warns)


def test_normalize_seconds_clamped_and_default(skills):
    plan, warns = C.normalize_plan(_plan("ok", {"skill": "bow", "seconds": 99}, {"skill": "nod", "seconds": 0.1},
                                         {"skill": "wave", "seconds": "abc"}, {"skill": "grip"}), skills)
    assert [s["seconds"] for s in plan["steps"]] == [C.SEC_MAX, C.SEC_MIN, 8.0, 8.0]
    assert len(warns) == 2


def test_normalize_nan_seconds_uses_default(skills):
    # Python の json.loads は NaN を受け付ける。以前は NaN がそのまま実行秒数になっていた
    obj = json.loads('{"status": "ok", "steps": [{"skill": "bow", "seconds": NaN}]}')
    plan, warns = C.normalize_plan(obj, skills)
    sec = plan["steps"][0]["seconds"]
    assert math.isfinite(sec) and sec == 8.0
    assert warns


def test_normalize_infinite_seconds_clamped(skills):
    obj = json.loads('{"status": "ok", "steps": [{"skill": "bow", "seconds": Infinity}]}')
    plan, _ = C.normalize_plan(obj, skills)
    assert math.isfinite(plan["steps"][0]["seconds"])


def test_normalize_status_fixups(skills):
    plan, warns = C.normalize_plan(_plan("maybe", {"skill": "bow", "seconds": 8}), skills)
    assert plan["status"] == "clarify" and plan["steps"]
    plan, _ = C.normalize_plan(_plan("ok"), skills)
    assert plan["status"] == "reject"
    plan, _ = C.normalize_plan(_plan("reject", {"skill": "bow", "seconds": 8}), skills)
    assert plan["status"] == "reject" and plan["steps"] == []
    plan, _ = C.normalize_plan(_plan("clarify"), skills)
    assert plan["status"] == "clarify" and plan["steps"] == []


def test_normalize_max_steps(skills):
    plan, _ = C.normalize_plan(_plan("ok", *[{"skill": "nod", "seconds": 2}] * (C.MAX_STEPS + 3)), skills)
    assert len(plan["steps"]) == C.MAX_STEPS


def test_normalize_malformed_steps(skills):
    # 以前は AttributeError / TypeError で commander が落ちていた(終了コード 4 にならなかった)
    plan, warns = C.normalize_plan({"status": "ok", "steps": ["bow", None, {"skill": ["bow"]},
                                                              {"skill": "bow", "seconds": 8}]}, skills)
    assert [s["skill"] for s in plan["steps"]] == ["bow"]
    assert len(warns) == 3
    plan, warns = C.normalize_plan({"status": "ok", "steps": {"skill": "bow"}}, skills)
    assert plan["steps"] == [] and plan["status"] == "reject"


def test_normalize_non_object_raises(skills):
    with pytest.raises(RuntimeError):
        C.normalize_plan(["bow"], skills)


# ------------------------------------------------------------------ make_plan with a fake LLM
def fake_ollama(answers):
    """ollama_chat の代わり。answers を順に返す(str ならそのまま content、dict は JSON にする)。"""
    calls = []

    def chat(model, messages, schema, timeout=120, num_ctx=3072):
        calls.append(messages[-1]["content"])
        a = answers[min(len(calls) - 1, len(answers) - 1)]
        content = a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
        return {"message": {"content": content}, "load_duration": 1e8, "eval_count": 10,
                "prompt_eval_count": 100}, 0.25
    chat.calls = calls
    return chat


def test_make_plan_fake_llm(skills, monkeypatch):
    chat = fake_ollama([_plan("ok", {"skill": "point_right", "seconds": 6, "reason": "右"},
                              {"skill": "bow", "seconds": 6, "reason": "礼"})])
    monkeypatch.setattr(C, "ollama_chat", chat)
    r = C.make_plan("右を指してからお辞儀して", skills, "fake")
    assert [s["skill"] for s in r["plan"]["steps"]] == ["point_right", "bow"]
    assert r["latency_s"] == 0.25 and r["load_s"] == 0.1 and r["attempts"] == 1
    assert chat.calls == ["右を指してからお辞儀して"]


def test_make_plan_retries_then_fails(skills, monkeypatch):
    monkeypatch.setattr(C, "ollama_chat", fake_ollama(["not json", _plan("reject", m="no")]))
    r = C.make_plan("x", skills, "fake", retries=1)
    assert r["attempts"] == 2 and r["plan"]["status"] == "reject"
    monkeypatch.setattr(C, "ollama_chat", fake_ollama(["[1, 2]"]))
    with pytest.raises(RuntimeError):
        C.make_plan("x", skills, "fake", retries=1)


# ------------------------------------------------------------------ srt / container
def test_srt_time():
    assert C.srt_time(0) == "00:00:00,000"
    assert C.srt_time(1.9996) == "00:00:02,000"
    assert C.srt_time(3723.456) == "01:02:03,456"


def test_write_srt(tmp_path):
    p = tmp_path / "a.srt"
    C.write_srt(p, "指令", [{"i": 1, "n": 2, "skill": "bow", "reason": "礼", "start": 0.0, "end": 6.2},
                          {"i": 2, "n": 2, "skill": "nod", "reason": "", "start": 6.2, "end": 12.0}], offset=1.0)
    text = p.read_text(encoding="utf-8")
    assert text.startswith("1\n00:00:01,000 --> 00:00:07,200\n指令: 指令\n[1/2] bow  礼\n\n2\n")
    assert "[2/2] nod\n" in text  # 理由が空なら末尾の空白を落とす


def test_container_cmd_quotes(monkeypatch):
    monkeypatch.setattr(C, "find_docker", lambda: "docker")

    class A:
        container, container_runner, container_env = "pai_ros2", None, ["AUTO_START_SIM=0"]
    cmd = C.container_cmd(A, "bow", "6.2")
    assert cmd[:6] == ["docker", "exec", "-u", "ubuntu", "pai_ros2", "bash"]
    assert cmd[-1] == "AUTO_START_SIM=0 bash /workspace/a_crane/run_skill.sh bow 6.2"


# ------------------------------------------------------------------ end to end without hardware
def _main(tmp_path, monkeypatch, answer, *extra):
    monkeypatch.setattr(C, "ollama_chat", fake_ollama([answer]))
    d = tmp_path / f"run{len(list(tmp_path.iterdir()))}"  # 呼び出しごとに別ディレクトリ(ログは追記なので)
    d.mkdir()
    log, srt, status = d / "log.jsonl", d / "out.srt", d / "status.json"
    rc = C.main(["--skills", str(MOCK_SKILLS), "--runner", str(NL / "mock_run_skill.sh"),
                 "--log", str(log), "--srt", str(srt), "--status-file", str(status),
                 "--min-seconds-ratio", "0", *extra, "右を指してからお辞儀して"])
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return rc, events, srt, status


def test_main_runs_mock_runner(tmp_path, monkeypatch):
    answer = _plan("ok", {"skill": "point_right", "seconds": 1, "reason": "右"},
                   {"skill": "bow", "seconds": 1, "reason": "礼"})
    rc, events, srt, status = _main(tmp_path, monkeypatch, answer)
    assert rc == 0
    kinds = [e["event"] for e in events]
    assert kinds[:3] == ["start", "command", "plan"] and kinds[-1] == "done"
    ends = [e for e in events if e["event"] == "step_end"]
    assert [e["skill"] for e in ends] == ["point_right", "bow"]
    assert all(e["rc"] == 0 and e["dry_run"] is False for e in ends)
    assert srt.read_text(encoding="utf-8").count(" --> ") == 2
    assert json.loads(status.read_text(encoding="utf-8"))["state"] == "done"


def test_main_dry_run_reject_clarify(tmp_path, monkeypatch):
    ok = _plan("ok", {"skill": "nod", "seconds": 5, "reason": ""})
    rc, events, _, _ = _main(tmp_path, monkeypatch, ok, "--dry-run")
    assert rc == 0 and all(e.get("dry_run") for e in events if e["event"] == "step_end")
    rc, _, _, status = _main(tmp_path, monkeypatch, _plan("reject", m="できません"))
    assert rc == 2 and json.loads(status.read_text(encoding="utf-8"))["state"] == "rejected"
    # clarify: 標準入力が端末でなく -y も無ければ実行しない
    monkeypatch.setattr("sys.stdin", open(__file__, encoding="utf-8"))
    rc, events, _, _ = _main(tmp_path, monkeypatch, _plan("clarify", {"skill": "bow", "seconds": 1}))
    assert rc == 3 and not any(e["event"] == "step_start" for e in events)


def test_main_plan_only_and_llm_failure(tmp_path, monkeypatch, capsys):
    rc, _, _, _ = _main(tmp_path, monkeypatch, _plan("ok", {"skill": "bow", "seconds": 1}), "--plan-only")
    assert rc == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["steps"][0]["skill"] == "bow"
    rc, events, _, _ = _main(tmp_path, monkeypatch, "{broken")
    assert rc == 4 and events[-1]["event"] == "plan_error"
