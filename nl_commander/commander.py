#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自然言語ロボットコマンダー (ローカル LLM / Ollama 版)

日本語の指令 -> LLM(JSON スキーマ制約付き) -> スキル列の計画 -> run_skill.sh で順に実行。
  python commander.py "右を指してからお辞儀して"            # 計画を表示して実行(A の run_skill.sh があれば)
  python commander.py --dry-run "こんにちはって挨拶して"      # 実行コマンドを表示するだけ
  python commander.py --plan-only "…"                         # 計画だけ(JSON を標準出力)
  python commander.py -i                                      # 対話モード

依存は標準ライブラリのみ(urllib)。ホスト(WSL/Windows)でもコンテナ(pai_ros2)内でも動く。

終了コード: 0=成功(または dry-run) 1=その他 2=範囲外で拒否 3=確認待ち(未承認) 4=LLM 失敗 5=スキル実行失敗
"""
import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get("COMMANDER_MODEL", "qwen2.5:3b-instruct")
KEEP_ALIVE = os.environ.get("COMMANDER_KEEP_ALIVE", "30s")
MAX_STEPS = 8
SEC_MIN, SEC_MAX = 1.0, 30.0
SEC_DEFAULT = 8.0  # A の run_skill.sh も default_seconds 未設定なら 8 秒

# A(ストリームA) が置く実ファイル。見つからなければモックにフォールバックする。
SKILLS_CANDIDATES = [
    Path("/workspace/a_crane/skills.json"),
    HERE.parent / "crane_act" / "skills.json",
]
RUNNER_CANDIDATES = [
    Path("/workspace/a_crane/run_skill.sh"),
    HERE.parent / "crane_act" / "run_skill.sh",
]
MOCK_SKILLS = HERE / "skills.mock.json"


# ----------------------------------------------------------------- skills
def find_skills_path(explicit=None):
    """(path, is_real) を返す。"""
    if explicit:
        p = Path(explicit)
        return p, p.resolve() != MOCK_SKILLS.resolve()
    for p in SKILLS_CANDIDATES:
        if p.exists():
            return p, True
    return MOCK_SKILLS, False


def load_skills(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # {skill: {model_path, description, [default_seconds, min_seconds, max_seconds, examples]}}
    if isinstance(data, dict) and "skills" in data and isinstance(data["skills"], dict):
        data = data["skills"]
    skills = {}
    for name, v in data.items():
        if isinstance(v, str):
            v = {"description": v}
        skills[name] = v
    return apply_hints(skills)


HINTS_FILE = HERE / "skill_hints.json"


def apply_hints(skills, path=None):
    """skill_hints.json の別名・例文を skills にマージ(skills.json 自体は書き換えない)。"""
    p = Path(path) if path else HINTS_FILE
    if not p.exists():
        return skills
    try:
        hints = json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa
        return skills
    for name, h in hints.items():
        if name in skills and isinstance(h, dict):
            sk = skills[name]
            if h.get("aliases") and "aliases" not in sk:
                sk["aliases"] = h["aliases"]
            if h.get("examples") and "examples" not in sk:
                sk["examples"] = h["examples"]
    return skills


def find_runner(explicit=None):
    if explicit:
        return Path(explicit).resolve() if Path(explicit).exists() else None
    for p in RUNNER_CANDIDATES:
        if p.exists():
            return p
    return None


# ----------------------------------------------------------------- prompt / schema
def sec_range(sk):
    lo = float(sk.get("min_seconds", SEC_MIN))
    hi = float(sk.get("max_seconds", SEC_MAX))
    df = float(sk.get("default_seconds", SEC_DEFAULT))
    return lo, hi, df


def build_schema(skills):
    # 生成順 = プロパティ順。先に解釈と判定(ok/reject/clarify)を決めさせてから steps を書かせる。
    step = {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "enum": list(skills.keys())},
            "seconds": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["skill", "seconds", "reason"],
    }
    return {
        "type": "object",
        "properties": {
            "interpretation": {"type": "string"},
            "status": {"type": "string", "enum": ["ok", "reject", "clarify"]},
            "message": {"type": "string"},
            "steps": {"type": "array", "maxItems": MAX_STEPS, "items": step},
        },
        "required": ["interpretation", "status", "message", "steps"],
    }


def build_system_prompt(skills):
    lines = [
        "あなたはロボットアーム(CRANE+ V2)の動作プランナーです。",
        "ユーザーの日本語の指令を、下の「使えるスキル」だけを使った実行計画 JSON に変換します。",
        "",
        "# 使えるスキル",
    ]
    for name, sk in skills.items():
        lo, hi, df = sec_range(sk)
        desc = re.sub(r"\([^)]*\)|（[^）]*）", "", sk.get("description", "")).strip()
        al = f" / 言い換え: {'・'.join(sk['aliases'])}" if sk.get("aliases") else ""
        lines.append(f"- {name}: {desc}{al} (標準 {df:g} 秒)")
    lines += [
        "",
        "# 出力 JSON のフィールド",
        "- interpretation: 指令を 1 文で言い換える",
        "- steps: 実行するスキルを実行順に並べた配列。各要素は skill / seconds / reason",
        "- status: ok(全部実行できる) / reject(実行できる動作が全く無い) / clarify(一部だけ可能・曖昧・確認が必要)",
        "- message: ユーザーへの一言。ok なら計画の要約、reject なら断る理由、clarify なら確認の質問",
        "",
        "# ルール",
        "1. 指令の中の動作を、言われた順に steps へ 1 つずつ入れる。「〜してから」「〜して、次に」「〜のあと」は文の順序どおり。",
        "2. 「2回」「3回」と回数があれば、その回数だけ同じスキルを繰り返し並べる。",
        "3. seconds は、指令に秒数があればその値、無ければそのスキルの標準秒数。",
        "4. 使えるスキルの言い換えは同じスキルとして扱う(意味が同じなら表現が違っても良い)。",
        "5. 使えるスキルで表せない動作(物を取る・歩く・歌う・質問への回答・危険な動作など)は絶対に steps に入れない。",
        "   全く実行できない指令は status=reject, steps=[] とし、message に理由を書く。",
        "   一部だけ実行できる指令は status=clarify とし、実行できる部分だけを steps に入れ、message で「〜はできませんが、〜だけ実行しますか?」と確認する。",
        "   何をしてよいか曖昧な指令(「あれをやって」など)は status=clarify, steps=[] とし、message で聞き返す。",
        "6. skill には上のスキル名しか使わない。interpretation / reason / message はユーザーが読むので、スキル名ではなく日本語で動作の内容を書く。",
    ]
    return "\n".join(lines)


def sample_phrase(name, sk):
    """few-shot 用の日本語指令例。skills.json の examples、無ければ description 中の「…」を使う。"""
    ex = sk.get("examples")
    if isinstance(ex, list) and ex:
        return str(ex[0])
    m = re.search(r"「([^」]+)」", sk.get("description", ""))
    if m:
        return m.group(1)
    label = short_desc(sk)
    return label if label != "動作" else None


def short_desc(sk):
    d = sk.get("description", "")
    d = re.split(r"[。(（:：]", d)[0].strip()
    return d or "動作"


def _ex(user, interp, steps, status, message, skills):
    """few-shot の user/assistant 1 組。steps = [(skill, seconds, reason)]"""
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": json.dumps({
            "interpretation": interp, "status": status, "message": message,
            "steps": [{"skill": k, "seconds": sec, "reason": r} for k, sec, r in steps]}, ensure_ascii=False)},
    ]


def build_messages(command, skills, fewshot=None):
    if fewshot is None:
        fewshot = os.environ.get("COMMANDER_FEWSHOT", "1") != "0"
    msgs = [{"role": "system", "content": build_system_prompt(skills)}]
    if not fewshot:
        msgs.append({"role": "user", "content": command})
        return msgs
    msgs += _ex("今日の天気は?", "天気を尋ねている。ロボットの動作ではない。", [], "reject",
                "それはロボットの動作ではないので実行できません。", skills)
    msgs += _ex("ドアまで走って", "走る動作を求めている。使えるスキルに無い。", [], "reject",
                "走る動作はできないので実行できません。", skills)
    names = [n for n in skills if sample_phrase(n, skills[n])]
    if names:
        # 先頭のスキルに偏らないよう、末尾側のスキルを例に使う
        a = names[-1]
        b = names[-2] if len(names) >= 2 else names[-1]
        c = names[-3] if len(names) >= 3 else names[0]
        pa, pb, pc = (sample_phrase(n, skills[n]) for n in (a, b, c))
        da, db, dc = (sec_range(skills[n])[2] for n in (a, b, c))
        xa, xb, xc = (short_desc(skills[n]) for n in (a, b, c))
        msgs += _ex(pa, f"{xa}。", [(a, da, xa)], "ok", f"{xa}ます。".replace("るます", "ります"), skills)
        if len(names) >= 2:  # 複数スキルのときだけ「順序」の例を入れる(1 スキルだと例が破綻する)
            sep = "から" if pb.endswith(("て", "で")) else "、そのあと"
            msgs += _ex(f"{pb}{sep}{pc}", f"先に{xb}、そのあと{xc}。",
                        [(b, db, f"最初に{xb}"), (c, dc, f"次に{xc}")], "ok",
                        "言われた順に実行します。", skills)
        msgs += _ex(f"7秒かけて{pc}", f"7秒かけて{xc}。", [(c, 7, f"7秒指定: {xc}")], "ok",
                    "7秒で実行します。", skills)
        msgs += _ex(f"2回続けて{pb}", f"{xb}を2回繰り返す。",
                    [(b, db, f"1回目: {xb}"), (b, db, f"2回目: {xb}")], "ok", "2回実行します。", skills)
        msgs += _ex(f"{pa}、それから窓を開けて", f"{xa}は可能だが、窓を開ける動作は無い。",
                    [(a, da, xa)], "clarify",
                    "窓を開けることはできません。できる部分だけ実行しますか?", skills)
    msgs.append({"role": "user", "content": command})
    return msgs


# ----------------------------------------------------------------- ollama
def _default_gateway():
    try:
        with open("/proc/net/route") as f:
            for line in f.read().splitlines()[1:]:
                p = line.split()
                if p[1] == "00000000":
                    b = bytes.fromhex(p[2])[::-1]
                    return ".".join(str(x) for x in b)
    except Exception:
        pass
    return None


def candidate_urls():
    env = os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST")
    c = []
    if env:
        c.append(env if env.startswith("http") else "http://" + env)
    c += ["http://host.docker.internal:11434", "http://127.0.0.1:11434"]
    gw = _default_gateway()
    if gw:
        c.append(f"http://{gw}:11434")
    return c


_URL_CACHE = None


def ollama_url():
    global _URL_CACHE
    if _URL_CACHE:
        return _URL_CACHE
    errs = []
    for u in candidate_urls():
        try:
            urllib.request.urlopen(u + "/api/version", timeout=2).read()
            _URL_CACHE = u
            return u
        except Exception as e:  # noqa
            errs.append(f"{u}: {e}")
    raise RuntimeError("Ollama に接続できません: " + " | ".join(errs))


def ollama_chat(model, messages, schema, timeout=120, num_ctx=3072):
    body = {
        "model": model, "messages": messages, "stream": False,
        "format": schema, "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": 700},
    }
    req = urllib.request.Request(
        ollama_url() + "/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.loads(r.read())
    wall = time.perf_counter() - t0
    return res, wall


# ----------------------------------------------------------------- cloud CLI backends (codex / claude)
def build_text_prompt(command, skills):
    """CLI 用: システム文 + スキーマ + few-shot 例 + 指令 を 1 つの文章にする。(system, user) を返す。"""
    msgs = build_messages(command, skills)
    system = msgs[0]["content"]
    schema = json.dumps(build_schema(skills), ensure_ascii=False)
    system += ("\n\n# 出力形式\n次の JSON スキーマに従う JSON オブジェクトを 1 つだけ出力する。"
               "説明文・前置き・コードフェンスは付けない。\n" + schema)
    user = []
    pairs = msgs[1:-1]
    if pairs:
        user.append("# 例")
        for i in range(0, len(pairs), 2):
            user.append(f"指令: {pairs[i]['content']}\n出力: {pairs[i + 1]['content']}")
    user.append(f"# 本番\n指令: {command}\n出力:")
    return system, "\n\n".join(user)


def extract_json(text):
    """出力に混じった前置き・コードフェンス等から、plan らしい JSON オブジェクト(最後に現れたもの)を取り出す。"""
    dec = json.JSONDecoder()
    found = None
    i = 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text, j)
            if isinstance(obj, dict) and "status" in obj:
                found = obj
            i = end
        except json.JSONDecodeError:
            i = j + 1
    if found is None:
        raise json.JSONDecodeError("plan JSON が見つからない", text, 0)
    return found


def _which(*names):
    import shutil
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    return None


def cli_plan(backend, model, command, skills, timeout=240):
    """codex / claude の CLI を 1 回呼んで (raw_text, wall_s, extra) を返す。ツールは使わせない。"""
    import tempfile
    system, user = build_text_prompt(command, skills)
    cwd = tempfile.mkdtemp(prefix="commander_cli_")  # プロジェクトを読ませないため空の作業ディレクトリ
    extra = {}
    if backend == "codex":
        exe = _which("codex", "codex.exe", "codex.cmd")
        if not exe:
            raise RuntimeError("codex CLI が見つからない(Windows 側で実行してください)")
        cmd = [exe, "exec", "--skip-git-repo-check", "-s", "read-only", "--ephemeral", "--color", "never"]
        if model and model != DEFAULT_MODEL:
            cmd += ["-m", model]
        cmd += ["-"]
        stdin = system + "\n\n" + user
    elif backend == "claude":
        exe = _which("claude", "claude.exe", "claude.cmd")
        if not exe:
            raise RuntimeError("claude CLI が見つからない(Windows 側で実行してください)")
        cmd = [exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--disable-slash-commands", "--system-prompt", system]
        if model and model != DEFAULT_MODEL:
            cmd += ["--model", model]
        stdin = user
    else:
        raise ValueError(backend)
    t0 = time.perf_counter()
    p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding="utf-8",
                       cwd=cwd, timeout=timeout)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise RuntimeError(f"{backend} 終了コード {p.returncode}: {(p.stderr or p.stdout)[-300:]}")
    out = p.stdout
    if backend == "claude":
        try:
            j = json.loads(out)
            extra = {"cost_usd": j.get("total_cost_usd"), "api_ms": j.get("duration_api_ms")}
            out = j.get("result", "")
        except json.JSONDecodeError:
            pass
    return out, wall, extra


# ----------------------------------------------------------------- planning
def normalize_plan(obj, skills):
    """LLM 出力を検証・正規化する。(plan, warnings)"""
    if not isinstance(obj, dict):
        # JSON としては正しいがオブジェクトでない(配列・文字列など)。make_plan の再試行対象にする
        raise RuntimeError(f"plan JSON がオブジェクトではない: {type(obj).__name__}")
    warns = []
    status = obj.get("status")
    if status not in ("ok", "reject", "clarify"):
        status = "clarify"
        warns.append("status 不正 -> clarify")
    steps = []
    raw_steps = obj.get("steps") or []
    if not isinstance(raw_steps, list):
        warns.append(f"steps が配列でない ({type(raw_steps).__name__}) -> 空として扱う")
        raw_steps = []
    for st in raw_steps:
        if not isinstance(st, dict):
            warns.append(f"不正なステップ {st!r} を除外")
            continue
        name = st.get("skill")
        if not isinstance(name, str) or name not in skills:
            warns.append(f"未知スキル {name!r} を除外")
            continue
        lo, hi, df = sec_range(skills[name])
        try:
            sec = float(st.get("seconds", df))
        except (TypeError, ValueError):
            sec = df
        if not math.isfinite(sec):  # Python の json は NaN / Infinity も受け付ける
            warns.append(f"{name} の秒数 {sec} は不正 -> 標準 {df:g} 秒")
            sec = df
        if not (lo <= sec <= hi):
            warns.append(f"{name} の秒数 {sec} を {lo:g}-{hi:g} に丸め")
            sec = min(max(sec, lo), hi)
        steps.append({"skill": name, "seconds": sec, "reason": str(st.get("reason", ""))})
    steps = steps[:MAX_STEPS]
    if status == "ok" and not steps:
        status = "reject"
        warns.append("ok だが steps が空 -> reject")
    if status == "reject" and steps:
        warns.append("reject なのに steps あり -> steps を破棄")
        steps = []
    return {
        "status": status,
        "interpretation": str(obj.get("interpretation", "")),
        "message": str(obj.get("message", "")),
        "steps": steps,
    }, warns


def make_plan(command, skills, model=DEFAULT_MODEL, retries=1, backend="ollama"):
    """指令を計画に変換する。戻り値: dict(plan, latency_s, load_s, eval_tokens, model, backend, warnings, raw)"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            if backend == "ollama":
                res, wall = ollama_chat(model, build_messages(command, skills), build_schema(skills))
                raw = res.get("message", {}).get("content", "")
                obj = json.loads(raw)
                meta = {"load_s": round(res.get("load_duration", 0) / 1e9, 3),
                        "eval_tokens": res.get("eval_count"), "prompt_tokens": res.get("prompt_eval_count")}
            else:
                raw, wall, extra = cli_plan(backend, model, command, skills)
                obj = extract_json(raw)
                meta = {"load_s": 0, "eval_tokens": None, "prompt_tokens": None, **extra}
            plan, warns = normalize_plan(obj, skills)
            label = model if backend == "ollama" or model != DEFAULT_MODEL else "(CLI 既定)"
            return {"plan": plan, "model": label, "backend": backend, "latency_s": round(wall, 3),
                    "warnings": warns, "raw": raw, "attempts": attempt + 1, **meta}
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError, RuntimeError, OSError,
                subprocess.TimeoutExpired) as e:
            last_err = e
    raise RuntimeError(f"LLM 計画の生成に失敗: {last_err}")


# ----------------------------------------------------------------- logging / srt / status
class RunLog:
    def __init__(self, path, run_id):
        self.path, self.run_id = path, run_id
        self.t0 = time.perf_counter()
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event, **kw):
        rec = {"ts": dt.datetime.now().isoformat(timespec="milliseconds"),
               "t": round(time.perf_counter() - self.t0, 3), "run": self.run_id, "event": event}
        rec.update(kw)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def srt_time(sec):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, command, timeline, offset=0.0):
    """timeline: [{'i','n','skill','reason','start','end'}] (実行開始を 0 とする秒)"""
    cues = []
    for tl in timeline:
        cues.append((tl["start"] + offset, tl["end"] + offset,
                     f"指令: {command}\n[{tl['i']}/{tl['n']}] {tl['skill']}  {tl['reason']}".rstrip()))
    with open(path, "w", encoding="utf-8") as f:
        for k, (a, b, txt) in enumerate(cues, 1):
            f.write(f"{k}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n")


def write_status(path, **kw):
    if not path:
        return
    kw["updated"] = dt.datetime.now().isoformat(timespec="milliseconds")
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(kw, f, ensure_ascii=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------- execute
CONTAINER_RUNNER = "/workspace/a_crane/run_skill.sh"


def find_docker():
    """docker 実行ファイル。WSL には docker が無いので Windows 側の docker.exe を探す。"""
    import shutil
    env = os.environ.get("COMMANDER_DOCKER")
    cands = [env] if env else []
    cands += ["docker.exe", "docker"]
    for c in cands:
        if not c:
            continue
        if os.path.isabs(c):
            if os.path.exists(c):
                return c
        else:
            w = shutil.which(c)
            if w:
                return w
    return None


def container_cmd(args, skill, sec):
    """docker exec 経由で A の run_skill.sh を呼ぶコマンド(リスト)。"""
    runner = args.container_runner or CONTAINER_RUNNER
    envs = "".join(f"{e} " for e in (args.container_env or []))
    inner = f"{envs}bash {shlex.quote(runner)} {shlex.quote(skill)} {sec}"
    return [find_docker() or "docker", "exec", "-u", "ubuntu", args.container, "bash", "-lc", inner]


def container_ready(args):
    try:
        p = subprocess.run([find_docker() or "docker", "exec", "-u", "ubuntu", args.container, "bash", "-lc",
                            f"test -f {shlex.quote(args.container_runner or CONTAINER_RUNNER)}"],
                           capture_output=True, timeout=30)
        return p.returncode == 0
    except Exception:  # noqa
        return False

def run_plan(command, result, skills, args, log):
    plan = result["plan"]
    dry = args.dry_run
    if args.container:
        runner = args.container_runner or CONTAINER_RUNNER
        if not dry and not container_ready(args):
            print(f"[警告] コンテナ {args.container} の {runner} が実行できないため dry-run にフォールバックします", file=sys.stderr)
            log.emit("fallback_dry_run", reason="container runner not ready")
            dry = True
    else:
        runner = find_runner(args.runner)
        if not dry and runner is None:
            print("[警告] run_skill.sh が見つからないため dry-run にフォールバックします", file=sys.stderr)
            log.emit("fallback_dry_run", reason="run_skill.sh not found")
            dry = True
    runner_s = str(runner) if runner else CONTAINER_RUNNER
    n = len(plan["steps"])
    timeline, clock, rc_all = [], 0.0, 0
    run_t0 = time.perf_counter()
    status_base = dict(command=command, message=plan["message"], total=n)
    write_status(args.status_file, state="running", index=0, **status_base)
    for i, st in enumerate(plan["steps"], 1):
        # ACT 方策は 1 回の動作全体を default_seconds 前後で学習している。それより短い秒数だと動作が途中で切れるため、
        # 実行時は default_seconds * --min-seconds-ratio 未満にならないようにする(計画と JSONL には LLM の値も残す)
        dflt = skills.get(st["skill"], {}).get("default_seconds")
        if dflt and st["seconds"] < float(dflt) * args.min_seconds_ratio:
            new_sec = round(float(dflt) * args.min_seconds_ratio, 2)
            print(f"  [補正] {st['skill']} の {st['seconds']:g} 秒は短すぎるため {new_sec:g} 秒で実行(動作が途中で切れないように)")
            log.emit("seconds_clamped", skill=st["skill"], requested=st["seconds"], used=new_sec)
            st = {**st, "seconds": new_sec}
        sec_s = f"{st['seconds']:g}"
        if args.container:
            cmd = container_cmd(args, st["skill"], sec_s)
        else:
            cmd = [runner_s, st["skill"], sec_s]
            if not os.access(runner_s, os.X_OK):  # 実行ビットが無い(Windows 共有経由など)ときは bash 経由
                cmd = ["bash"] + cmd
        log.emit("step_start", i=i, n=n, skill=st["skill"], seconds=st["seconds"],
                 cmd=" ".join(shlex.quote(c) for c in cmd), dry_run=dry)
        print(f"[{i}/{n}] {' '.join(shlex.quote(c) for c in cmd)}   # {st['reason']}")
        write_status(args.status_file, state="running", index=i, skill=st["skill"],
                     reason=st["reason"], seconds=st["seconds"], **status_base)
        t0 = time.perf_counter()
        rc = 0
        motion = None      # (開始, 終了) 実行開始からの秒。runner が "wall=" を出すときだけ分かる
        skill_ok = None    # A の eval_traj が出す success=True/False
        if dry:
            if args.realtime:
                time.sleep(st["seconds"])
            elapsed = st["seconds"]
            start_abs = clock
        else:
            start_abs = t0 - run_t0
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", errors="replace")
                timer = threading.Timer(st["seconds"] + args.exec_timeout_margin, proc.kill)
                timer.start()
                for line in proc.stdout:
                    sys.stdout.write("    | " + line)
                    sys.stdout.flush()
                    m = re.search(r"done: .*?wall=([0-9.]+)s", line)
                    if m:  # runner 自身が測ったスキル実行時間。字幕の区間をこれに合わせる
                        t_done = time.perf_counter() - run_t0
                        motion = (t_done - float(m.group(1)), t_done)
                    m = re.search(r"success=(True|False)", line)
                    if m:
                        skill_ok = m.group(1) == "True"
                rc = proc.wait()
                timer.cancel()
                if rc < 0:
                    rc = 124
            except OSError as e:
                print(f"[エラー] 実行失敗: {e}", file=sys.stderr)
                rc = 127
            elapsed = time.perf_counter() - t0
        if motion:
            tl_start, tl_end = motion
        else:
            tl_start, tl_end = start_abs, start_abs + elapsed
        timeline.append({"i": i, "n": n, "skill": st["skill"], "reason": st["reason"],
                         "start": round(tl_start, 3), "end": round(tl_end, 3)})
        clock += elapsed
        log.emit("step_end", i=i, skill=st["skill"], rc=rc, elapsed_s=round(elapsed, 3), dry_run=dry,
                 motion_window=[round(x, 3) for x in motion] if motion else None, skill_success=skill_ok)
        if skill_ok is False:
            print(f"[警告] {st['skill']}: A の評価(eval_traj)が success=False(動作が基準に届かなかった可能性)", file=sys.stderr)
        if rc != 0:
            print(f"[エラー] {st['skill']} が終了コード {rc} で失敗。以降を中止します。", file=sys.stderr)
            rc_all = 5
            break
    write_status(args.status_file, state="done" if rc_all == 0 else "error", index=n, **status_base)
    if args.srt and timeline:
        write_srt(args.srt, command, timeline, args.srt_offset)
        log.emit("srt_written", path=str(args.srt), cues=len(timeline))
    return rc_all, timeline


def confirm(msg):
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{msg} 実行しますか? [y/N] ").strip().lower() in ("y", "yes", "はい")
    except EOFError:
        return False


def handle(command, skills, args, log):
    log.emit("command", text=command, model=args.model, skills=list(skills.keys()))
    try:
        result = make_plan(command, skills, args.model, backend=args.backend)
    except RuntimeError as e:
        print(f"[エラー] {e}", file=sys.stderr)
        log.emit("plan_error", error=str(e))
        write_status(args.status_file, state="error", command=command, message=str(e))
        return 4
    plan = result["plan"]
    log.emit("plan", **{k: v for k, v in result.items() if k != "raw"}, raw=result["raw"])
    print(f"指令: {command}")
    print(f"解釈: {plan['interpretation']}")
    print(f"判定: {plan['status']}   ({result['backend']}:{result['model']}, {result['latency_s']:.2f}s)")
    print(f"応答: {plan['message']}")
    for w in result["warnings"]:
        print(f"  [補正] {w}")
    for i, st in enumerate(plan["steps"], 1):
        print(f"  {i}. {st['skill']}  {st['seconds']:g}s  - {st['reason']}")
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False))
        return 0 if plan["status"] == "ok" else (2 if plan["status"] == "reject" else 3)
    if plan["status"] == "reject":
        write_status(args.status_file, state="rejected", command=command, message=plan["message"], total=0, index=0)
        log.emit("rejected", message=plan["message"])
        return 2
    if plan["status"] == "clarify":
        ok = args.yes or confirm(plan["message"])
        log.emit("clarify", message=plan["message"], approved=bool(ok))
        if not ok or not plan["steps"]:
            write_status(args.status_file, state="clarify", command=command, message=plan["message"], total=0, index=0)
            return 3
    rc, timeline = run_plan(command, result, skills, args, log)
    log.emit("done", rc=rc, steps=len(timeline), total_s=timeline[-1]["end"] if timeline else 0)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description="日本語指令 -> ローカル LLM -> スキル列 -> ACT 方策実行")
    ap.add_argument("command", nargs="*", help="日本語の指令")
    ap.add_argument("--backend", choices=["ollama", "codex", "claude"], default=os.environ.get("COMMANDER_BACKEND", "ollama"),
                    help="計画を作る LLM。ollama=ローカル(既定) / codex・claude=クラウド CLI(比較用。ホスト側で実行)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL,
                    help="ollama のモデル名。codex/claude では省略時 CLI の既定モデル")
    ap.add_argument("--skills", help="skills.json のパス(省略時: a_crane/skills.json、無ければ skills.mock.json)")
    ap.add_argument("--runner", help="run_skill.sh のパス")
    ap.add_argument("--container", default=os.environ.get("COMMANDER_CONTAINER"),
                    help="指定すると `docker exec -u ubuntu <名前>` 経由で run_skill.sh を呼ぶ(例: pai_ros2)")
    ap.add_argument("--container-env", action="append", metavar="KEY=VAL",
                    help="コンテナ内コマンドへ渡す環境変数(例: AUTO_START_SIM=0 でシムの自動起動を止める)。複数指定可")
    ap.add_argument("--container-runner", help=f"コンテナ内の run_skill.sh (既定 {CONTAINER_RUNNER})")
    ap.add_argument("--dry-run", action="store_true", help="実行コマンドを表示するだけ")
    ap.add_argument("--realtime", action="store_true", help="dry-run でもスキルの秒数だけ待つ")
    ap.add_argument("--plan-only", action="store_true", help="計画だけ出力して実行しない")
    ap.add_argument("-y", "--yes", action="store_true", help="clarify の確認を自動承認")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--log", default=str(HERE / "logs" / "commander.jsonl"), help="JSONL ログ(追記)")
    ap.add_argument("--srt", help="字幕 SRT の出力先(最後の指令のみ)")
    ap.add_argument("--srt-offset", type=float, default=0.0, help="SRT の開始オフセット秒(録画頭出し用)")
    ap.add_argument("--status-file", help="現在の状態を書く JSON(オーバーレイ用)")
    ap.add_argument("--min-seconds-ratio", type=float, default=1.0, help="実行秒数の下限 = default_seconds × この値(0 で無効)")
    ap.add_argument("--exec-timeout-margin", type=float, default=180.0, help="スキル秒数に足す待ち時間(起動・排他待ちの分)")
    args = ap.parse_args(argv)

    spath, is_real = find_skills_path(args.skills)
    skills = load_skills(spath)
    if not is_real:
        print(f"[注意] A の skills.json が無いためモック({spath.name})を使用", file=sys.stderr)
    log = RunLog(args.log, uuid.uuid4().hex[:8])
    log.emit("start", skills_path=str(spath), skills_real=is_real, model=args.model,
             runner=(f"docker:{args.container}" if args.container else str(find_runner(args.runner))), dry_run=args.dry_run)

    if args.interactive:
        rc = 0
        while True:
            try:
                line = input("指令> ").strip()
            except EOFError:
                break
            if not line or line in ("q", "quit", "exit", "終了"):
                break
            rc = handle(line, skills, args, log)
        return rc
    if not args.command:
        ap.error("指令を渡すか -i を指定してください")
    return handle(" ".join(args.command), skills, args, log)


if __name__ == "__main__":
    sys.exit(main())
