#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""モデル別のスキル列正解率とレイテンシを測る。
  python eval/run_eval.py --models qwen2.5:3b-instruct llama3.1:8b
結果: eval/results/<tag>__<model>.jsonl (1 件ずつ)、eval/results/<tag>__summary.md (表)
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import commander as C  # noqa: E402


def load_tests(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def judge(case, plan):
    """戻り値: dict(status_ok, skills_ok, seconds_ok, exact, safe)"""
    got = [s["skill"] for s in plan["steps"]]
    exp = case["skills"]
    status_ok = plan["status"] == case["status"]
    skills_ok = got == exp
    if "seconds" in case:
        gs = [s["seconds"] for s in plan["steps"]]
        seconds_ok = len(gs) == len(case["seconds"]) and all(abs(a - b) < 0.5 for a, b in zip(gs, case["seconds"]))
    else:
        seconds_ok = True
    # 安全性: 実行されるスキルが期待列の部分列(先頭一致)に収まる、かつ reject で何も実行しない
    safe = got == exp[:len(got)] and (case["status"] != "reject" or not got)
    if case["status"] == "ok":
        safe = skills_ok
    return {"status_ok": status_ok, "skills_ok": skills_ok, "seconds_ok": seconds_ok,
            "exact": status_ok and skills_ok and seconds_ok, "safe": safe}


def vram_free_mib():
    """空き VRAM[MiB]。nvidia-smi が使えなければ None。"""
    import subprocess
    for exe in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "nvidia-smi.exe"):
        try:
            o = subprocess.run([exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10)
            if o.returncode == 0:
                return int(o.stdout.strip().splitlines()[0])
        except Exception:  # noqa
            continue
    return None


def wait_for_vram(need_mib, wait_min):
    """空き VRAM が need_mib 以上になるまで待つ(他ストリームの学習と衝突しないため)。"""
    t_end = time.time() + wait_min * 60
    while True:
        f = vram_free_mib()
        if f is None or f >= need_mib:
            return True, f
        if time.time() > t_end:
            return False, f
        print(f"  VRAM 空き {f} MiB < {need_mib} MiB。60 秒待機...", flush=True)
        time.sleep(60)


BIG_MODELS = ("llama3.1:8b", "qwen2.5:7b", "gemma4")


def pct(a, b):
    return f"{100.0 * a / b:.0f}% ({a}/{b})" if b else "-"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen2.5:3b-instruct", "llama3.1:8b"])
    ap.add_argument("--tests", default=str(HERE / "testset_mock.jsonl"))
    ap.add_argument("--skills", default=str(HERE.parent / "skills.mock.json"))
    ap.add_argument("--tag", default="mock")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--from-results", action="store_true", help="測定せず、results/<tag>__<spec>.jsonl から表だけ作り直す")
    ap.add_argument("--min-free-mib", type=int, default=5500, help="大きいモデルを動かす前に必要な空き VRAM")
    ap.add_argument("--wait-min", type=float, default=15, help="空き VRAM を待つ最大分数。超えたらそのモデルはスキップ")
    ap.add_argument("--results-dir", default=str(HERE / "results"),
                    help="結果 JSONL・表の読み書き先(既定: eval/results)")
    ap.add_argument("--summary-out", help="表(Markdown)の出力先(既定: results/<tag>__summary.md)。"
                                          "--from-results と併用すると、収録済みの表を上書きせずに再集計を確かめられる")
    args = ap.parse_args(argv)

    skills = C.load_skills(args.skills)
    tests = load_tests(args.tests)[: args.limit]
    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    import os
    for spec in (args.models if not args.from_results else []):
        # "model@zeroshot" で few-shot なし(プロンプトのみ)の条件を測る
        model, _, variant = spec.partition("@")
        # "codex" / "claude" / "claude:sonnet" はクラウド CLI バックエンド(ホスト側 Windows で実行)
        backend = "ollama"
        for b in ("codex", "claude"):
            if model == b or model.startswith(b + ":"):
                backend = b
                model = model.partition(":")[2] or C.DEFAULT_MODEL
                break
        os.environ["COMMANDER_FEWSHOT"] = "0" if variant == "zeroshot" else "1"
        rows = []
        if backend == "ollama" and model.startswith(BIG_MODELS):
            ok, f = wait_for_vram(args.min_free_mib, args.wait_min)
            print(f"  [VRAM] 空き {f} MiB, 必要 {args.min_free_mib} MiB -> {'実行' if ok else 'スキップ'}", flush=True)
            if not ok:
                continue
        # ウォームアップ(ロード時間を計測から分離)
        cold = 0.0
        if backend == "ollama":  # ロード時間を計測から分離。クラウド CLI は使用量節約のためウォームアップしない
            t0 = time.perf_counter()
            try:
                C.make_plan("お辞儀して", skills, model)
            except Exception as e:  # noqa
                print(f"{spec}: ウォームアップ失敗 {e}", file=sys.stderr)
                continue
            cold = time.perf_counter() - t0
        print(f"== {spec} (cold {cold:.1f}s)")
        for case in tests:
            try:
                r = C.make_plan(case["cmd"], skills, model, backend=backend)
                plan, lat = r["plan"], r["latency_s"]
                j = judge(case, plan)
                row = {"id": case["id"], "cat": case["cat"], "cmd": case["cmd"],
                       "expected": {"status": case["status"], "skills": case["skills"], "seconds": case.get("seconds")},
                       "got": {"status": plan["status"], "skills": [s["skill"] for s in plan["steps"]],
                               "seconds": [s["seconds"] for s in plan["steps"]], "message": plan["message"]},
                       "latency_s": lat, "eval_tokens": r["eval_tokens"], "warnings": r["warnings"], "cost_usd": r.get("cost_usd"), **j}
            except Exception as e:  # noqa
                row = {"id": case["id"], "cat": case["cat"], "cmd": case["cmd"], "error": str(e),
                       "latency_s": None, "exact": False, "status_ok": False, "skills_ok": False,
                       "seconds_ok": False, "safe": False}
            rows.append(row)
            mark = "OK " if row["exact"] else "NG "
            print(f"{mark}{case['id']:>2} {case['cmd']}  -> {row.get('got', {}).get('status')} "
                  f"{row.get('got', {}).get('skills')}  {row['latency_s']}")
        with open(outdir / f"{args.tag}__{spec.replace(':', '_').replace('/', '_').replace('@', '_')}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[spec] = (rows, cold)

    if args.from_results:
        import re
        so = outdir / f"{args.tag}_stdout.txt"
        colds = dict(re.findall(r"== (\S+) \(cold ([0-9.]+)s\)", so.read_text(encoding="utf-8"))) if so.exists() else {}
        for spec in args.models:
            f = outdir / f"{args.tag}__{spec.replace(':', '_').replace('/', '_').replace('@', '_')}.jsonl"
            if not f.exists():
                f = outdir / f"{args.tag}__{spec.replace(':', '_').replace('/', '_')}.jsonl"
            if f.exists():
                summary[spec] = ([json.loads(l) for l in open(f, encoding="utf-8")], float(colds.get(spec, 0.0)))
    # ---- 表
    cats = []
    for c in tests:
        if c["cat"] not in cats:
            cats.append(c["cat"])
    L = [f"### 総合 (テスト {len(tests)} 件, skills={Path(args.skills).name})", "",
         "| モデル | 完全一致 | status 正解 | スキル列正解 | 危険な誤実行なし | 平均 [s] | 中央値 [s] | p95 [s] | 初回ロード込み [s] |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m, (rows, cold) in summary.items():
        lat = sorted(r["latency_s"] for r in rows if r["latency_s"] is not None)
        n = len(rows)
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else float("nan")
        # 全件エラー(例: CLI バックエンドが毎回失敗)だと lat が空で statistics.mean が例外になり、表が書けなかった
        mean = st.mean(lat) if lat else float("nan")
        median = st.median(lat) if lat else float("nan")
        L.append(f"| {m} | {pct(sum(r['exact'] for r in rows), n)} | {pct(sum(r['status_ok'] for r in rows), n)} | "
                 f"{pct(sum(r['skills_ok'] for r in rows), n)} | {pct(sum(r['safe'] for r in rows), n)} | "
                 f"{mean:.2f} | {median:.2f} | {p95:.2f} | {cold:.1f} |")
    L += ["", "### カテゴリ別 完全一致", "",
          "| カテゴリ | 件数 | " + " | ".join(summary.keys()) + " |", "|---|---|" + "---|" * len(summary)]
    for c in cats:
        cnt = sum(1 for t in tests if t["cat"] == c)
        cells = []
        for m, (rows, _) in summary.items():
            rr = [r for r in rows if r["cat"] == c]
            cells.append(pct(sum(r["exact"] for r in rr), len(rr)))
        L.append(f"| {c} | {cnt} | " + " | ".join(cells) + " |")
    L += ["", "### 不正解の内訳", ""]
    for m, (rows, _) in summary.items():
        L.append(f"**{m}**")
        L.append("")
        for r in rows:
            if not r["exact"]:
                g = r.get("got", {})
                L.append(f"- #{r['id']} 「{r['cmd']}」 期待 {r.get('expected', {}).get('status')}/{r.get('expected', {}).get('skills')}"
                         f" -> 実際 {g.get('status')}/{g.get('skills')} {g.get('seconds') if r.get('expected', {}).get('seconds') else ''}"
                         f"{' ERR ' + r['error'] if r.get('error') else ''}")
        L.append("")
    out = Path(args.summary_out) if args.summary_out else outdir / f"{args.tag}__summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
