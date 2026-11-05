# d_llm: 日本語の自然言語指令 -> ローカル LLM -> ACT スキル実行

```
指令(日本語) --> commander.py --(JSON スキーマ制約)--> Ollama(qwen2.5:3b など)
                    |                                    計画 {status, interpretation, message, steps[{skill,seconds,reason}]}
                    +--> run_skill.sh <skill> <sec> を順に実行(A の ACT 方策 / Gazebo CRANE+ V2)
                    +--> JSONL ログ / SRT 字幕 / オーバーレイ用 status.json
```

## 使い方
```bash
# 計画だけ見る / 実行コマンドを表示するだけ
python3 commander.py --plan-only "右を指してからお辞儀して"
python3 commander.py --dry-run   "こんにちはって挨拶して"
# A のコンテナ pai_ros2 で実際に動かす(WSL/Windows のホスト側から docker exec 経由)
python3 commander.py --container pai_ros2 --srt out.srt --status-file /tmp/status.json "右を指してからお辞儀して"
# コンテナ内で動かす場合(docker cp nl_commander pai_ros2:/workspace/nl_commander)
python3 commander.py "左を指して、そのあとうなずいて"      # /workspace/a_crane/run_skill.sh を直接呼ぶ
# 対話モード
python3 commander.py -i --container pai_ros2
# クラウド CLI との比較用(Windows 側 Python で実行): --backend codex | claude
```
- 判定: `ok`=実行 / `reject`=範囲外で拒否 / `clarify`=一部不可・曖昧 -> 確認メッセージ(対話で y、または `-y`)。
- 終了コード: 0 成功 2 拒否 3 確認待ち 4 LLM 失敗 5 スキル実行失敗。
- skills.json(A 管理, `/workspace/a_crane/skills.json`)の description と default_seconds を自動でプロンプトに入れる。無ければ `skills.mock.json`。`skill_hints.json` で別名・例文を補える(任意)。
- Ollama の場所: `$OLLAMA_URL` -> host.docker.internal:11434 -> 127.0.0.1:11434 -> デフォルトゲートウェイ:11434 の順に自動検出。`keep_alive` は 30 秒。

## 評価
```bash
python3 eval/run_eval.py --models qwen2.5:3b-instruct qwen2.5:3b-instruct@zeroshot llama3.1:8b   # 結果は eval/results/
python -X utf8 eval/run_eval.py --tag codex40 --models codex                                      # Windows 側
```
8b/7b 系は空き VRAM が 5500MiB 以上になるまで待つ(`--min-free-mib`, `--wait-min`)。結果の表と考察は docs/D_log.md。

LLM なしでできること(収録済みの表は上書きしない):
```bash
# 収録済み JSONL から表だけ作り直して比較する(--summary-out で出力先を変える。--results-dir で読み書き先も変えられる)
python3 eval/run_eval.py --from-results --tag mock40 --models qwen2.5:3b-instruct --summary-out /tmp/mock40.md
diff /tmp/mock40.md eval/results/mock40__summary.md
# 計画・採点ロジックと、偽 LLM + mock_run_skill.sh での逐次実行のテスト(リポジトリのルートで)
python -m pytest -q
```

## オーバーレイ(動画用)
`python overlay.py --status-file /tmp/status.json` -> http://127.0.0.1:8765/ (`?bg=black` で黒背景)。画面録画で Gazebo に重ねる。
