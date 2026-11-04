# 日本語の指令でロボットアームの学習済みスキルを順に動かす

Physical AI 応用1講座 最終課題。アカウント：**Rick0218**

CRANE+ V2の6つの動作を、スクリプトで自動生成したお手本からACT（Action Chunking with Transformers）で学習した。日本語の指令はローカルLLMが動作の列に変換し、Gazebo上で順に実行する。数値はすべて [docs/](docs/) の実験ログと各 `results/` に記録した実測値である。

- 提出動画（2分10秒、等速。指令デモはLLMの応答待ちのみカット）：[YouTube](https://youtu.be/JBdugjrzdls) ／ リポジトリ内の [submission_ad.mp4](crane_act/videos/submission_ad.mp4)
- 説明資料：[docs/report.md](docs/report.md)

```text
日本語指令
  └─ nl_commander/commander.py
       ├─ Ollama LLM → JSON schema制約の計画
       ├─ status / スキル名 / 秒数の検証（ok / reject / clarify）
       └─ crane_act/run_skill.sh を順に呼ぶ
            └─ 常駐サーバまたは単発実行 → スキル別ACT → ROS 2 / Gazebo / CRANE+ V2
                 ↑ シム時刻・phase観測・lag augmentation・std clamp・chunk実行
```

## 収録内容

| 場所 | 内容 |
|---|---|
| [crane_act](crane_act/) | 動作の学習：デモ生成、変換、学習、6個のACT方策、推論・評価・動画 |
| [nl_commander](nl_commander/) | 言語による計画：計画、逐次実行、40件評価、字幕、実行ログ |
| [説明資料](docs/report.md) | 概要・目的・工夫・結果・考察 |
| [学習の作業ログ](docs/A_log.md) / [言語計画の作業ログ](docs/D_log.md) | 一次資料。旧ディレクトリ名（a_crane / d_llm）は当時の記録として残る |
| [Colab結果](docs/colab_llm_result.md) | A100での40件評価の集計 |
| [来歴・ライセンス](docs/provenance.md) | 上流そのまま／改変／自作の区分と確認できた根拠 |
| [tests](tests/) / [CI](.github/workflows/ci.yml) | ハードウェア不要のテスト（pytest）と、それを回すGitHub Actions |

## どこで何が動くか

| 内容 | 必要なもの | CIで確認 |
|---|---|---|
| 学習済みスキルの実行、データ収集・学習、軌道評価、動画の録画・組み立て | ROS 2 Humble・Gazebo・CRANE+パッケージ・pai_training（pixi、torch / LeRobot）。学習・推論はGPU推奨 | ×（構文チェックのみ） |
| 日本語指令の計画・40件評価の測定 | Ollama（またはCodex / Claude CLI）とLLM | ×（偽のLLMで配線のみ） |
| 計画の検証・採点ロジック、収録済み結果からの表の再集計、お手本軌道の生成と成功判定、同梱チェックポイントの整合 | Python 3.10以上、pytest、numpy | ○ |

### ハードウェアなしで確認する

```bash
pip install -r requirements-dev.txt
python -m compileall -q nl_commander crane_act tests
python -m pytest -q
```

テストは収録済みの結果を読むだけで書き換えない。主な確認内容：

- `nl_commander/eval/results/` の4つの表（mock40 / ollama40 / codex40 / real40）が、同じフォルダのJSONLから1文字違わず再集計できる
- 収録済みJSONLの各行を現在の採点関数で採点し直すと、記録された判定と一致する
- 偽のLLM（`commander.ollama_chat` を差し替え）と [mock_run_skill.sh](nl_commander/mock_run_skill.sh) で、指令 → 計画 → 逐次実行 → JSONLログ・字幕が最後まで通る
- `crane_act/scripts/skills.py` のお手本軌道が `eval_traj.py` の成功基準を満たし、同梱の6チェックポイントが欠けていない（torchは使わない）

表の再集計だけを手で行う例（出力先を変えるので収録済みの表は上書きしない）：

```bash
cd nl_commander
python3 eval/run_eval.py --from-results --tag real40 --skills ../crane_act/skills.json \
    --models qwen2.5:3b-instruct qwen2.5:7b-instruct llama3.1:8b --summary-out /tmp/real40__summary.md
diff /tmp/real40__summary.md eval/results/real40__summary.md
```

## 実行環境とパス

以下は再現用の手順である。必要な依存ソフト・モデルは別途用意する。**この節以降のコマンドはROS 2・Gazebo・GPU・Ollamaが必要で、CIでは実行・検証していない**（作者の環境で使った手順を記載している）。

- 動作の学習・実行：ROS 2 Humble、Ignition Gazebo、CRANE+パッケージ、physicalai_teleop、matsuolab/pai_trainingとそのpixi環境が必要。作業時は `airobotbook/ros2-desktop-ai-robot-book-humble` イメージを使用した。pai_trainingのコミットやLeRobotの厳密な版は記録から確定できない。
- 言語による計画：Python標準ライブラリとOllama（0.40.13）。主にqwen2.5:3b-instructを使用。`OLLAMA_URL` で接続先を指定する。
- `crane_act` のコードはコンテナ内の固定マウント `/workspace/a_crane` を使うので、**ホストの `crane_act` をそこにマウント**する。`nl_commander` は `/workspace/nl_commander` に置くと下の例に合う。
- `crane_act/env.sh` は自身から `A_DIR` を求める。講義環境の場所は `PAI_TRAINING_ROOT`（既定 `~/pai_training`）、Docker実行ファイルは `COMMANDER_DOCKER` またはPATHで指定する。ROSワークスペースは同ファイルの `~/airobot_ws` と `~/pai_ws` を自分の環境に合わせる。
- `crane_act/configs` のデータ・出力パスはpai_trainingから見た `../dataset` / `../checkpoints`。生データは収録していないので、再学習には収集から必要。

## 学習済みスキルを動かす

（要ROS 2 / Gazebo。CIでは未検証）ROS環境を用意し、上記のマウントを行ったLinuxコンテナ内で：

```bash
SIM=demo bash /workspace/a_crane/scripts/sim_ctl.sh start true
bash /workspace/a_crane/scripts/skill_server.sh start
bash /workspace/a_crane/run_skill.sh bow
bash /workspace/a_crane/run_skill.sh wave
```

データ収集・学習はヘッドレスで行う（250エピソードの収集には約40分かかる）。

```bash
bash /workspace/a_crane/scripts/sim_ctl.sh start false
SIM_TIME=1 bash /workspace/a_crane/scripts/collect.sh bow 50 0
bash /workspace/a_crane/scripts/build_and_train.sh bow 50 phase 2500
```

詳しいスクリプトは [crane_act/README.md](crane_act/README.md)、スキル設定は [skills.json](crane_act/skills.json)。常駐サーバの終了は `skill_server.sh stop`。プロセスの終了コードだけで成功を判断せず、`success=` と軌道評価も確認する。

## 日本語の指令で動かす

（要Ollama。実行には上記のコンテナも必要。CIでは未検証）

```bash
cd /workspace/nl_commander
export OLLAMA_URL=http://host.docker.internal:11434
python3 commander.py --plan-only "右を指してからお辞儀して"
python3 commander.py --srt /tmp/sequence.srt "右を指してからお辞儀して"
```

ホストから起動済みコンテナへ実行を委ねる場合は `--container pai_ros2` を追加する。`clarify` は確認待ち、`reject` は実行しない。`--dry-run` は実行を省略するが、計画にはLLMを使う。

40件の計画評価（ロボットの成功率とは別の指標）：

```bash
python3 eval/run_eval.py --skills ../a_crane/skills.json --tag review40 --models qwen2.5:3b-instruct
```

詳しくは [nl_commander/README.md](nl_commander/README.md)、[テストセット](nl_commander/eval/testset_mock.jsonl)、[実スキル40件結果](nl_commander/eval/results/real40__summary.md)。

提出動画は `crane_act/scripts/make_submission_video.py` で録画済みの動画から組み立てた（コンテナ内のffmpegとNoto Sans CJKを使用）。

## 主な結果

| 実験 | 結果 | 解釈上の注意 |
|---|---|---|
| 講義モデルの実行系比較 | 講義の推論 0/3 → 改良した実行器 5/5 | 同一モデル。学習法だけの効果ではない |
| 最終6スキル | 各5/5、計30/30 | 同じ直立初期姿勢での少数試行。nodの閾値を事後変更 |
| 開始姿勢をj2/j3/j4各+0.2 rad | bow / wave 各0/3 | 分布外姿勢への弱さ |
| 計画の40件評価、ローカル8GB | 3B 33/40、7B 36/40、Llama 8B 36/40 | 平均1.99 / 3.54 / 4.18秒 |
| 同、Colab A100 | 14B / 32B 各37/40 | 平均0.94 / 2.19秒。異なる実行環境 |
| 同、Codex CLI | 40/40、平均6.74秒 | クラウド。この40件に限った値 |
| 常駐サーバ経由の2動作連続実行 | 3通り×5回、各ステップ15/15 | waveのデモをv4に直した後 |
| 1動作の所要時間 | 約40秒 → 約10秒 | モデルを常駐させて起動待ちをなくした |

出典と全表は [説明資料](docs/report.md)。

## 動画・モデルの収録

ACTの最終チェックポイント6個（各17,030,228 bytes）を通常のgitで収録した。どれもGitHubの単一ファイル警告（50MB）未満で、cloneだけで同じ方策を確認できる。

- [提出動画（2分10秒）](crane_act/videos/submission_ad.mp4)（[YouTube](https://youtu.be/JBdugjrzdls)）：日本語指令のデモ → 学習した6動作。説明は [説明資料](docs/report.md) にまとめた
- [日本語指令のデモ（36.8秒）](crane_act/videos/demo_nl_commander.mp4)：指令 → LLMの判定・計画 → ACTスキルの逐次実行。範囲外は拒否、曖昧なものは確認
- [学習した動作のまとめ（128.9秒）](crane_act/videos/montage_a_crane.mp4)：講義手順と改良版の比較、6スキル

rosbag、生データセット、途中のモデルは含めない。

## 上流コード・ライセンス・クレジット

**リポジトリ全体に適用するLICENSEは置いていない。** physicalai_teleop、LeRobotについてLICENSE原文と適用条件を確認できなかったためで、自由な再利用を許諾する表明ではない。

- **上流をもとに改変**：`crane_act/launch/sim.launch.py`（crane_plus_gazebo、RT Corporation、ヘッダにApache-2.0）、`crane_act/scripts/train_act.py`（pai_training由来・ヘッダにApache-2.0）、`convert_ext.py`（rosbag2lerobot拡張）、`skill_runner.py`（infer.py拡張）、`configs/*.yaml`（講義設定を変更）。
- **本課題の自作・拡張（AI支援を含む）**：自動デモ・スキル定義・学習高速化・時刻整合・常駐実行・評価・録画、commander・計画評価・字幕。`debug_policy_save.py` は講義のdebug_policyを無改変で呼ぶラッパで、講義スクリプト本体は同梱していない。
- 詳細とファイル別分類は [来歴](docs/provenance.md)。

講義の基盤（matsuolab/pai_training、physicalai_teleop）、CRANE+ / crane_plus_gazebo、LeRobot ACT、Ollama / Qwen / Llamaに謝意を示す。実装・デバッグにClaude、整理・解説草稿にCodexを使用した。
