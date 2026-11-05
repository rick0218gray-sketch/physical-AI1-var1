"""ハードウェア不要のテスト用の共通設定。

ROS 2 / Gazebo / GPU / Ollama は使わない。LLM は monkeypatch した偽物(fake)で置き換える。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NL = ROOT / "nl_commander"
CRANE = ROOT / "crane_act"

for p in (NL, NL / "eval", CRANE / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
