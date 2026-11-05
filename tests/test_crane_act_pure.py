"""crane_act のうち ROS 2 / torch / GPU なしで動く部分のテスト。

skills.py(デモ軌道の生成)と eval_traj.py(成功判定)は numpy だけで動く。
skill_runner.py / skill_server.py などは rclpy・torch・lerobot が必要なので、ここでは扱わない
(構文は CI の compileall で確認する)。
"""
import csv
import json
import struct

import pytest

np = pytest.importorskip("numpy")

from conftest import CRANE  # noqa: E402

import eval_traj  # noqa: E402
import record_nl_demo  # noqa: E402
import skills as S  # noqa: E402

NAMES = ["bow", "nod", "wave", "point_left", "point_right", "grip"]


def write_runner_csv(path, traj):
    """skills.py の軌道(意味順 j1 j2 j3 j4 g)を、skill_runner.py の CSV(列順 j1 j2 j4 j3 g)で書く。"""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"s_{n}" for n in ("j1", "j2", "j4", "j3", "g")])
        for i, q in enumerate(traj):
            w.writerow([round(i * 0.05, 3), q[0], q[1], q[3], q[2], q[4]])


@pytest.mark.parametrize("skill", NAMES)
def test_trajectory_starts_and_ends_at_home(skill):
    rng = np.random.default_rng(0)
    kf = S.make_keyframes(skill, rng)
    traj = S.sample_trajectory(kf, dt=0.05)
    assert traj.shape[1] == 5
    assert len(traj) == sum(max(1, int(round(d / 0.05))) for d, _ in kf)
    np.testing.assert_allclose(traj[-1], S.HOME)
    # min-jerk 補間なので、最初のサンプルは HOME のごく近く
    assert np.max(np.abs(traj[0] - S.HOME)) < 0.01
    g = traj[:, 4]
    assert g.min() >= S.GRIPPER_OPEN - 1e-9 and g.max() <= S.GRIPPER_CLOSE + 1e-9


def test_make_keyframes_is_seeded_and_rejects_unknown():
    a = S.sample_trajectory(S.make_keyframes("wave", np.random.default_rng(3)))
    b = S.sample_trajectory(S.make_keyframes("wave", np.random.default_rng(3)))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        S.make_keyframes("jump", np.random.default_rng(0))


def test_min_jerk_endpoints():
    assert S.min_jerk(0.0, 1.0, 0.0) == 0.0
    assert S.min_jerk(0.0, 1.0, 1.0) == 1.0
    assert S.min_jerk(0.0, 1.0, 0.5) == pytest.approx(0.5)
    assert S.min_jerk(0.0, 1.0, 2.0) == 1.0  # clip


@pytest.mark.parametrize("skill", NAMES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_scripted_demo_passes_success_criterion(skill, seed, tmp_path):
    """理想的に追従した場合のお手本軌道は、eval_traj の成功基準を満たす。"""
    traj = S.sample_trajectory(S.make_keyframes(skill, np.random.default_rng(seed)))
    p = tmp_path / "run.csv"
    write_runner_csv(p, traj)
    ok, motion, home_err, detail = eval_traj.evaluate(skill, p)
    assert ok and motion and home_err < 1e-9, detail


@pytest.mark.parametrize("skill", NAMES)
def test_staying_at_home_fails(skill, tmp_path):
    p = tmp_path / "home.csv"
    write_runner_csv(p, np.tile(S.HOME, (100, 1)))
    ok, motion, home_err, _ = eval_traj.evaluate(skill, p)
    assert not ok and not motion and home_err == 0


def test_not_returning_home_fails(tmp_path):
    traj = S.sample_trajectory(S.make_keyframes("bow", np.random.default_rng(0)))[:-25]
    p = tmp_path / "run.csv"
    write_runner_csv(p, traj)
    ok, motion, home_err, _ = eval_traj.evaluate("bow", p)
    assert motion and not ok and home_err >= eval_traj.TOL_HOME


def test_crossings():
    assert eval_traj.crossings([0, 0.3, 0.1, 0.3, 0.0], 0.25, 0.15) == 2
    assert eval_traj.crossings([0, 0.3, 0.2, 0.3], 0.25, 0.15) == 1  # 谷が浅いと 1 回と数える


def test_descriptions_match_skills_json():
    cfg = json.load(open(CRANE / "skills.json", encoding="utf-8"))
    assert set(cfg) == set(S.DESCRIPTIONS)
    for k, v in cfg.items():
        assert v["description"] == S.DESCRIPTIONS[k]


def test_record_nl_demo_srt_time():
    assert record_nl_demo.srt_time(-1) == "00:00:00,000"
    assert record_nl_demo.srt_time(3723.456) == "01:02:03,456"
    # 以前は小数部だけを丸めていたため "00:00:01,1000" になっていた
    assert record_nl_demo.srt_time(1.9996) == "00:00:02,000"
    assert record_nl_demo.srt_time(59.9999) == "00:01:00,000"


# ------------------------------------------------------------------ shipped checkpoints (no torch needed)
def read_safetensors_header(path):
    """(ヘッダ長, ヘッダ JSON)。safetensors の先頭 8 バイトはヘッダ長(little endian u64)。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


@pytest.mark.parametrize("skill", NAMES)
def test_checkpoint_files_are_complete(skill):
    """skills.json の model_path(コンテナ内 /workspace/a_crane/...)に対応する同梱チェックポイントを確認する。"""
    cfg = json.load(open(CRANE / "skills.json", encoding="utf-8"))[skill]
    assert cfg["model_path"] == f"/workspace/a_crane/checkpoints/{skill}"
    d = CRANE / "checkpoints" / skill
    for name in ("config.json", "a_crane_meta.json", "model.safetensors"):
        assert (d / name).is_file(), name
    st = d / "model.safetensors"
    assert st.stat().st_size == 17_030_228  # README / report に記載のサイズ
    n_header, header = read_safetensors_header(st)
    size = st.stat().st_size
    ends = [v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"]
    assert max(ends) == size - 8 - n_header  # 途中で切れていない(Git LFS のポインタでもない)
    meta = json.load(open(d / "a_crane_meta.json", encoding="utf-8"))
    assert meta["env_mode"] in ("phase", "phasedq", "state")
    conf = json.load(open(d / "config.json", encoding="utf-8"))
    assert conf.get("chunk_size") == meta["chunk_size"]
