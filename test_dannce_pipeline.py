"""
test_dannce_pipeline.py
========================
端到端测试：mock DANNCE → dannce_loader → STAC IK → snapshot (.pt)

放在 RatMIND/ 根目录，运行：
    python test_dannce_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from data import (
    # ── dannce_loader ──
    DANNCESequence,
    load_dannce,
    load_multiple,
    N_KEYPOINTS,
    STATE_DIM,
    POS_DIM,
    SPEED_DIM,
    COM_VEL_DIM,
    # ── mock_generator_dannce ──
    generate_mock_session,
    generate_state_sequence,
    generate_com_trajectory,
    validate_and_report,
    BehaviourState,
    BEHAVIOURS,
    STATE_NAMES,
    TRANSITION_MATRIX,
    JOINT_NAMES,
    N_KP,
    REST_POSE,
    # ── stac_loader ──
    STACResult,
    run_stac,
    keypoints_to_joint_angles,
    batch_keypoints_to_joint_angles,
    N_JOINTS_DOF,
    JOINT_NAMES_DOF,
    JOINT_LIMITS,
    BONE_LENGTHS,
    # ── mock_generator_stac ──
    generate_stac_snapshot,
    validate_snapshot,
)


def heading(msg: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {msg}")
    print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
#  Part A: DANNCE 层测试 (1-10)
# ═══════════════════════════════════════════════════════════════════════════

def test_constants():
    heading("1. 常量一致性")

    assert N_KP == N_KEYPOINTS == 23
    assert len(JOINT_NAMES) == N_KP
    assert REST_POSE.shape == (N_KP, 3)
    assert len(BEHAVIOURS) == len(STATE_NAMES) == TRANSITION_MATRIX.shape[0]
    assert STATE_DIM == 116
    assert POS_DIM + SPEED_DIM + COM_VEL_DIM == STATE_DIM
    assert N_JOINTS_DOF == 38
    assert len(JOINT_NAMES_DOF) == N_JOINTS_DOF

    print(f"  N_KP={N_KP}, STATE_DIM={STATE_DIM}, N_JOINTS_DOF={N_JOINTS_DOF}")
    print(f"  BEHAVIOURS: {STATE_NAMES}")
    print("  ✅ 通过")


def test_mock_generation():
    heading("2. generate_mock_session 基本功能")

    kp, labels = generate_mock_session(T=1000, fps=50.0, seed=42)

    assert kp.shape == (1000, N_KP, 3), f"Shape: {kp.shape}"
    assert labels.shape == (1000,), f"Labels shape: {labels.shape}"
    assert np.isfinite(kp).all(), "Contains NaN/Inf"
    assert labels.min() >= 0
    assert labels.max() < len(STATE_NAMES)

    print(f"  keypoints : {kp.shape}")
    print(f"  labels    : {labels.shape}, unique={np.unique(labels).tolist()}")
    print("  ✅ 通过")


def test_state_sequence():
    heading("3. generate_state_sequence 行为序列")

    rng = np.random.default_rng(0)
    labels = generate_state_sequence(T=5000, fps=50.0, rng=rng)

    assert labels.shape == (5000,)
    assert labels[0] == 0, "应从 rest (0) 开始"
    assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0)

    unique, counts = np.unique(labels, return_counts=True)
    dist = {STATE_NAMES[u]: f"{c / 5000 * 100:.1f}%" for u, c in zip(unique, counts)}
    print(f"  分布: {dist}")
    print("  ✅ 通过")


def test_com_trajectory():
    heading("4. COM 轨迹")

    rng = np.random.default_rng(10)
    labels = generate_state_sequence(T=2000, fps=50.0, rng=rng)
    com_xy = generate_com_trajectory(T=2000, labels=labels, fps=50.0,
                                     arena_radius=300.0, rng=rng)

    assert com_xy.shape == (2000, 2)
    dist = np.linalg.norm(com_xy, axis=1)
    assert dist.max() < 350.0, f"COM 超出 arena: max={dist.max():.1f}"

    print(f"  Arena dist max: {dist.max():.1f} mm")
    print("  ✅ 通过")


def test_validate():
    heading("5. validate_and_report")

    kp, labels = generate_mock_session(T=1000, fps=50.0, seed=99)
    stats = validate_and_report(kp, labels, fps=50.0)

    assert "com_speed_mm_s" in stats
    assert "behaviour_distribution" in stats
    assert stats["frames"] == 1000
    print("  ✅ 通过")


def test_save_load_roundtrip():
    heading("6. .npy 保存 → load_dannce 闭环")

    kp, _ = generate_mock_session(T=500, fps=50.0, seed=7)

    with tempfile.TemporaryDirectory() as tmpdir:
        npy_path = Path(tmpdir) / "test.npy"
        np.save(npy_path, kp.astype(np.float32))
        seq = load_dannce(npy_path, fps=50.0)
        diff = np.abs(kp.astype(np.float32) - seq.raw_keypoints.astype(np.float32))
        assert diff.max() < 1e-4

    print(f"  Max diff: {diff.max():.2e}")
    print("  ✅ 通过")


def test_state_vector():
    heading("7. 116维状态向量")

    kp, _ = generate_mock_session(T=500, fps=50.0, seed=33)
    seq = DANNCESequence(raw_keypoints=kp, fps=50.0)
    s = seq.states
    T = seq.T

    assert s.shape == (T, STATE_DIM)

    pos  = s[:, :POS_DIM]
    spd  = s[:, POS_DIM:POS_DIM + SPEED_DIM]

    cvel = s[:, -COM_VEL_DIM:]
    pos_3d = pos.reshape(T, N_KEYPOINTS, 3)
    assert np.abs(pos_3d.mean(axis=1)).max() < 1e-8

    assert spd.min() >= 0
    assert cvel[0, 0] == 0.0
    assert cvel.min() >= 0

    print(f"  {POS_DIM}+{SPEED_DIM}+{COM_VEL_DIM} = {STATE_DIM}")
    print("  ✅ 通过")


def test_windowing():
    heading("8. 窗口采样")

    kp, _ = generate_mock_session(T=500, fps=50.0, seed=0)
    seq = DANNCESequence(raw_keypoints=kp, fps=50.0)

    win = seq.get_window(start=10, length=64)
    assert win.shape == (64, STATE_DIM)

    win_tail = seq.get_window(start=seq.T - 10, length=64)
    assert win_tail.shape[0] == 10

    windows = list(seq.iter_windows(length=64, stride=32))
    expected = (seq.T - 64) // 32 + 1
    assert len(windows) == expected

    print(f"  iter(64/32): {len(windows)} windows / {seq.T} frames")
    print("  ✅ 通过")


def test_multi_session():
    heading("9. 多 Session 批量")

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(3):
            kp, _ = generate_mock_session(T=250, fps=50.0, seed=100 + i)
            p = Path(tmpdir) / f"s{i:03d}.npy"
            np.save(p, kp.astype(np.float32))
            paths.append(p)

        sequences = load_multiple(paths, fps=50.0)
        all_states = np.concatenate([seq.states for seq in sequences], axis=0)
        assert all_states.shape == (750, STATE_DIM)

    print(f"  拼接: {all_states.shape}")
    print("  ✅ 通过")


def test_reproducibility():
    heading("10. 可复现性")

    kp1, lb1 = generate_mock_session(T=300, fps=50.0, seed=777)
    kp2, lb2 = generate_mock_session(T=300, fps=50.0, seed=777)
    assert np.array_equal(kp1, kp2)
    assert np.array_equal(lb1, lb2)

    kp3, _ = generate_mock_session(T=300, fps=50.0, seed=888)
    assert not np.array_equal(kp1, kp3)

    print("  同 seed 一致，异 seed 不同")
    print("  ✅ 通过")


# ═══════════════════════════════════════════════════════════════════════════
#  Part B: STAC 层测试 (11-17)
# ═══════════════════════════════════════════════════════════════════════════

def test_single_frame_ik():
    heading("11. 单帧 IK (keypoints_to_joint_angles)")

    kp, _ = generate_mock_session(T=10, fps=50.0, seed=50)
    q = keypoints_to_joint_angles(kp[0])

    assert q.shape == (N_JOINTS_DOF,), f"Shape: {q.shape}"
    assert np.isfinite(q).all()

    for j, name in enumerate(JOINT_NAMES_DOF):
        lo, hi = JOINT_LIMITS[name]
        assert q[j] >= lo - 1e-6, f"{name}={q[j]:.4f} < {lo}"
        assert q[j] <= hi + 1e-6, f"{name}={q[j]:.4f} > {hi}"

    print(f"  q: {q.shape}, range [{q.min():.3f}, {q.max():.3f}]")
    print("  ✅ 通过")


def test_batch_ik():
    heading("12. 批量 IK (batch_keypoints_to_joint_angles)")

    kp, _ = generate_mock_session(T=500, fps=50.0, seed=60)
    angles = batch_keypoints_to_joint_angles(kp, smooth_sigma=1.0)

    assert angles.shape == (500, N_JOINTS_DOF)
    assert np.isfinite(angles).all()

    for j, name in enumerate(JOINT_NAMES_DOF):
        lo, hi = JOINT_LIMITS[name]
        assert angles[:, j].min() >= lo - 1e-6
        assert angles[:, j].max() <= hi + 1e-6

    print(f"  Shape: {angles.shape}, range [{angles.min():.3f}, {angles.max():.3f}] rad")
    print("  ✅ 通过")


def test_stac_result():
    heading("13. run_stac → STACResult")

    kp, _ = generate_mock_session(T=500, fps=50.0, seed=70)
    seq = DANNCESequence(raw_keypoints=kp, fps=50.0)
    result = run_stac(keypoints=kp, states=seq.states, fps=50.0)

    assert result.joint_angles.shape == (500, N_JOINTS_DOF)
    assert result.states.shape == (500, STATE_DIM)
    assert result.T == 500

    vel = result.joint_velocities
    assert vel.shape == (500, N_JOINTS_DOF)
    assert vel[0].sum() == 0.0

    root_tz = result.get_joint_by_name("root_tz")
    assert root_tz.shape == (500,)

    print(f"  angles: {result.joint_angles.shape}")
    print(f"  vel max: {np.abs(vel).max():.1f} rad/s")
    print("  ✅ 通过")


def test_joint_angle_groups():
    heading("14. 关节角分组统计")

    kp, _ = generate_mock_session(T=1000, fps=50.0, seed=80)
    seq = DANNCESequence(raw_keypoints=kp, fps=50.0)
    result = run_stac(keypoints=kp, states=seq.states, fps=50.0)
    q = result.joint_angles

    groups = {
        "Root trans": (0, 3), "Root rot": (3, 6),
        "Spine": (6, 12), "Head": (12, 15),
        "L-forelimb": (15, 20), "R-forelimb": (20, 25),
        "L-hindlimb": (25, 30), "R-hindlimb": (30, 35),
        "Tail": (35, 38),
    }

    for gname, (i0, i1) in groups.items():
        g = q[:, i0:i1]
        assert np.isfinite(g).all()
        print(f"  {gname:14s}: mean={g.mean():+.3f}  std={g.std():.3f}  "
              f"[{g.min():.3f}, {g.max():.3f}]")

    l_fl = q[:, 15:20]
    r_fl = q[:, 20:25]
    mean_diff = np.abs(l_fl.mean() - r_fl.mean())
    assert mean_diff < 0.5, f"左右前肢不对称: diff={mean_diff:.3f}"

    print(f"  左右前肢均值差: {mean_diff:.4f}")
    print("  ✅ 通过")


def test_snapshot_save_load():
    heading("15. Snapshot 保存/加载 (.pt)")

    kp, _ = generate_mock_session(T=500, fps=50.0, seed=90)
    seq = DANNCESequence(raw_keypoints=kp, fps=50.0)
    result = run_stac(keypoints=kp, states=seq.states, fps=50.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        pt_path = Path(tmpdir) / "test_snapshot.pt"
        result.save_snapshot(pt_path)

        assert pt_path.exists()
        size_kb = pt_path.stat().st_size / 1024
        print(f"  文件大小: {size_kb:.1f} KB")

        snap = STACResult.load_snapshot(pt_path)
        assert np.asarray(snap["states"]).shape == (500, STATE_DIM)
        assert np.asarray(snap["joint_angles"]).shape == (500, N_JOINTS_DOF)
        assert snap["meta"]["T"] == 500
        assert snap["meta"]["fps"] == 50.0

        s_orig = result.states.astype(np.float32)
        s_load = np.asarray(snap["states"]).astype(np.float32)
        assert np.allclose(s_orig, s_load, atol=1e-5)

    print("  ✅ 通过")


def test_generate_stac_snapshot():
    heading("16. generate_stac_snapshot 一键生成")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "full_snap.pt")
        snap = generate_stac_snapshot(T=500, fps=50.0, seed=42, out_path=out_path)

        assert np.asarray(snap["states"]).shape == (500, STATE_DIM)
        assert np.asarray(snap["joint_angles"]).shape == (500, N_JOINTS_DOF)
        assert snap["meta"]["T"] == 500
        assert Path(out_path).exists()

    print(f"  states: {np.asarray(snap['states']).shape}")
    print(f"  angles: {np.asarray(snap['joint_angles']).shape}")
    print("  ✅ 通过")


def test_snapshot_validation():
    heading("17. validate_snapshot")

    snap = generate_stac_snapshot(T=500, fps=50.0, seed=55)
    validate_snapshot(snap)
    print("  ✅ 通过")


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  RatMIND — DANNCE + STAC Pipeline 端到端测试")
    print("=" * 60)

    tests = [
        # Part A: DANNCE (1-10)
        test_constants,
        test_mock_generation,
        test_state_sequence,
        test_com_trajectory,
        test_validate,
        test_save_load_roundtrip,
        test_state_vector,
        test_windowing,
        test_multi_session,
        test_reproducibility,
        # Part B: STAC (11-17)
        test_single_frame_ik,
        test_batch_ik,
        test_stac_result,
        test_joint_angle_groups,
        test_snapshot_save_load,
        test_generate_stac_snapshot,
        test_snapshot_validation,
    ]

    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    heading("Summary")
    print(f"  Passed : {passed}/{len(tests)}")
    if failed:
        print(f"  Failed : {failed}/{len(tests)}")
        sys.exit(1)
    else:
        print(f"  🎉 ALL {len(tests)} TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()