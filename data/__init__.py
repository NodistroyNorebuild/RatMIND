"""
RatMIND.data
============
数据加载、骨架注册、模拟数据生成的统一接口。

用法::

    # ── 加载 + STAC 全流程 ──
    from data import load_dannce, run_stac
    seq = load_dannce("keypoints.npy")
    result = run_stac(seq.raw_keypoints, seq.states)
    result.save_snapshot("snapshot.pt")

    # ── 一键生成 mock snapshot ──
    from data import generate_stac_snapshot
    snap = generate_stac_snapshot(T=5000, seed=42, out_path="snapshot.pt")

    # ── 下游直接读 snapshot ──
    import torch
    snap = torch.load("snapshot.pt")
    states = snap["states"]           # (T, 93)
    joint_angles = snap["joint_angles"]  # (T, 38)
"""

# ═══════════════════════════════════════════════════════════════════
#  dannce_loader
# ═══════════════════════════════════════════════════════════════════
from .dannce_loader import (
    DANNCESequence,
    load_dannce,
    load_multiple,
    N_KEYPOINTS,
    STATE_DIM,
    POS_DIM,
    SPEED_DIM,
    COM_VEL_DIM,
    DEFAULT_KEYPOINT_NAMES,
)

# ═══════════════════════════════════════════════════════════════════
#  stac_loader
# ═══════════════════════════════════════════════════════════════════
from .stac_loader import (
    STACResult,
    run_stac,
    keypoints_to_joint_angles,
    batch_keypoints_to_joint_angles,
    N_JOINTS_DOF,
    JOINT_NAMES_DOF,
    JOINT_LIMITS,
    BONE_LENGTHS,
)

# ═══════════════════════════════════════════════════════════════════
#  mock_generator_dannce  (注意：你的文件名)
# ═══════════════════════════════════════════════════════════════════
from .mock_generator_dannce import (
    BehaviourState,
    generate_state_sequence,
    generate_com_trajectory,
    synthesize_keypoints,
    generate_mock_session,
    validate_and_report,
    JOINT_NAMES,
    N_KP,
    REST_POSE,
    BEHAVIOURS,
    STATE_NAMES,
    TRANSITION_MATRIX,
)

# ═══════════════════════════════════════════════════════════════════
#  mock_generator_stac
# ═══════════════════════════════════════════════════════════════════
from .mock_generator_stac import (
    generate_stac_snapshot,
    validate_snapshot,
)

__all__ = [
    # dannce_loader
    "DANNCESequence", "load_dannce", "load_multiple",
    "N_KEYPOINTS", "STATE_DIM", "POS_DIM", "SPEED_DIM", "COM_VEL_DIM",
    "DEFAULT_KEYPOINT_NAMES",
    # stac_loader
    "STACResult", "run_stac",
    "keypoints_to_joint_angles", "batch_keypoints_to_joint_angles",
    "N_JOINTS_DOF", "JOINT_NAMES_DOF", "JOINT_LIMITS", "BONE_LENGTHS",
    # mock_generator_dannce
    "BehaviourState", "generate_state_sequence", "generate_com_trajectory",
    "synthesize_keypoints", "generate_mock_session", "validate_and_report",
    "JOINT_NAMES", "N_KP", "REST_POSE", "BEHAVIOURS", "STATE_NAMES", "TRANSITION_MATRIX",
    # mock_generator_stac
    "generate_stac_snapshot", "validate_snapshot",
]