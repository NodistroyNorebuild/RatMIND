"""
RatMIND.upstream
================
上游模块：行为状态推断 (HMM) + 效用函数学习 (IRL)。

用法::

    from upstream import GaussianHMM, MaxEntIRL, BaseUpstream

    # HMM
    hmm = GaussianHMM(K=20, D=116, seed=42)
    hmm.fit(states, n_iter=50)
    posteriors = hmm.forward(states)     # (T, K)
    hmm.save_result("hmm_result.pt", states)

    # IRL
    irl = MaxEntIRL(D=116, hidden_dim=128, seed=42)
    irl.fit(demo_states=states, max_iter=200)
    utilities = irl.forward(states)      # (T, 1)
    irl.save_result("irl_result.pt", states)
"""

from .hmm import GaussianHMM, BaseUpstream
from .irl import MaxEntIRL

__all__ = [
    "BaseUpstream",
    "GaussianHMM",
    "MaxEntIRL",
]