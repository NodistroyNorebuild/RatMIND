"""
upstream/hmm.py
===============
Hidden Markov Model with Gaussian emissions — Baum-Welch (EM).

Spec (from project doc):
    - Baum-Welch EM on 116-dim state vectors
    - K hidden states: 20 (debug) / 100 (production)
    - Output: posterior zₜ ∈ ℝᴷ (soft assignment per frame)
    - Interface: forward() / reset() / state_dict()

Emission model: diagonal-covariance Gaussian per hidden state.
    p(xₜ | sₜ=k) = N(xₜ; μₖ, diag(σₖ²))

Usage:
    from upstream.hmm import GaussianHMM

    hmm = GaussianHMM(K=20, D=116, seed=42)
    hmm.fit(states, n_iter=50)           # Baum-Welch EM
    posteriors = hmm.forward(states)     # (T, K)
    hmm.save("hmm_result.pt")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Abstract base class (文档要求: 含抽象基类)
# ═══════════════════════════════════════════════════════════════════

class BaseUpstream(ABC):
    """Abstract base class for all upstream modules."""

    @abstractmethod
    def forward(self, states: np.ndarray) -> np.ndarray:
        """Process state sequence, return output."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state / parameters to initial."""
        ...

    @abstractmethod
    def state_dict(self) -> dict:
        """Return serializable parameter dict."""
        ...

    @abstractmethod
    def load_state_dict(self, d: dict) -> None:
        """Load parameters from dict."""
        ...


# ═══════════════════════════════════════════════════════════════════
#  Gaussian HMM
# ═══════════════════════════════════════════════════════════════════

class GaussianHMM(BaseUpstream):
    """
    Hidden Markov Model with diagonal Gaussian emissions.

    Parameters
    ----------
    K : int
        Number of hidden states (20 debug / 100 production).
    D : int
        Observation dimensionality (116 for our state vector).
    seed : int or None
        Random seed for initialization.
    reg : float
        Regularization added to diagonal covariances to prevent collapse.
    """

    def __init__(
        self,
        K: int = 20,
        D: int = 116,
        seed: Optional[int] = None,
        reg: float = 1e-3,
    ):
        self.K = K
        self.D = D
        self.reg = reg
        self.rng = np.random.default_rng(seed)

        # ── Parameters ──
        self.pi: np.ndarray = np.zeros(K)         # log initial probs
        self.A: np.ndarray = np.zeros((K, K))      # log transition matrix
        self.mu: np.ndarray = np.zeros((K, D))     # emission means
        self.log_var: np.ndarray = np.zeros((K, D))  # log emission variances

        # ── Training history ──
        self.log_likelihoods: list[float] = []
        self.fitted: bool = False

        self.reset()

    # ── BaseUpstream interface ─────────────────────────────────────

    def forward(self, states: np.ndarray) -> np.ndarray:
        """
        Compute posterior probabilities p(sₜ=k | x₁:T) for each frame.

        Parameters
        ----------
        states : np.ndarray, shape (T, D)

        Returns
        -------
        posteriors : np.ndarray, shape (T, K)
            Each row sums to 1. This is zₜ ∈ ℝᴷ.
        """
        T = states.shape[0]
        log_B = self._log_emission(states)  # (T, K)

        # Forward pass
        log_alpha = self._forward_pass(log_B)

        # Backward pass
        log_beta = self._backward_pass(log_B)

        # Posterior: γₜ(k) = p(sₜ=k | x₁:T)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

        return np.exp(log_gamma)  # (T, K)

    def reset(self) -> None:
        """Reinitialize all parameters randomly."""
        K, D = self.K, self.D

        # Uniform initial distribution
        self.pi = np.full(K, -np.log(K))

        # Slightly sticky transition matrix (diagonal dominant)
        A_raw = self.rng.dirichlet(np.ones(K) * 0.5 + np.eye(K).sum(axis=0) * 5, size=K)
        self.A = np.log(A_raw + 1e-12)

        # Emission means: K-means++ style from unit Gaussian
        self.mu = self.rng.standard_normal((K, D)) * 0.1
        self.log_var = np.zeros((K, D))  # Start with unit variance

        self.log_likelihoods = []
        self.fitted = False

    def state_dict(self) -> dict:
        """Serialize all parameters."""
        return {
            "K": self.K,
            "D": self.D,
            "reg": self.reg,
            "pi": self.pi.copy(),
            "A": self.A.copy(),
            "mu": self.mu.copy(),
            "log_var": self.log_var.copy(),
            "log_likelihoods": list(self.log_likelihoods),
            "fitted": self.fitted,
        }

    def load_state_dict(self, d: dict) -> None:
        """Load parameters from dict."""
        self.K = d["K"]
        self.D = d["D"]
        self.reg = d.get("reg", 1e-3)
        self.pi = d["pi"].copy()
        self.A = d["A"].copy()
        self.mu = d["mu"].copy()
        self.log_var = d["log_var"].copy()
        self.log_likelihoods = list(d.get("log_likelihoods", []))
        self.fitted = d.get("fitted", True)

    # ── Initialization ─────────────────────────────────────────────

    def initialize_from_data(self, states: np.ndarray) -> None:
        """
        Smart initialization using data statistics + K-means.

        Parameters
        ----------
        states : np.ndarray, shape (T, D)
        """
        T, D = states.shape
        assert D == self.D

        # Global stats for variance init
        global_var = np.var(states, axis=0) + self.reg
        self.log_var = np.tile(np.log(global_var), (self.K, 1))

        # K-means++ initialization for means
        self.mu = self._kmeans_plus_plus(states, self.K)

        # Run a few K-means iterations to refine
        for _ in range(10):
            # Assign
            dists = np.array([
                np.sum((states - self.mu[k]) ** 2, axis=1)
                for k in range(self.K)
            ]).T  # (T, K)
            labels = np.argmin(dists, axis=1)

            # Update
            for k in range(self.K):
                mask = labels == k
                if mask.sum() > 1:
                    self.mu[k] = states[mask].mean(axis=0)
                    self.log_var[k] = np.log(states[mask].var(axis=0) + self.reg)

        # Transition matrix from label sequence
        A_counts = np.ones((self.K, self.K)) * 0.1  # Laplace smoothing
        for t in range(T - 1):
            A_counts[labels[t], labels[t + 1]] += 1
        A_probs = A_counts / A_counts.sum(axis=1, keepdims=True)
        self.A = np.log(A_probs + 1e-12)

        # Initial distribution from first frames
        pi_counts = np.ones(self.K) * 0.1
        pi_counts[labels[0]] += 1
        self.pi = np.log(pi_counts / pi_counts.sum())

        logger.info("Initialized from data: K=%d, T=%d, D=%d", self.K, T, D)

    def _kmeans_plus_plus(self, X: np.ndarray, K: int) -> np.ndarray:
        """K-means++ center initialization."""
        T, D = X.shape
        centers = np.zeros((K, D))
        centers[0] = X[self.rng.integers(T)]

        for k in range(1, K):
            dists = np.min([
                np.sum((X - centers[j]) ** 2, axis=1) for j in range(k)
            ], axis=0)
            probs = dists / dists.sum()
            centers[k] = X[self.rng.choice(T, p=probs)]

        return centers

    # ── Core Baum-Welch ────────────────────────────────────────────

    def fit(
        self,
        states: np.ndarray,
        n_iter: int = 50,
        tol: float = 1e-4,
        init_from_data: bool = True,
        verbose: bool = True,
    ) -> list[float]:
        """
        Fit HMM via Baum-Welch (EM) algorithm.

        Parameters
        ----------
        states : np.ndarray, shape (T, D)
        n_iter : int
            Maximum EM iterations.
        tol : float
            Convergence tolerance on log-likelihood change.
        init_from_data : bool
            If True, run K-means initialization first.
        verbose : bool
            Print progress.

        Returns
        -------
        log_likelihoods : list[float]
            Training log-likelihood per iteration.
        """
        T, D = states.shape
        assert D == self.D, f"Expected D={self.D}, got {D}"

        if init_from_data:
            self.initialize_from_data(states)

        self.log_likelihoods = []

        for it in range(n_iter):
            # ── E-step ──
            log_B = self._log_emission(states)  # (T, K)
            log_alpha = self._forward_pass(log_B)
            log_beta = self._backward_pass(log_B)

            # Log-likelihood
            ll = logsumexp(log_alpha[-1])
            self.log_likelihoods.append(float(ll))

            # Posterior γₜ(k)
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)  # (T, K)

            # Transition posterior ξₜ(i,j) — aggregated
            log_xi_sum = self._compute_xi_sum(log_alpha, log_beta, log_B)

            # ── M-step ──
            # Initial distribution
            self.pi = log_gamma[0] - logsumexp(log_gamma[0])

            # Transition matrix
            self.A = log_xi_sum - logsumexp(log_xi_sum, axis=1, keepdims=True)

            # Emission parameters
            gamma_sum = gamma.sum(axis=0) + 1e-12  # (K,)
            self.mu = (gamma.T @ states) / gamma_sum[:, None]  # (K, D)
            diff = states[:, None, :] - self.mu[None, :, :]  # (T, K, D)
            self.log_var = np.log(
                (gamma[:, :, None] * diff ** 2).sum(axis=0) / gamma_sum[:, None] + self.reg
            )  # (K, D)


            if verbose and (it % 10 == 0 or it == n_iter - 1):
                logger.info("  EM iter %3d/%d  LL=%.2f", it + 1, n_iter, ll)

            # Convergence check
            if len(self.log_likelihoods) >= 2:
                delta = abs(self.log_likelihoods[-1] - self.log_likelihoods[-2])
                if delta < tol:
                    logger.info("  Converged at iter %d (δLL=%.6f)", it + 1, delta)
                    break

        self.fitted = True
        logger.info("Baum-Welch done: %d iters, final LL=%.2f",
                    len(self.log_likelihoods), self.log_likelihoods[-1])
        return self.log_likelihoods

    # ── Internal helpers ───────────────────────────────────────────

    def _log_emission(self, states: np.ndarray) -> np.ndarray:
        """
        Compute log p(xₜ | sₜ=k) for all t and k.

        Returns shape (T, K, D).
        """
        diff = states[:, None, :] - self.mu[None, :, :]  # (T, K, D)
        var = np.exp(self.log_var)  # (K, D)
        mahal = (diff ** 2 / var[None, :, :]).sum(axis=2)  # (T, K)
        log_det = self.log_var.sum(axis=1)  # (K,)
        return -0.5 * (self.D * np.log(2 * np.pi) + log_det[None, :] + mahal)

    def _forward_pass(self, log_B: np.ndarray) -> np.ndarray:
        """
        Forward algorithm in log space.

        Returns log_alpha: (T, K)
        """
        T, K = log_B.shape
        log_alpha = np.full((T, K), -np.inf)

        # t=0
        log_alpha[0] = self.pi + log_B[0]

        # t=1..T-1
        for t in range(1, T):
            log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + self.A, axis=0) + log_B[t]

        return log_alpha

    def _backward_pass(self, log_B: np.ndarray) -> np.ndarray:
        """
        Backward algorithm in log space.

        Returns log_beta: (T, K)
        """
        T, K = log_B.shape
        log_beta = np.full((T, K), -np.inf)

        # t=T-1
        log_beta[-1] = 0.0

        # t=T-2..0
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(self.A + log_B[t + 1] + log_beta[t + 1], axis=1)

        return log_beta

    def _compute_xi_sum(self, log_alpha, log_beta, log_B):
        log_xi = (
                log_alpha[:-1, :, None]  # (T-1, K, 1)
                + self.A[None, :, :]  # (1,   K, K)
                + log_B[1:, None, :]  # (T-1, 1, K)
                + log_beta[1:, None, :]  # (T-1, 1, K)
        )  # → (T-1, K, K)
        return logsumexp(log_xi, axis=0)  # (K, K)

    # ── Decoding ───────────────────────────────────────────────────

    def viterbi(self, states: np.ndarray) -> np.ndarray:
        """
        Viterbi decoding: most likely hidden state sequence.

        Parameters
        ----------
        states : (T, D)

        Returns
        -------
        path : (T,) int array of state indices
        """
        T = states.shape[0]
        log_B = self._log_emission(states)
        K = self.K

        # Forward (max instead of sum)
        V = np.full((T, K), -np.inf)
        ptr = np.zeros((T, K), dtype=np.int32)

        V[0] = self.pi + log_B[0]
        for t in range(1, T):
            for k in range(K):
                scores = V[t - 1] + self.A[:, k]
                ptr[t, k] = np.argmax(scores)
                V[t, k] = scores[ptr[t, k]] + log_B[t, k]

        # Backtrack
        path = np.zeros(T, dtype=np.int32)
        path[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]

        return path

    # ── Analysis ───────────────────────────────────────────────────

    def posterior_entropy(self, posteriors: np.ndarray) -> np.ndarray:
        """
        Per-frame entropy of posterior distribution.

        Parameters
        ----------
        posteriors : (T, K)

        Returns
        -------
        entropy : (T,) — in nats
        """
        safe_p = np.clip(posteriors, 1e-12, 1.0)
        return -np.sum(safe_p * np.log(safe_p), axis=1)

    def state_usage(self, posteriors: np.ndarray) -> np.ndarray:
        """Average posterior mass per state. Shape (K,)."""
        return posteriors.mean(axis=0)

    # ── Save / Load ────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> Path:
        """
        Save HMM result as hmm_result.pt

        Contents:
            params     : state_dict()
            posteriors : None (call forward() to recompute, or pass in)
            config     : K, D, reg
        """
        path = Path(path)
        try:
            import torch
            torch.save(self.state_dict(), path)
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self.state_dict(), f, protocol=4)
        logger.info("Saved HMM → %s (K=%d, D=%d)", path, self.K, self.D)
        return path

    def load(self, path: Union[str, Path]) -> None:
        """Load HMM from file (supports both params-only and full result files)."""
        path = Path(path)
        try:
            import torch
            d = torch.load(path, map_location="cpu", weights_only=False)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                d = pickle.load(f)
        # If this is a full result file, extract the params sub-dict
        if "params" in d and "pi" not in d:
            d = d["params"]
        # Convert any torch tensors to numpy
        for key in ("pi", "A", "mu", "log_var"):
            if hasattr(d[key], "numpy"):
                d[key] = d[key].numpy()
        self.load_state_dict(d)
        logger.info("Loaded HMM ← %s (K=%d, D=%d)", path, self.K, self.D)

    def save_result(
        self,
        path: Union[str, Path],
        states: np.ndarray,
        posteriors: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Save complete HMM result file (hmm_result.pt).

        Contents:
            posteriors   : (T, K) float32
            viterbi_path : (T,) int32
            params       : state_dict
            meta         : K, D, T, log_likelihoods
        """
        path = Path(path)

        if posteriors is None:
            posteriors = self.forward(states)

        viterbi_path = self.viterbi(states)
        entropy = self.posterior_entropy(posteriors)

        result = {
            "posteriors": posteriors.astype(np.float32),
            "viterbi_path": viterbi_path,
            "entropy": entropy.astype(np.float32),
            "params": self.state_dict(),
            "meta": {
                "K": self.K,
                "D": self.D,
                "T": states.shape[0],
                "log_likelihoods": self.log_likelihoods,
                "mean_entropy": float(entropy.mean()),
                "active_states": int((self.state_usage(posteriors) > 0.01).sum()),
            },
        }

        try:
            import torch
            # Convert numpy arrays to torch tensors for .pt format
            result_torch = {
                "posteriors": torch.from_numpy(result["posteriors"]),
                "viterbi_path": torch.from_numpy(result["viterbi_path"]),
                "entropy": torch.from_numpy(result["entropy"]),
                "params": result["params"],
                "meta": result["meta"],
            }
            torch.save(result_torch, path)
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(result, f, protocol=4)

        logger.info("Saved HMM result → %s (T=%d, K=%d, active=%d)",
                    path, states.shape[0], self.K,
                    result["meta"]["active_states"])
        return path


# ═══════════════════════════════════════════════════════════════════
#  Quick test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Synthetic test
    np.random.seed(0)
    T, D, K = 500, 10, 3

    # Generate data from a known HMM
    true_labels = np.repeat([0, 1, 2, 0, 1], T // 5)
    centers = np.random.randn(K, D) * 3
    X = centers[true_labels] + np.random.randn(T, D) * 0.5

    hmm = GaussianHMM(K=K, D=D, seed=42)
    lls = hmm.fit(X, n_iter=30, verbose=True)

    posteriors = hmm.forward(X)
    path = hmm.viterbi(X)

    print(f"\nPosteriors shape: {posteriors.shape}")
    print(f"Viterbi path unique: {np.unique(path)}")
    print(f"Mean entropy: {hmm.posterior_entropy(posteriors).mean():.4f}")
    print(f"State usage: {hmm.state_usage(posteriors)}")

    # Accuracy vs true labels (up to permutation)
    from itertools import permutations
    best_acc = 0
    for perm in permutations(range(K)):
        mapped = np.array([perm[p] for p in path])
        acc = (mapped == true_labels).mean()
        best_acc = max(best_acc, acc)
    print(f"Best Viterbi accuracy (permuted): {best_acc:.1%}")
    print("✅ HMM sanity check passed.")