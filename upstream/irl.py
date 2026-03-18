"""
upstream/irl.py
===============
Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL).

Spec (from 工程流程.docx):
    - MaxEnt IRL
    - 3-layer residual MLP, 116-dim input
    - Output: utility value u(sₜ) ∈ ℝ
    - Interface: forward() / reset() / state_dict()

Architecture:
    Input sₜ (116-dim)
        → Linear(116, H) → ReLU → [residual block] ×2 → Linear(H, 1)
    Residual block: x + Linear(ReLU(Linear(x)))

Training:
    MaxEnt IRL gradient:
        ∇θ L = E_demo[∇θ r(s)] - E_policy[∇θ r(s)]
    Approximated with importance sampling for the policy expectation.
    Also supports simple feature-matching baseline.

Pure numpy implementation (no torch dependency).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .hmm import BaseUpstream

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────

IRL_DEFAULTS = {
    "hidden_dim": 128,
    "n_residual_blocks": 2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_iter": 200,
    "batch_size": 256,
    "seed": 42,
}


# ═══════════════════════════════════════════════════════════════════
#  Numpy MLP building blocks
# ═══════════════════════════════════════════════════════════════════

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def _he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He initialization for ReLU networks."""
    std = np.sqrt(2.0 / fan_in)
    return rng.standard_normal((fan_in, fan_out)) * std


class LinearLayer:
    """Single linear layer: y = xW + b."""

    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator):
        self.W = _he_init(in_dim, out_dim, rng)
        self.b = np.zeros(out_dim)
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # Cache for backward
        self._x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Returns gradient w.r.t. input."""
        x = self._x
        self.dW = x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(self.W, self.dW), (self.b, self.db)]


class ResidualBlock:
    """
    Residual block: out = x + Linear2(ReLU(Linear1(x)))

    Both linear layers: H → H.
    """

    def __init__(self, dim: int, rng: np.random.Generator):
        self.fc1 = LinearLayer(dim, dim, rng)
        self.fc2 = LinearLayer(dim, dim, rng)
        # Scale residual branch init to small values
        self.fc2.W *= 0.1
        # Cache
        self._x: Optional[np.ndarray] = None
        self._h1: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        h1 = self.fc1.forward(x)
        self._h1 = h1
        h1_relu = _relu(h1)
        h2 = self.fc2.forward(h1_relu)
        return x + h2  # residual connection

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        # grad through residual: grad_out flows to both branches
        grad_h2 = grad_out  # from the addition
        grad_h1_relu = self.fc2.backward(grad_h2)
        grad_h1 = grad_h1_relu * _relu_grad(self._h1)
        grad_x_branch = self.fc1.backward(grad_h1)
        grad_x_skip = grad_out  # skip connection
        return grad_x_skip + grad_x_branch

    def params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.fc1.params() + self.fc2.params()


class ResidualMLP:
    """
    3-layer residual MLP: Linear → ReLU → ResBlock ×N → Linear → scalar

    Input:  (batch, D_in)
    Output: (batch, 1)
    """

    def __init__(
        self,
        D_in: int = 116,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        rng: Optional[np.random.Generator] = None,
    ):
        rng = rng or np.random.default_rng()
        self.D_in = D_in
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks

        # Input projection
        self.fc_in = LinearLayer(D_in, hidden_dim, rng)

        # Residual blocks
        self.res_blocks = [ResidualBlock(hidden_dim, rng) for _ in range(n_blocks)]

        # Output projection → scalar
        self.fc_out = LinearLayer(hidden_dim, 1, rng)

        # Cache
        self._h_in: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """(batch, D_in) → (batch, 1)."""
        h = self.fc_in.forward(x)
        self._h_in = h
        h = _relu(h)
        for block in self.res_blocks:
            h = block.forward(h)
        out = self.fc_out.forward(h)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        """Backprop through entire network."""
        grad = self.fc_out.backward(grad_out)
        for block in reversed(self.res_blocks):
            grad = block.backward(grad)
        grad = grad * _relu_grad(self._h_in)
        grad = self.fc_in.backward(grad)
        return grad

    def all_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return all (param, grad) tuples for optimizer."""
        params = self.fc_in.params()
        for block in self.res_blocks:
            params.extend(block.params())
        params.extend(self.fc_out.params())
        return params

    def param_count(self) -> int:
        return sum(p.size for p, _ in self.all_params())


# ═══════════════════════════════════════════════════════════════════
#  Adam optimizer (numpy)
# ═══════════════════════════════════════════════════════════════════

class AdamOptimizer:
    """Adam optimizer for numpy parameter lists."""

    def __init__(
        self,
        params: list[tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # Moments
        self.m = [np.zeros_like(p) for p, _ in params]
        self.v = [np.zeros_like(p) for p, _ in params]

    def step(self) -> None:
        self.t += 1
        for i, (param, grad) in enumerate(self.params):
            g = grad.copy()
            if self.weight_decay > 0 and param.ndim > 1:  # Don't decay biases
                g += self.weight_decay * param

            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2

            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)

            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for _, grad in self.params:
            grad[:] = 0


# ═══════════════════════════════════════════════════════════════════
#  MaxEnt IRL
# ═══════════════════════════════════════════════════════════════════

class MaxEntIRL(BaseUpstream):
    """
    Maximum Entropy Inverse Reinforcement Learning.

    Learns a reward function r(s) = MLP(s) such that the demonstrated
    behaviour appears maximum-entropy optimal.

    Parameters
    ----------
    D : int
        State dimensionality (116).
    hidden_dim : int
        MLP hidden layer size.
    n_blocks : int
        Number of residual blocks.
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularization.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        D: int = 116,
        hidden_dim: int = 128,
        n_blocks: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: Optional[int] = None,
    ):
        self.D = D
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed

        self.rng = np.random.default_rng(seed)
        self.mlp = ResidualMLP(D_in=D, hidden_dim=hidden_dim, n_blocks=n_blocks, rng=self.rng)
        self.optimizer: Optional[AdamOptimizer] = None

        self.train_losses: list[float] = []
        self.fitted: bool = False

    # ── BaseUpstream interface ─────────────────────────────────────

    def forward(self, states: np.ndarray) -> np.ndarray:
        """
        Compute utility values u(sₜ) for each frame.

        Parameters
        ----------
        states : (T, D) or (D,)

        Returns
        -------
        utilities : (T, 1) or (1,)
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        return self.mlp.forward(states)

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.mlp = ResidualMLP(
            D_in=self.D, hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks, rng=self.rng,
        )
        self.optimizer = None
        self.train_losses = []
        self.fitted = False

    def state_dict(self) -> dict:
        layers_data = []
        for param, _ in self.mlp.all_params():
            layers_data.append(param.copy())
        return {
            "D": self.D,
            "hidden_dim": self.hidden_dim,
            "n_blocks": self.n_blocks,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "layer_params": layers_data,
            "train_losses": list(self.train_losses),
            "fitted": self.fitted,
        }

    def load_state_dict(self, d: dict) -> None:
        self.D = d["D"]
        self.hidden_dim = d["hidden_dim"]
        self.n_blocks = d["n_blocks"]
        self.lr = d["lr"]
        self.weight_decay = d["weight_decay"]
        self.seed = d["seed"]
        self.train_losses = list(d.get("train_losses", []))
        self.fitted = d.get("fitted", True)

        # Rebuild MLP and load weights
        self.rng = np.random.default_rng(0)  # Dummy, will be overwritten
        self.mlp = ResidualMLP(
            D_in=self.D, hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks, rng=self.rng,
        )
        for (param, _), saved in zip(self.mlp.all_params(), d["layer_params"]):
            if hasattr(saved, "numpy"):
                saved = saved.numpy()
            param[:] = saved

    # ── Training ───────────────────────────────────────────────────

    def fit(
        self,
        demo_states: np.ndarray,
        background_states: Optional[np.ndarray] = None,
        max_iter: int = 200,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list[float]:
        """
        Train MaxEnt IRL reward function.

        MaxEnt IRL gradient:
            ∇θ L = (1/N_demo) Σ ∇θ r(s_demo) - (1/N_bg) Σ w(s) ∇θ r(s_bg)

        where w(s) = exp(r(s)) / Z is the importance weight, approximating
        the policy expectation via background samples.

        If background_states is None, we generate them by adding Gaussian
        noise to the demo states (simple baseline strategy).

        Parameters
        ----------
        demo_states : (T, D) demonstrated state trajectories.
        background_states : (T_bg, D) or None.
        max_iter : int
        batch_size : int
        verbose : bool

        Returns
        -------
        losses : list[float]

        Note
        ----
        Currently, I use a fixed noise perturbation distribution as the
        background.
        The approximate policy expectation E_π[∇r(s)] is equivalent to the
        contrastive learning (NCE) objective, rather than the strict
        MaxEnt IRL.
        After the interface of the downstream behavioral model is
        determined, the background_states should be replaced with policy
        rollout samples. At that time, only this function needs to be
        modified, and the network structure remains unchanged.
        """
        T, D = demo_states.shape
        assert D == self.D

        if background_states is None:
            # Generate background by perturbing demo
            noise_scale = demo_states.std(axis=0) * 0.5
            background_states = demo_states + self.rng.standard_normal(demo_states.shape) * noise_scale
            logger.info("Generated background states: %s (noise σ=%.2f)",
                        background_states.shape, noise_scale.mean())

        T_bg = background_states.shape[0]

        # Initialize optimizer
        self.optimizer = AdamOptimizer(
            self.mlp.all_params(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.train_losses = []
        logger.info("MaxEnt IRL training: T_demo=%d, T_bg=%d, max_iter=%d, batch=%d",
                     T, T_bg, max_iter, batch_size)

        for it in range(max_iter):
            # ── Sample mini-batches ──
            demo_idx = self.rng.choice(T, size=min(batch_size, T), replace=False)
            bg_idx = self.rng.choice(T_bg, size=min(batch_size, T_bg), replace=False)

            s_demo = demo_states[demo_idx]
            s_bg = background_states[bg_idx]

            # ── Forward: compute rewards ──
            r_demo = self.mlp.forward(s_demo)  # (B, 1)
            r_bg = self.mlp.forward(s_bg)      # (B, 1)

            # ── MaxEnt loss: maximize demo reward, minimize log-partition ──
            # L = -mean(r_demo) + log(mean(exp(r_bg)))
            r_bg_max = r_bg.max()
            log_Z = r_bg_max + np.log(np.exp(r_bg - r_bg_max).mean() + 1e-12)
            loss = -r_demo.mean() + log_Z

            self.train_losses.append(float(loss))

            # ── Backward: demo gradient ──
            self.optimizer.zero_grad()

            # Grad w.r.t. demo: -1/B for each sample
            grad_r_demo = np.full_like(r_demo, -1.0 / len(demo_idx))
            self.mlp.backward(grad_r_demo)

            # Save demo gradients
            demo_grads = [(g.copy()) for _, g in self.mlp.all_params()]

            # Grad w.r.t. background: importance weights
            self.optimizer.zero_grad()
            weights = np.exp(r_bg - r_bg_max)
            weights /= weights.sum() + 1e-12
            grad_r_bg = weights  # (B, 1)
            self.mlp.forward(s_bg)  # Re-cache
            self.mlp.backward(grad_r_bg)

            # Combine gradients: demo + background
            for (_, grad), dg in zip(self.mlp.all_params(), demo_grads):
                grad += dg

            # ── Update ──
            self.optimizer.step()

            if verbose and (it % 50 == 0 or it == max_iter - 1):
                r_demo_mean = r_demo.mean()
                r_bg_mean = r_bg.mean()
                logger.info("  IRL iter %3d/%d  loss=%.4f  r_demo=%.3f  r_bg=%.3f",
                            it + 1, max_iter, loss, r_demo_mean, r_bg_mean)

        self.fitted = True
        logger.info("MaxEnt IRL done: %d iters, final loss=%.4f, params=%d",
                     max_iter, self.train_losses[-1], self.mlp.param_count())
        return self.train_losses

    # ── Analysis ───────────────────────────────────────────────────

    def utility_stats(self, states: np.ndarray) -> dict:
        """Compute summary statistics of utility values."""
        u = self.forward(states).flatten()
        return {
            "mean": float(u.mean()),
            "std": float(u.std()),
            "min": float(u.min()),
            "max": float(u.max()),
            "p5": float(np.percentile(u, 5)),
            "p95": float(np.percentile(u, 95)),
        }

    # ── Save / Load ────────────────────────────────────────────────

    def save_result(
        self,
        path: Union[str, Path],
        states: np.ndarray,
    ) -> Path:
        """
        Save IRL result as irl_result.pt.

        Contents:
            utilities    : (T, 1) float32  — u(sₜ) for all frames
            params       : state_dict
            meta         : config + training info
        """
        path = Path(path)
        utilities = self.forward(states).astype(np.float32)
        ustats = self.utility_stats(states)

        result = {
            "utilities": utilities,
            "params": self.state_dict(),
            "meta": {
                "D": self.D,
                "hidden_dim": self.hidden_dim,
                "n_blocks": self.n_blocks,
                "T": states.shape[0],
                "param_count": self.mlp.param_count(),
                "n_iter": len(self.train_losses),
                "final_loss": self.train_losses[-1] if self.train_losses else None,
                "utility_stats": ustats,
            },
        }

        try:
            import torch
            result_t = result.copy()
            result_t["utilities"] = torch.from_numpy(utilities)
            torch.save(result_t, path)
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(result, f, protocol=4)

        logger.info("Saved IRL result → %s (T=%d, u=[%.3f, %.3f])",
                    path, states.shape[0], ustats["min"], ustats["max"])
        return path

    @staticmethod
    def load_result(path: Union[str, Path]) -> dict:
        """Load IRL result from .pt file."""
        path = Path(path)
        try:
            import torch
            data = torch.load(path, map_location="cpu", weights_only=False)
            if hasattr(data["utilities"], "numpy"):
                data["utilities"] = data["utilities"].numpy()
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
        logger.info("Loaded IRL result ← %s (T=%d)", path, data["meta"]["T"])
        return data