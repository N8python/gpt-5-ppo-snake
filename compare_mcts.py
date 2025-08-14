#!/usr/bin/env python3
"""
compare_mcts.py

Loads checkpoints/best_avg.safetensors and compares the model's performance on Snake
using (1) Greedy decoding (argmax policy) vs (2) MCTS.

MCTS modes:
  - "alphago":   policy priors + short policy rollouts + value bootstrap
  - "alphazero": policy priors + NO rollouts; leaves evaluated ONLY by the value head
  - "value_only": UNIFORM priors + NO rollouts; leaves evaluated ONLY by the value head

Requires:
  - main_ppo.py in the same directory (for SnakeEnv, ActorCritic, grid_to_onehot3_np)
  - MLX (mlx)
  - tqdm

Examples:
  # AlphaZero-style (policy priors, value-only leaves)
  python compare_mcts.py --checkpoint checkpoints/best_avg.safetensors --num-games 128 --mcts-mode alphazero --sims 64

  # AlphaGo-style (policy priors + rollouts + value)
  python compare_mcts.py --checkpoint checkpoints/best_avg.safetensors --num-games 128 --mcts-mode alphago --sims 64 --rollout-depth 8

  # Value-only (uniform priors; value-only leaves)
  python compare_mcts.py --checkpoint checkpoints/best_avg.safetensors --num-games 128 --mcts-mode value_only --sims 64
"""

import argparse
import math
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import random
from tqdm import tqdm

import mlx.core as mx
import mlx.nn as nn

from main_ppo import SnakeEnv, ActorCritic, grid_to_onehot3_np

mx.set_default_device(mx.cpu)

# -----------------------------
# --- Utility / NN helpers  ---
# -----------------------------

def softmax_np(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    s = e.sum(axis=axis, keepdims=True)
    return e / (s + eps)


def policy_value(net: nn.Module, grid: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Given an integer grid observation [H, W], returns (policy_probs[4], value).
    """
    enc = grid_to_onehot3_np(grid)  # uint8 [H*W*3]
    obs_mx = mx.array(enc[None, :], dtype=mx.float32)
    logits, value = net(obs_mx)  # logits: [1,4], value: [1]
    logits_np = np.array(logits.tolist(), dtype=np.float32)[0]
    probs = softmax_np(logits_np, axis=-1)
    v = float(np.array(value.tolist(), dtype=np.float32)[0])
    return probs, v


def value_only(net: nn.Module, grid: np.ndarray) -> float:
    """
    Return just the scalar value estimate for a single grid state.
    """
    enc = grid_to_onehot3_np(grid)
    obs_mx = mx.array(enc[None, :], dtype=mx.float32)
    _, value = net(obs_mx)
    v = float(np.array(value.tolist(), dtype=np.float32)[0])
    return v


def greedy_action(net: nn.Module, grid: np.ndarray) -> int:
    enc = grid_to_onehot3_np(grid)
    obs_mx = mx.array(enc[None, :], dtype=mx.float32)
    logits, _ = net(obs_mx)
    logits_np = np.array(logits.tolist(), dtype=np.float32)[0]
    return int(np.argmax(logits_np))


def clone_env(env: SnakeEnv) -> SnakeEnv:
    """
    Create a deep-ish copy of a SnakeEnv
    """
    new_env = SnakeEnv(width=env.width, height=env.height)
    # Copy core fields
    new_env.snake = list(env.snake)
    new_env.direction = tuple(env.direction)
    new_env.apple = tuple(env.apple)
    new_env.done = bool(env.done)
    new_env.score = int(env.score)
    # Copy RNG state if available
    #try:
    #    new_env.rng.setstate(env.rng.getstate())
    #except Exception:
    #    pass
    new_env.reseed(random.randrange(2**31-1))
    return new_env


# -----------------
# --- MCTS code ---
# -----------------

class Node:
    __slots__ = ("state_key", "P", "children", "N", "W", "Q", "expanded")
    def __init__(self, state_key: bytes, priors: Optional[np.ndarray] = None):
        self.state_key = state_key            # bytes key of encoded onehot (uint8).tobytes()
        self.P: Optional[np.ndarray] = priors # action priors [4], filled on expand
        self.children: Dict[int, "Node"] = {}
        self.N: int = 0    # total visits
        self.W: float = 0  # total value
        self.Q: float = 0  # mean value
        self.expanded: bool = False


class MCTS:
    def __init__(
        self,
        net: nn.Module,
        gamma: float = 0.99,
        c_puct: float = 1.5,
        num_simulations: int = 64,
        rollout_depth: int = 8,
        use_greedy_rollout: bool = False,
        use_policy_priors: bool = True,
        use_policy_rollout: bool = True,
    ):
        self.net = net
        self.gamma = gamma
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.use_greedy_rollout = use_greedy_rollout
        self.use_policy_priors = use_policy_priors
        self.use_policy_rollout = use_policy_rollout
        # cache stores value ALWAYS; priors only when use_policy_priors is True
        self.cache_val: Dict[bytes, float] = {}
        self.cache_pi: Dict[bytes, np.ndarray] = {}

    # ---- helpers ----

    def _state_key(self, grid: np.ndarray) -> bytes:
        onehot = grid_to_onehot3_np(grid).astype(np.uint8)
        return onehot.tobytes()

    def _get_value(self, grid: np.ndarray) -> float:
        k = self._state_key(grid)
        if k in self.cache_val:
            return self.cache_val[k]
        v = value_only(self.net, grid)
        self.cache_val[k] = v
        return v

    def _get_priors(self, grid: np.ndarray) -> np.ndarray:
        if not self.use_policy_priors:
            return np.full((4,), 1.0 / 4.0, dtype=np.float32)
        k = self._state_key(grid)
        if k in self.cache_pi:
            return self.cache_pi[k]
        pi, _ = policy_value(self.net, grid)
        self.cache_pi[k] = pi
        return pi

    def _policy_rollout(self, env: SnakeEnv, max_depth: int) -> float:
        """
        Run a short rollout using the policy head to pick actions (sample or greedy),
        accumulate discounted rewards, and bootstrap with value at the last state.
        If disabled or depth <= 0, returns just the value estimate.
        """
        if (not self.use_policy_rollout) or max_depth <= 0:
            return self._get_value(env.get_observation())

        total = 0.0
        discount = 1.0
        for _ in range(max_depth):
            grid = env.get_observation()
            pi = self._get_priors(grid)
            if self.use_greedy_rollout:
                a = int(np.argmax(pi))
            else:
                a = int(np.random.choice(4, p=pi / (pi.sum() + 1e-8)))
            _, r, done = env.step(a)
            total += discount * float(r)
            if done:
                return total  # terminal; no bootstrap
            discount *= self.gamma
        # bootstrap with value of last state
        v_tail = self._get_value(env.get_observation())
        return total + discount * v_tail

    # ---- core search ----

    def search(self, root_env: SnakeEnv) -> int:
        """
        Run MCTS from the given root state, return action to play.
        """
        root_grid = root_env.get_observation()
        root_key = self._state_key(root_grid)
        root = Node(root_key)

        # Ensure root expansion (get priors/value cached)
        root.P = self._get_priors(root_grid)
        root.expanded = True

        # Run simulations
        for _ in range(self.num_simulations):
            env = clone_env(root_env)
            node = root
            path: List[Tuple[Node, int, float]] = []  # (node, action, reward)

            # SELECTION & EXPANSION
            while True:
                best_score = -1e9
                best_action = 0
                sqrt_N = math.sqrt(max(1, node.N))
                for a in range(4):
                    child = node.children.get(a)
                    q = 0.0 if (child is None or child.N == 0) else child.Q
                    p = float(node.P[a]) if node.P is not None else 0.25
                    u = self.c_puct * p * sqrt_N / (1 + (0 if child is None else child.N))
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_action = a

                # apply chosen action
                _, reward, done = env.step(best_action)
                path.append((node, best_action, float(reward)))

                # move to child
                grid = env.get_observation()
                key = self._state_key(grid)
                child = node.children.get(best_action)
                if child is None:
                    child = Node(key)
                    node.children[best_action] = child

                node = child

                # If terminal, evaluate as 0 (future value) and back up
                if done:
                    leaf_value = 0.0
                    break

                # If this child hasn't been expanded before, expand & evaluate
                if not node.expanded:
                    node.P = self._get_priors(grid)
                    node.expanded = True
                    # Evaluate leaf via short policy rollout + bootstrap OR value-only
                    rollout_env = clone_env(env)
                    leaf_value = self._policy_rollout(rollout_env, self.rollout_depth)
                    break
                # else: continue down the tree

            # BACKUP (discounted along the path)
            G = float(leaf_value)
            for parent, action, r in reversed(path):
                G = r + self.gamma * G
                child = parent.children[action]
                child.N += 1
                child.W += G
                child.Q = child.W / child.N
                parent.N += 1

        # Choose action with highest visit count (AlphaZero/AlphaGo convention)
        visits = np.array([root.children[a].N if a in root.children else 0 for a in range(4)], dtype=np.int32)
        if visits.sum() == 0:
            # fallback: greedy according to policy
            grid = root_env.get_observation()
            pi = self._get_priors(grid)
            return int(np.argmax(pi))
        return int(np.argmax(visits))

# --------------------------
# --- Evaluation Routines --
# --------------------------

def play_game_greedy(net: nn.Module, width: int, height: int, seed: int, max_steps: int) -> int:
    env = SnakeEnv(width=width, height=height, seed=seed)
    steps = 0
    while not env.done and steps < max_steps:
        a = greedy_action(net, env.get_observation())
        env.step(a)
        steps += 1
    return env.score


def play_game_mcts(
    net: nn.Module,
    width: int,
    height: int,
    seed: int,
    max_steps: int,
    sims: int,
    c_puct: float,
    rollout_depth: int,
    gamma: float,
    greedy_rollout: bool,
    mcts_mode: str,
) -> int:
    env = SnakeEnv(width=width, height=height, seed=seed)
    use_policy_priors = (mcts_mode in ["alphago", "alphazero"])
    use_policy_rollout = (mcts_mode == "alphago")
    mcts = MCTS(
        net,
        gamma=gamma,
        c_puct=c_puct,
        num_simulations=sims,
        rollout_depth=rollout_depth,
        use_greedy_rollout=greedy_rollout,
        use_policy_priors=use_policy_priors,
        use_policy_rollout=use_policy_rollout,
    )
    steps = 0
    while not env.done and steps < max_steps:
        a = mcts.search(env)
        env.step(a)
        steps += 1
    return env.score


def summarize(scores: List[int]) -> Tuple[float, float, int, int]:
    arr = np.array(scores, dtype=np.float32)
    return float(arr.mean()), float(arr.std()), int(arr.min()), int(arr.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_avg.safetensors",
                        help="Path to .safetensors weights")
    parser.add_argument("--num-games", type=int, default=16)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=16384,
                        help="Cap on steps per game (MCTS is expensive; tune as needed)")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sims", type=int, default=16, help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.5, help="PUCT exploration constant")
    parser.add_argument("--rollout-depth", type=int, default=8,
                        help="Short policy rollout steps before bootstrapping with value (AlphaGo-style). Ignored in alphazero/value_only.")
    parser.add_argument("--greedy-rollout", action="store_true",
                        help="Use greedy policy for rollout instead of sampling (AlphaGo mode only)")
    parser.add_argument("--eval-greedy-only", action="store_true",
                        help="Only run greedy evaluation")
    parser.add_argument("--mcts-mode", type=str, choices=["alphago", "alphazero", "value_only"], default="alphago",
                        help="alphago: policy priors + rollouts + value; alphazero: policy priors, value-only leaves; value_only: uniform priors, value-only leaves")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load model
    net = ActorCritic()
    net.load_weights(args.checkpoint)
    mx.eval(net.parameters())

    # Evaluate Greedy
    print(f"Evaluating Greedy (argmax policy) on {args.num_games} games...")
    greedy_scores = []
    for seed in tqdm(range(args.num_games), desc="Greedy"):
        score = play_game_greedy(net, args.width, args.height, seed, args.max_steps)
        greedy_scores.append(score)
    g_mean, g_std, g_min, g_max = summarize(greedy_scores)
    print(f"\nGreedy Results over {args.num_games} games:")
    print(f"  Mean: {g_mean:.3f}  Std: {g_std:.3f}  Min: {g_min}  Max: {g_max}")

    if args.eval_greedy_only:
        return

    # Configure rollout depth for modes without rollouts
    rollout_depth = 0 if args.mcts_mode in ["alphazero", "value_only"] else args.rollout_depth

    # Evaluate MCTS
    label = f"MCTS-{args.mcts_mode}"
    print(f"\nEvaluating {label} (sims={args.sims}, c_puct={args.c_puct}, rollout_depth={rollout_depth}) "
          f"on {args.num_games} games...")
    mcts_scores = []
    for seed in tqdm(range(args.num_games), desc=label):
        score = play_game_mcts(
            net,
            args.width,
            args.height,
            seed,
            args.max_steps,
            sims=args.sims,
            c_puct=args.c_puct,
            rollout_depth=rollout_depth,
            gamma=args.gamma,
            greedy_rollout=args.greedy_rollout,
            mcts_mode=args.mcts_mode,
        )
        mcts_scores.append(score)
    m_mean, m_std, m_min, m_max = summarize(mcts_scores)

    # Report comparison
    print(f"\n{label} Results over {args.num_games} games:")
    print(f"  Mean: {m_mean:.3f}  Std: {m_std:.3f}  Min: {m_min}  Max: {m_max}")

    delta = m_mean - g_mean
    rel = (delta / (g_mean + 1e-8)) * 100.0
    print("\n==================== COMPARISON ====================")
    print(f"Mean improvement ({label} - Greedy): {delta:.3f}  ({rel:.2f}%)")
    print("====================================================")

    # Optionally, write detailed results to file
    out_path = f"mcts_vs_greedy_results_{args.mcts_mode}.txt"
    with open(out_path, "w") as f:
        f.write(f"MCTS ({args.mcts_mode}) vs Greedy on Snake\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Games: {args.num_games}\n")
        f.write(f"Board: {args.width}x{args.height}\n")
        f.write(f"Max steps: {args.max_steps}\n")
        f.write(f"MCTS sims: {args.sims}, c_puct: {args.c_puct}, rollout_depth={rollout_depth}\n\n")
        f.write(f"Greedy: mean={g_mean:.3f}, std={g_std:.3f}, min={g_min}, max={g_max}\n")
        f.write(f"MCTS-{args.mcts_mode}:   mean={m_mean:.3f}, std={m_std:.3f}, min={m_min}, max={m_max}\n")
        f.write(f"Mean improvement: {delta:.3f} ({rel:.2f}%)\n")
    print(f"\nDetailed results written to {out_path}")

if __name__ == "__main__":
    main()
