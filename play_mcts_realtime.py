
import argparse
import math
import os
import time
import random
from typing import Dict, Optional, Tuple, List

import numpy as np
import pygame

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


def clone_env(env: SnakeEnv, seed: Optional[int] = None) -> SnakeEnv:
    """
    Deep-ish copy without peeking into exact RNG future.
    Use a separate seed so MCTS samples different futures.
    """
    new_env = SnakeEnv(width=env.width, height=env.height)
    new_env.snake = list(env.snake)
    new_env.direction = tuple(env.direction)
    new_env.apple = tuple(env.apple)
    new_env.done = bool(env.done)
    new_env.score = int(env.score)
    if seed is None:
        seed = random.randrange(2**31 - 1)
    new_env.reseed(seed)
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
        num_simulations: int = 16,
        rollout_depth: int = 8,
        use_greedy_rollout: bool = False,
        mode: str = "alphazero",  # "alphazero" or "alphago"
        clone_seed: int = 0,
    ):
        self.net = net
        self.gamma = gamma
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.use_greedy_rollout = use_greedy_rollout
        assert mode in ("alphazero", "alphago")
        self.mode = mode
        self._clone_rng = random.Random(clone_seed)
        # caches
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
        k = self._state_key(grid)
        if k in self.cache_pi:
            return self.cache_pi[k]
        pi, _ = policy_value(self.net, grid)
        self.cache_pi[k] = pi
        return pi

    def _rollout_eval(self, env: SnakeEnv) -> float:
        """
        For alphago mode: do a short policy rollout (sample or greedy),
        accumulate discounted rewards, then bootstrap with value.
        For alphazero mode: just return value.
        """
        if self.mode != "alphago" or self.rollout_depth <= 0:
            return self._get_value(env.get_observation())

        total = 0.0
        discount = 1.0
        for _ in range(self.rollout_depth):
            grid = env.get_observation()
            pi = self._get_priors(grid)
            if self.use_greedy_rollout:
                a = int(np.argmax(pi))
            else:
                a = int(np.random.choice(4, p=pi / (pi.sum() + 1e-8)))
            _, r, done = env.step(a)
            total += discount * float(r)
            if done:
                return total
            discount *= self.gamma
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

        # Expand root
        root.P = self._get_priors(root_grid)
        root.expanded = True

        for _ in range(self.num_simulations):
            # independent clone with its own RNG seed
            seed = self._clone_rng.randrange(2**31 - 1)
            env = clone_env(root_env, seed=seed)

            node = root
            path: List[Tuple[Node, int, float]] = []  # (node, action, reward)

            while True:
                # PUCT selection
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

                _, reward, done = env.step(best_action)
                path.append((node, best_action, float(reward)))

                grid = env.get_observation()
                key = self._state_key(grid)
                child = node.children.get(best_action)
                if child is None:
                    child = Node(key)
                    node.children[best_action] = child
                node = child

                if done:
                    leaf_value = 0.0
                    break

                if not node.expanded:
                    node.P = self._get_priors(grid)
                    node.expanded = True
                    # evaluate leaf
                    rollout_env = clone_env(env, seed=self._clone_rng.randrange(2**31 - 1))
                    leaf_value = self._rollout_eval(rollout_env)
                    break

            # backup
            G = float(leaf_value)
            for parent, action, r in reversed(path):
                G = r + self.gamma * G
                child = parent.children[action]
                child.N += 1
                child.W += G
                child.Q = child.W / child.N
                parent.N += 1

        visits = np.array([root.children[a].N if a in root.children else 0 for a in range(4)], dtype=np.int32)
        if visits.sum() == 0:
            pi = self._get_priors(root_grid)
            return int(np.argmax(pi))
        return int(np.argmax(visits))

# ----------------------
# --- Pygame display ---
# ----------------------

def draw_grid(screen, grid: np.ndarray, cell: int, margin: int, font, info_text: str):
    height, width = grid.shape
    screen.fill((245, 245, 245))

    # draw cells
    for y in range(height):
        for x in range(width):
            v = grid[y, x]
            if v == 0:      # empty
                color = (255, 255, 255)
            elif v == 1:    # body
                color = (0, 200, 0)
            elif v == 2:    # apple
                color = (220, 0, 0)
            elif v == 3:    # head
                color = (0, 0, 220)
            else:
                color = (200, 200, 200)
            pygame.draw.rect(
                screen,
                color,
                (margin + x * cell, margin + y * cell, cell - 1, cell - 1),
                border_radius=3
            )

    # grid border
    rect = (margin - 2, margin - 2, width * cell + 4, height * cell + 4)
    pygame.draw.rect(screen, (50, 50, 50), rect, width=2, border_radius=6)

    # overlay text
    if info_text:
        text_surf = font.render(info_text, True, (20, 20, 20))
        screen.blit(text_surf, (margin, margin + height * cell + 10))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="good_policy.safetensors")
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--cell-size", type=int, default=32)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--rollout-depth", type=int, default=8)
    parser.add_argument("--mode", type=str, choices=["alphazero", "alphago"], default="alphago")
    parser.add_argument("--greedy-rollout", action="store_true", help="Use greedy rollout (alphago mode only)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # Load network
    net = ActorCritic()
    net.load_weights(args.checkpoint)
    mx.eval(net.parameters())

    # Pygame init
    pygame.init()
    font = pygame.font.SysFont("consolas", 18)
    width_px = args.margin * 2 + args.width * args.cell_size
    height_px = args.margin * 2 + args.height * args.cell_size + 40
    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Snake — MCTS + Policy (Realtime)")
    clock = pygame.time.Clock()

    # Env + MCTS
    def new_env(seed=None):
        if seed is None:
            seed = random.randrange(2**31 - 1)
        return SnakeEnv(width=args.width, height=args.height, seed=seed)

    env = new_env()
    mcts = MCTS(
        net,
        gamma=args.gamma,
        c_puct=args.c_puct,
        num_simulations=args.sims,
        rollout_depth=(0 if args.mode == "alphazero" else args.rollout_depth),
        use_greedy_rollout=args.greedy_rollout,
        mode=args.mode,
        clone_seed=0,
    )

    running = True
    last_step_time = time.time()
    steps = 0
    games_played = 0
    best_score = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    env = new_env()
                    steps = 0
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    mcts.num_simulations += 8
                elif event.key == pygame.K_MINUS:
                    mcts.num_simulations = max(1, mcts.num_simulations - 8)
                elif event.key == pygame.K_m:
                    if mcts.mode == "alphazero":
                        mcts.mode = "alphago"
                        mcts.rollout_depth = args.rollout_depth
                    else:
                        mcts.mode = "alphazero"
                        mcts.rollout_depth = 0
                elif event.key == pygame.K_g:
                    mcts.use_greedy_rollout = not mcts.use_greedy_rollout
                elif event.key == pygame.K_LEFTBRACKET:
                    if mcts.mode == "alphago":
                        mcts.rollout_depth = max(0, mcts.rollout_depth - 1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    if mcts.mode == "alphago":
                        mcts.rollout_depth += 1

        # Decide action via MCTS (this may take longer than 1/fps; the clock will cap display rate only)
        a = mcts.search(env)
        env.step(a)
        steps += 1

        # Render
        grid = env.get_observation()
        info = f"score={env.score}   sims={mcts.num_simulations}   \nmode={mcts.mode},rollout={mcts.rollout_depth},fps={args.fps}"
        draw_grid(screen, grid, args.cell_size, args.margin, font, info)
        pygame.display.flip()
        clock.tick(args.fps)

        # Handle terminal
        if env.done:
            best_score = max(best_score, env.score)
            games_played += 1
            # small pause to show the result
            end_info = f"GAME OVER — score={env.score}  best={best_score}  games={games_played}. Restarting..."
            draw_grid(screen, grid, args.cell_size, args.margin, font, end_info)
            pygame.display.flip()
            pygame.time.delay(600)
            env = new_env()
            steps = 0

    pygame.quit()


if __name__ == "__main__":
    main()