import random
import numpy as np
import imageio  # For writing mp4 video
import time
import matplotlib.pyplot as plt
from tqdm import trange

# MLX imports
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# ------------------------
# --- Encoding helpers ---
# ------------------------

def grid_to_onehot3_np(grid: np.ndarray) -> np.ndarray:
    """
    Convert a [H, W] int grid to a flattened 3-channel one-hot vector [H*W*3].
    Channel order matches decoder:
      ch0 = apple(2), ch1 = snake body(1), ch2 = snake head(3)
    Returns uint8 (0/1) for stable round-trip with one_hot_to_grid; you’ll cast to float later.
    """
    h, w = grid.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., 0] = (grid == 2)  # apple
    out[..., 1] = (grid == 1)  # body
    out[..., 2] = (grid == 3)  # head
    return out.reshape(-1)

# ----------------
# --- ENV CODE ---
# ----------------

class SnakeEnv:
    def __init__(self, width=10, height=10, seed=None):
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.reset()

    def reseed(self, seed):
        self.rng = random.Random(seed)

    def reset(self):
        # Initialize the snake in the center
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (1, 0)  # Start moving right
        self.generate_apple()
        self.done = False
        self.score = 0
        return self.get_observation()

    def generate_apple(self):
        # Get all empty spaces
        empty_spaces = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.snake:
                    empty_spaces.append((x, y))
        
        # If no empty space exists, return None
        if not empty_spaces:
            return None
        
        # Randomly select an empty space
        self.apple = self.rng.choice(empty_spaces)
        return self.apple

    def get_observation(self):
        """
        Returns a grid representation:
          - 0: empty
          - 1: snake body
          - 2: apple
          - 3: snake head
        """
        grid = np.zeros((self.height, self.width), dtype=int)
        for i, (x, y) in enumerate(self.snake):
            grid[y, x] = 3 if i == 0 else 1
        ax, ay = self.apple
        grid[ay, ax] = 2
        return grid

    def step(self, action):
        """
        action: 0=up, 1=right, 2=down, 3=left
        Returns: (observation, reward, done)
        """
        action_mapping = {
            0: (0, -1),  # up
            1: (1, 0),   # right
            2: (0, 1),   # down
            3: (-1, 0)   # left
        }
        new_direction = action_mapping[action]

        # Prevent reversing if snake length > 1
        if (len(self.snake) > 1 and
            new_direction[0] == -self.direction[0] and
            new_direction[1] == -self.direction[1]):
            new_direction = self.direction
        self.direction = new_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.width or
            new_head[1] < 0 or new_head[1] >= self.height):
            self.done = True
            return self.get_observation(), -1, self.done

        # Self collision
        if new_head in self.snake:
            self.done = True
            return self.get_observation(), -1, self.done

        # Move
        self.snake.insert(0, new_head)
        reward = 0  # small move penalty - encourage efficiency
        if new_head == self.apple:
            reward = 1
            self.score += 1
            apple_result = self.generate_apple()
            if apple_result is None:
                # Won the game - filled the entire board!
                self.done = True
                reward = 10
        else:
            self.snake.pop()

        return self.get_observation(), reward, self.done

# ------------------------------
# --- Actor-Critic (PPO) NN ---
# ------------------------------

class ActorCritic(nn.Module):
    def __init__(self, input_size=300, hidden1=512, hidden2=512, hidden3=256, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden1)
        self.fc3 = nn.Linear(hidden1, hidden3)
        self.pi = nn.Linear(hidden3, action_dim)   # policy logits
        self.v  = nn.Linear(hidden3, 1)            # state-value

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.relu(self.fc3(x))
        logits = self.pi(x)
        value = self.v(x).reshape((-1,))
        return logits, value

# ---------------
# --- PPO Core ---
# ---------------

class PPOAgent:
    def __init__(self,
                 temperature=1.0,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_range=0.2,
                 value_clip_range=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01):
        self.net = ActorCritic()
        mx.eval(self.net.parameters())
        self.temperature = temperature
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    # --------- small utils ---------
    def one_hot_encode(self, grid):
        return grid_to_onehot3_np(grid)

    def sample_actions(self, logits):
        adj = logits / self.temperature
        sampled = mx.random.categorical(adj, stream=mx.cpu)
        return np.array(sampled.tolist(), dtype=np.int32)

# ---- math helpers shared by rollout & update ----

def log_probs_from_logits(logits: mx.array, actions: mx.array) -> mx.array:
    logp_all = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    idx = actions.reshape((-1, 1))
    logp = mx.take_along_axis(logp_all, idx, axis=1).reshape((-1,))
    return logp

def entropy_from_logits(logits: mx.array) -> mx.array:
    logp_all = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    p_all = mx.exp(logp_all)
    ent = -mx.sum(p_all * logp_all, axis=1)
    return mx.mean(ent)

# ---------------------
# --- PPO Rollouts  ---
# ---------------------

def collect_rollout(num_envs, num_steps, width, height, agent: PPOAgent, seed=None):
    envs = []
    for i in range(num_envs):
        env = SnakeEnv(width, height, seed=(None if seed is None else seed + i))
        envs.append(env)

    observations = [env.get_observation() for env in envs]

    obs_buf = []      # [T, N, 300]
    acts_buf = []     # [T, N]
    logp_buf = []     # [T, N]
    val_buf = []      # [T, N]
    rew_buf = []      # [T, N]
    done_buf = []     # [T, N]

    # For logging/videos and curriculum
    ep_returns = [[] for _ in range(num_envs)]
    finished_episode_returns = []
    finished_episode_scores = []          # <— add this
    record_env = 0
    record_states, record_actions = [], []
    recorded_traj = None
    max_score_seen = 0

    for t in range(num_steps):
        # Encode all current observations
        enc = np.stack([agent.one_hot_encode(obs) for obs in observations], axis=0)
        obs_mx = mx.array(enc, dtype=mx.float32)

        # Policy pass
        logits, values = agent.net(obs_mx)
        actions_np = agent.sample_actions(logits)
        actions_mx = mx.array(actions_np, dtype=mx.int32)
        logp = log_probs_from_logits(logits, actions_mx)

        # Store
        obs_buf.append(enc)
        acts_buf.append(actions_np)
        logp_buf.append(np.array(logp.tolist(), dtype=np.float32))
        val_buf.append(np.array(values.tolist(), dtype=np.float32))

        # Interact with envs
        rewards = np.zeros((num_envs,), dtype=np.float32)
        dones = np.zeros((num_envs,), dtype=np.float32)

        for i, env in enumerate(envs):
            if i == record_env:
                record_states.append(enc[i].copy())
                record_actions.append(int(actions_np[i]))

            obs, r, d = env.step(int(actions_np[i]))
            rewards[i] = r
            dones[i] = 1.0 if d else 0.0
            ep_returns[i].append(r)

            # Track best score seen so far across envs
            if env.score > max_score_seen:
                max_score_seen = env.score

            if d:
                # Episode ended: track return and reset env
                finished_episode_returns.append(sum(ep_returns[i]))
                finished_episode_scores.append(env.score)   # <— add this
                ep_returns[i].clear()
                obs = env.reset()
                if i == record_env and recorded_traj is None:
                    recorded_traj = (sum(ep_returns[i]) if ep_returns[i] else 0.0,
                                     list(zip(record_states, record_actions)))
                    record_states, record_actions = [], []

            observations[i] = obs

        rew_buf.append(rewards)
        done_buf.append(dones)

    # Bootstrap with last value for each env
    enc_last = np.stack([agent.one_hot_encode(obs) for obs in observations], axis=0)
    last_obs_mx = mx.array(enc_last, dtype=mx.float32)
    _, last_values_mx = agent.net(last_obs_mx)
    last_values = np.array(last_values_mx.tolist(), dtype=np.float32)  # [N]

    # Convert buffers to arrays
    obs_arr = np.asarray(obs_buf, dtype=np.float32)   # [T, N, 300]
    acts_arr = np.asarray(acts_buf, dtype=np.int32)   # [T, N]
    logp_arr = np.asarray(logp_buf, dtype=np.float32) # [T, N]
    val_arr = np.asarray(val_buf, dtype=np.float32)   # [T, N]
    rew_arr = np.asarray(rew_buf, dtype=np.float32)   # [T, N]
    done_arr = np.asarray(done_buf, dtype=np.float32) # [T, N]

    # Compute GAE advantages and returns
    adv_arr, ret_arr = compute_gae(rew_arr, done_arr, val_arr, last_values,
                                   gamma=agent.gamma, lam=agent.gae_lambda)

    # Flatten [T, N] -> [T*N]
    T, N = acts_arr.shape
    flat_obs = obs_arr.reshape(T * N, -1)
    flat_act = acts_arr.reshape(T * N)
    flat_oldlogp = logp_arr.reshape(T * N)
    flat_adv = adv_arr.reshape(T * N)
    flat_ret = ret_arr.reshape(T * N)
    flat_oldv = val_arr.reshape(T * N)

    # Advantage normalization
    if flat_adv.size > 1 and np.std(flat_adv) > 1e-6:
        flat_adv = (flat_adv - np.mean(flat_adv)) / (np.std(flat_adv) + 1e-8)

    avg_ep_return = float(np.mean(finished_episode_returns)) if finished_episode_returns else 0.0

    # Provide some trajectory for rendering
    if recorded_traj is None:
        recorded_traj = (avg_ep_return, list(zip(list(flat_obs[:min(128, len(flat_obs))]), list(flat_act[:min(128, len(flat_act))]))))

    avg_ep_score = float(np.mean(finished_episode_scores)) if finished_episode_scores else 0.0
    
    return flat_obs, flat_act, flat_oldlogp, flat_adv, flat_ret, flat_oldv, recorded_traj, avg_ep_return, avg_ep_score, max_score_seen, finished_episode_scores


def compute_gae(rew: np.ndarray, done: np.ndarray, val: np.ndarray, last_val: np.ndarray, gamma=0.99, lam=0.95):
    """
    rew:  [T, N]
    done: [T, N] (1 where episode ended at that step)
    val:  [T, N]
    last_val: [N] value of final states for bootstrapping
    Returns (adv, ret) each [T, N]
    """
    T, N = rew.shape
    adv = np.zeros((T, N), dtype=np.float32)
    ret = np.zeros((T, N), dtype=np.float32)

    values_ext = np.zeros((T + 1, N), dtype=np.float32)
    values_ext[:-1] = val
    values_ext[-1] = last_val

    gae = np.zeros((N,), dtype=np.float32)
    for t in reversed(range(T)):
        mask = 1.0 - done[t]
        delta = rew[t] + gamma * values_ext[t + 1] * mask - values_ext[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        ret[t] = adv[t] + values_ext[t]

    return adv, ret

# -----------------------
# --- PPO Update step ---
# -----------------------

def iterate_minibatches(batch_size, *arrays_mx):
    N = len(arrays_mx[0])
    perm = mx.random.permutation(N)
    for s in range(0, N, batch_size):
        idx = perm[s: s + batch_size]
        yield [arr[idx] for arr in arrays_mx]


def ppo_loss(agent: PPOAgent, net, obs_b, act_b, oldlogp_b, adv_b, ret_b, oldv_b):
    logits, values = net(obs_b)  # values: [B]

    # Policy loss (clipped)
    newlogp = log_probs_from_logits(logits, act_b)
    ratio = mx.exp(newlogp - oldlogp_b)
    clipped_ratio = mx.clip(ratio, 1.0 - agent.clip_range, 1.0 + agent.clip_range)
    loss_pi = -mx.mean(mx.minimum(ratio * adv_b, clipped_ratio * adv_b))

    # Value loss with clipping
    v_pred = values
    v_pred_clipped = oldv_b + mx.clip(v_pred - oldv_b, -agent.value_clip_range, agent.value_clip_range)
    v_loss_unclipped = (ret_b - v_pred) ** 2
    v_loss_clipped = (ret_b - v_pred_clipped) ** 2
    v_loss = mx.mean(mx.maximum(v_loss_unclipped, v_loss_clipped))

    # Entropy bonus
    ent = entropy_from_logits(logits)

    loss = loss_pi + agent.value_coef * v_loss - agent.entropy_coef * ent
    return loss


def train_one_iteration(agent: PPOAgent,
                        optimizer,
                        flat_obs_np, flat_act_np, flat_oldlogp_np, flat_adv_np, flat_ret_np, flat_oldv_np,
                        batch_size=4096, update_epochs=4):
    # Move to MX
    obs = mx.array(flat_obs_np, dtype=mx.float32)
    acts = mx.array(flat_act_np, dtype=mx.int32)
    oldlogp = mx.array(flat_oldlogp_np, dtype=mx.float32)
    adv = mx.array(flat_adv_np, dtype=mx.float32)
    ret = mx.array(flat_ret_np, dtype=mx.float32)
    oldv = mx.array(flat_oldv_np, dtype=mx.float32)

    net = agent.net
    loss_and_grad = nn.value_and_grad(
        net,
        lambda net, ob, ac, olp, ad, rt, ov: ppo_loss(agent, net, ob, ac, olp, ad, rt, ov)
    )

    # Epochs of SGD with minibatches
    last_loss = 0.0
    for _ in range(update_epochs):
        for ob_b, ac_b, olp_b, ad_b, rt_b, ov_b in iterate_minibatches(batch_size, obs, acts, oldlogp, adv, ret, oldv):
            loss, grads = loss_and_grad(net, ob_b, ac_b, olp_b, ad_b, rt_b, ov_b)
            optimizer.update(net, grads)
            last_loss = loss.item()

    return last_loss

# -------------------------
# --- Rendering Helpers ---
# -------------------------

def one_hot_to_grid(state, height=10, width=10):
    reshaped = state.reshape((height, width, 3))
    grid = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            cell = reshaped[i, j]
            if cell[0] == 1:
                grid[i, j] = 2   # apple
            elif cell[1] == 1:
                grid[i, j] = 1   # snake body
            elif cell[2] == 1:
                grid[i, j] = 3   # snake head
            else:
                grid[i, j] = 0   # empty
    return grid


def render_board(grid, cell_size=20):
    color_map = {
        0: [255, 255, 255],  # empty: white
        1: [0, 255, 0],      # snake body: green
        2: [255, 0, 0],      # apple: red
        3: [0, 0, 255]       # snake head: blue
    }
    height, width = grid.shape
    img = np.zeros((height * cell_size, width * cell_size, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            color = color_map[grid[i, j]]
            img[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size] = color
    return img

# --------------------------
# --- Main Training Loop ---
# --------------------------

# Clear "videos" directory
import os
import shutil

# Suppress ffmpeg_writer warnings by setting the log level to "error"
os.environ["IMAGEIO_FFMPEG_LOG_LEVEL"] = "error"

shutil.rmtree("videos", ignore_errors=True)
os.makedirs("videos", exist_ok=True)

if __name__ == "__main__":
    import sys
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PPO agent on Snake game')
    parser.add_argument('--num-updates', type=int, default=128,
                        help='Number of training iterations (default: 128)')
    args = parser.parse_args()
    
    # PPO config
    num_updates   = args.num_updates  # training iterations from command line
    num_envs      = 64     # parallel envs
    base_steps    = 256    # base steps per env per update
    batch_size    = 4096   # SGD minibatch size
    update_epochs = 4      # PPO epochs per update

    agent = PPOAgent(
        temperature=1.0,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        value_clip_range=0.20,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    
    optimizer = optim.Adam(1e-3)
    
    # Load weights if resuming
    # Best model tracking
    best_avg_score = 0  # Track best average score for model saving
    best_score_ever = 0  # progressive curriculum: apples eaten best so far

    avg_returns = []
    avg_scores = []  # Track average scores for plotting
    p90_prev = 0  # 90th percentile score from the previous update (none yet)
    
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)

    pbar = trange(num_updates, desc="PPO Training")
    for update in pbar:
        # Progressive curriculum: extend horizon with best score so far
        curr_steps = int(base_steps + 100 * min(best_score_ever, 99))

        # 1) Rollout
        flat_obs, flat_act, flat_oldlogp, flat_adv, flat_ret, flat_oldv, traj_for_render, avg_ep_ret, avg_ep_score, max_score_seen, finished_episode_scores = collect_rollout(
            num_envs=num_envs,
            num_steps=curr_steps,
            width=10,
            height=10,
            agent=agent,
            seed=random.randint(0, 2**31 - 1),
        )

        # Update curriculum stat
        if max_score_seen > best_score_ever:
            best_score_ever = int(max_score_seen)

        last_loss = train_one_iteration(
            agent,
            optimizer,
            flat_obs, flat_act, flat_oldlogp, flat_adv, flat_ret, flat_oldv,
            batch_size=batch_size,
            update_epochs=update_epochs,
        )
        
        # Update learning rate with cosine decay
        progress = update / (num_updates - 1)  # 0 to 1
        lr = 1e-3 * (0.5 * (1 + np.cos(np.pi * progress)))
        optimizer.learning_rate = lr

        if finished_episode_scores:
            p90_this = int(np.percentile(finished_episode_scores, 90))
        else:
            p90_this = p90_prev  # fallback if no episodes finished (unlikely)

        p90_prev = p90_this  # carry forward for the next update’s horizon

        avg_returns.append(avg_ep_ret)
        avg_scores.append(avg_ep_score)
        pbar.set_postfix(avg_return=avg_ep_ret, avg_score=avg_ep_score, loss=last_loss,
                         samples=len(flat_act), steps=curr_steps,
                         p90_score=p90_this, best_score=best_score_ever, lr=lr)
        
        # Save best model if average score improved
        weights = dict(tree_flatten(agent.net.trainable_parameters()))
        if avg_ep_score > best_avg_score:
            best_avg_score = avg_ep_score
            best_model_path = "checkpoints/best_model.safetensors"
            mx.save_safetensors(best_model_path, weights)
            print(f"\nNew best model saved with average score: {best_avg_score:.2f}")
        # Save model every 10 epochs and track best model
        if (update + 1) % 10 == 0:
            # Save checkpoint
            checkpoint_path = f"checkpoints/model_epoch_{update + 1}.safetensors"
            mx.save_safetensors(checkpoint_path, weights)
            
            
            # Update plot - show average score instead of return
            plt.figure(figsize=(10, 6))
            plt.plot(avg_scores)
            plt.xlabel("Update")
            plt.ylabel("Avg Score (Apples Eaten)")
            plt.title(f"PPO: Avg Score per Update (Epoch {update + 1})")
            plt.grid(True)
            plt.savefig(f"checkpoints/training_progress_epoch_{update + 1}.png")
            plt.close()

        # 3) Render a trajectory
        if traj_for_render is not None:
            _, pairs = traj_for_render
            frames = []
            for state_encoded, action in pairs:
                grid = one_hot_to_grid(np.array(state_encoded), 10, 10)
                frame = render_board(grid, cell_size=16)
                frames.append(frame)
            video_filename = f"videos/update_{update+1}_sample_run.mp4"
            imageio.mimwrite(
                video_filename,
                frames,
                fps=5,
                codec="h264"            # or "mpeg4" if h264 isn’t available
            )

    # Save training curves and weights
    with open("avg_returns.txt", "w") as f:
        for r in avg_returns:
            f.write(f"{r}\n")
    
    with open("avg_scores.txt", "w") as f:
        for s in avg_scores:
            f.write(f"{s}\n")

    weights = dict(tree_flatten(agent.net.trainable_parameters()))
    mx.save_safetensors("snake_ppo.safetensors", weights)

    # Plot average scores instead of returns
    plt.plot(avg_scores)
    plt.xlabel("Update")
    plt.ylabel("Avg Score (Apples Eaten)")
    plt.title("PPO: Avg Score per Update")
    plt.grid(True)
    plt.savefig("avg_scores.png")
    plt.show()
