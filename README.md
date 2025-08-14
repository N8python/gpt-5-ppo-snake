# GPT-5-PPO-Snake
This repository contains an implementation of Proximal Policy Optimization (PPO) for training an AI agent to play a 10x10 Snake game. The code for the PPO and MCTS was mostly written by GPT-5, hence the name of the repository. This repository is an **experiment** in what modern LLMs can write autonomously. DO NOT use this code for production. Uses MLX, so Mac only.

https://github.com/user-attachments/assets/540a4e48-f6fd-40fb-b59f-796a2133fc68

## Requirements
To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

You can see an already-trained agent in action by running:

```bash
python play_mcts_realtime.py
```

To train your own PPO agent, run:

```bash
python main_ppo.py --num-updates 128
```

This should take 1 hour on an M3 Max. By the end of training, your agent should get an average score of 40-50 apples each game.

(Run for 512 updates to get an average score of 80 apples per game.)

To prepare your agent for play, run:

```bash
python evaluate_checkpoints.py
```

This will evaluate all saved checkpoints and save the best one to `checkpoints/best_avg.safetensors`. This 'best' checkpoint is determined by the checkpoint that performs the best under greedy decoding.

You can see the difference MCTS makes by running:

```bash
python compare_mcts.py --num-games 16
```

For reference, on a sample training run for 128 steps, the default MCTS parameters improved an average score across 16 games from 54.7 (no MCTS) to 92.7 (with MCTS).

On a fully trained policy, the default MCTS parameters typically guarantee a perfect score of 99 apples per game.

To watch your trained agent play with MCTS, run:

```bash
python play_mcts_realtime.py --checkpoint checkpoints/best_avg.safetensors
```
