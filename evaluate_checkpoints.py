import os
import glob
import numpy as np
from tqdm import tqdm
import mlx.core as mx
import mlx.nn as nn
from main_ppo import SnakeEnv, ActorCritic, grid_to_onehot3_np
mx.set_default_device(mx.cpu)
def evaluate_checkpoint(checkpoint_path, num_games=128):
    """Evaluate a single checkpoint across multiple games in parallel"""
    # Load the model
    agent_net = ActorCritic()
    agent_net.load_weights(checkpoint_path)
    mx.eval(agent_net.parameters())
    
    # Create parallel environments
    envs = []
    for i in range(num_games):
        env = SnakeEnv(width=10, height=10, seed=i)
        envs.append(env)
    
    # Reset all environments
    observations = [env.get_observation() for env in envs]
    dones = [False] * num_games
    
    # Run all games until completion
    max_steps = 256 + 100 * 99  # Match the max curriculum from training
    for _ in range(max_steps):
        # Check if all games are done
        if all(dones):
            break
        
        # Encode all observations at once
        enc_list = []
        active_indices = []
        for i, (obs, done) in enumerate(zip(observations, dones)):
            if not done:
                enc = grid_to_onehot3_np(obs)
                enc_list.append(enc)
                active_indices.append(i)
        
        if not enc_list:
            break
        
        # Get actions for all active games at once
        enc_batch = np.stack(enc_list, axis=0)
        obs_mx = mx.array(enc_batch, dtype=mx.float32)
        logits, _ = agent_net(obs_mx)
        actions = mx.argmax(logits, axis=1)
        actions_np = np.array(actions.tolist(), dtype=np.int32)
        
        # Step all active environments
        action_idx = 0
        for i in active_indices:
            if not dones[i]:
                obs, reward, done = envs[i].step(int(actions_np[action_idx]))
                observations[i] = obs
                dones[i] = done
                action_idx += 1
    
    # Collect final scores
    scores = [env.score for env in envs]
    return scores

def main():
    # Get all checkpoint files
    checkpoint_files = glob.glob("checkpoints/model_epoch_*.safetensors")
    checkpoint_files.append("checkpoints/best_model.safetensors")
    
    # Sort checkpoint files by epoch number
    def get_epoch(path):
        if "best_model" in path:
            return -1  # Put best_model first
        try:
            return int(path.split("_")[-1].replace(".safetensors", ""))
        except:
            return 0
    
    checkpoint_files.sort(key=get_epoch)
    
    print(f"Found {len(checkpoint_files)} checkpoints to evaluate")
    print("Evaluating each checkpoint on 128 games...")
    
    results = {}
    best_avg_score = -1
    best_checkpoint = None
    
    for checkpoint_path in tqdm(checkpoint_files, desc="Evaluating checkpoints"):
        try:
            scores = evaluate_checkpoint(checkpoint_path, num_games=128)
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            
            checkpoint_name = os.path.basename(checkpoint_path)
            results[checkpoint_name] = {
                'avg': avg_score,
                'std': std_score,
                'max': max_score,
                'min': min_score,
                'scores': scores
            }
            
            print(f"\n{checkpoint_name}:")
            print(f"  Average score: {avg_score:.2f} ± {std_score:.2f}")
            print(f"  Min/Max: {min_score}/{max_score}")
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_checkpoint = checkpoint_path
                
        except Exception as e:
            print(f"Error evaluating {checkpoint_path}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    # Sort by average score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg'], reverse=True)
    
    print("\nTop 10 checkpoints by average score:")
    for i, (name, stats) in enumerate(sorted_results[:10]):
        print(f"{i+1}. {name}: {stats['avg']:.2f} ± {stats['std']:.2f}")
    
    print(f"\nBest checkpoint: {os.path.basename(best_checkpoint)}")
    print(f"Best average score: {best_avg_score:.2f}")
    
    # Save the best checkpoint as best_avg.safetensors
    if best_checkpoint:
        print(f"\nSaving best checkpoint to checkpoints/best_avg.safetensors...")
        weights = mx.load(best_checkpoint)
        mx.save_safetensors("checkpoints/best_avg.safetensors", weights)
        print("Done!")
    
    # Save detailed results to a text file
    with open("checkpoint_evaluation_results.txt", "w") as f:
        f.write("Checkpoint Evaluation Results\n")
        f.write("="*50 + "\n\n")
        for name, stats in sorted_results:
            f.write(f"{name}:\n")
            f.write(f"  Average: {stats['avg']:.2f}\n")
            f.write(f"  Std Dev: {stats['std']:.2f}\n")
            f.write(f"  Min: {stats['min']}\n")
            f.write(f"  Max: {stats['max']}\n")
            f.write(f"  All scores: {stats['scores']}\n\n")
    
    print("\nDetailed results saved to checkpoint_evaluation_results.txt")

if __name__ == "__main__":
    main()