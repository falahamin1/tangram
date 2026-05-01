from hrep import train_h_rep
from vrep import train_v_rep
from x_example import SymbolicTangramGym
import torch
from DeepSetRL import DeepSetActorCritic

# def compare_representations(model_h, model_v, num_episodes=5):
#     """Compare H-rep and V-rep models with faster evaluation."""
#     env = SymbolicTangramGym()

#     results = {}

#     for rep_type, model in [("H-Representation", model_h), ("V-Representation", model_v)]:
#         print(f"\nEvaluating {rep_type}...")
#         episode_rewards = []

#         for ep in range(num_episodes):
#             obs = env.reset()
#             total_reward = 0
#             steps = 0

#             while steps < 200:  # Limit steps for faster evaluation
#                 key = 'h_rep' if rep_type == "H-Representation" else 'v_rep'
#                 state = torch.tensor(obs[key], dtype=torch.float32).unsqueeze(0)

#                 with torch.no_grad():
#                     logits, _ = model(state)
#                     action = torch.argmax(logits, dim=-1)

#                 obs, reward, done, _ = env.step(action.item())
#                 total_reward += reward
#                 steps += 1

#                 if done:
#                     break

#             episode_rewards.append(total_reward)
#             print(f"  Episode {ep+1}: {total_reward:.2f} reward in {steps} steps")

#         avg_reward = sum(episode_rewards) / len(episode_rewards)
#         results[rep_type] = avg_reward
#         print(f"{rep_type} Average: {avg_reward:.2f}")

#     return results
def compare_representations(best_h, best_v, num_episodes=1):
    env = SymbolicTangramGym()
    results = {}

    # Map the passed objects to their respective configuration
    configs = [
        {"name": "H-Rep", "model": best_h, "key": "h_rep", "dim": 15},
        {"name": "V-Rep", "model": best_v, "key": "v_rep", "dim": 8}
    ]

    for config in configs:
        print(f"\n--- Evaluating {config['name']} ---")
        
        # We no longer need torch.load() because 'model' is already the best_model!
        eval_model = config['model']
        eval_model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_model.to(device)

        obs = env.reset()
        total_reward = 0

        for steps in range(1, 101):
            raw_data = torch.tensor(obs[config['key']], dtype=torch.float32).to(device)
            state = raw_data.view(1, 6, config['dim'])

            with torch.no_grad():
                logits, _ = eval_model(state)
                mask = torch.tensor(env.get_action_mask(), dtype=torch.bool).to(device)
                logits[0][~mask] = -1e10
                action = torch.argmax(logits, dim=-1)
                print(f"Step {steps}: Action {action.item()} | Reward: {total_reward:.2f}")

            obs, reward, done, _ = env.step(action.item())
            total_reward += reward

            # Render logic
            if steps % 10 == 0 or done:
                label = "FINAL" if done else f"Step-{steps}"
                env.inner.render(f"{config['name']}-{label}")

            if done:
                print(f"  Success! {config['name']} solved it in {steps} steps.")
                break
        
        results[config['name']] = total_reward

    return results

    
def run_quick_comparison():
    """Run a quick comparison with reduced training."""
    print("--- Starting Quick H-Representation Training (100 episodes) ---")
    model_h = train_h_rep(episodes=100)

    print("\n--- Starting Quick V-Representation Training (100 episodes) ---")
    model_v = train_v_rep(episodes=100)

    print("\n--- Quick Head-to-Head Evaluation ---")
    results = compare_representations(model_h, model_v, num_episodes=3)

    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    for rep, score in results.items():
        print(f"{rep}: {score:.2f} average reward")

    return results

def run_full_comparison():
    """Original full comparison (will take very long)."""
    print("WARNING: Full comparison will take several hours!")
    print("Consider using run_quick_comparison() instead.")
    user_input = input("Continue with full training? (y/N): ")
    if user_input.lower() != 'y':
        print("Aborting full comparison.")
        return

    print("--- Starting H-Representation Training ---")
    model_h, best_model_h = train_h_rep(episodes=10000)

    print("\n--- Starting V-Representation Training ---")
    model_v, best_model_v = train_v_rep(episodes=10000)

    print("\n--- Final Head-to-Head Evaluation ---")
    results = compare_representations(best_model_h, best_model_v, num_episodes=1)
    return results

def run_ultra_quick_comparison():
    """Run an ultra-quick comparison with minimal training for testing."""
    print("--- Starting Ultra-Quick H-Representation Training (5 episodes) ---")
    model_h = train_h_rep(episodes=5)

    print("\n--- Starting Ultra-Quick V-Representation Training (5 episodes) ---")
    model_v = train_v_rep(episodes=5)

    print("\n--- Quick Head-to-Head Evaluation ---")
    results = compare_representations(model_h, model_v, num_episodes=2)

    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    for rep, score in results.items():
        print(f"{rep}: {score:.2f} average reward")

    return results

if __name__ == "__main__":
    # Use ultra-quick comparison for testing
    run_full_comparison()