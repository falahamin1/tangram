import os
import glob
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from x_example import SymbolicTangramGym
from DeepSetRL import DeepSetActorCritic
from PPOBuffer import PPOBuffer
import copy
import numpy as np

_KEEP = 3


def _save_checkpoint(ckpt_dir, ep, model, optimizer,
                     reward_history, best_moving_avg, best_weights):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f'checkpoint_ep{ep:07d}.pth')
    tmp  = path + '.tmp'
    torch.save({
        'episode':              ep,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'reward_history':       reward_history,
        'best_moving_avg':      best_moving_avg,
        'best_weights':         best_weights,
    }, tmp)
    os.replace(tmp, path)

    torch.save(best_weights, os.path.join(ckpt_dir, 'vrep_best.pth'))

    old = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_ep*.pth')))
    for f in old[:-_KEEP]:
        try:
            os.remove(f)
        except OSError:
            pass
    print(f"[V-rep] checkpoint saved → ep={ep}")


def _load_latest_checkpoint(ckpt_dir, model, optimizer):
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_ep*.pth')))
    if not ckpts:
        return 0, [], -float('inf'), None
    latest = ckpts[-1]
    print(f"[V-rep] Resuming from {latest}")
    data = torch.load(latest, map_location='cpu')
    model.load_state_dict(data['model_state_dict'])
    optimizer.load_state_dict(data['optimizer_state_dict'])
    return (data['episode'] + 1, data['reward_history'],
            data['best_moving_avg'], data['best_weights'])


def train_v_rep(episodes=1000,
                checkpoint_dir='checkpoints/vrep',
                checkpoint_interval=500):
    HP = {
        "lr": 3e-4,
        "clip_eps": 0.2,
        "ppo_epochs": 5,
        "steps_per_rollout": 1024,
        "batch_size": 64,
        "gamma": 0.99,
        "entropy_coef": 0.05,
        "critic_coef": 0.5,
        "max_grad_norm": 0.5,
        "moving_avg_window": 20,
    }

    device = torch.device("cpu")
    env = SymbolicTangramGym()
    model = DeepSetActorCritic(input_dim=8, num_pieces=6, num_actions=24).to(device)
    optimizer = optim.Adam(model.parameters(), lr=HP["lr"])
    buffer = PPOBuffer(size=HP["steps_per_rollout"], state_shape=(6, 8))

    start_ep, reward_history, best_moving_avg, loaded_best = \
        _load_latest_checkpoint(checkpoint_dir, model, optimizer)
    best_model_weights = loaded_best if loaded_best is not None else copy.deepcopy(model.state_dict())

    print(f"[V-rep] episodes={episodes}  start_ep={start_ep}  ckpt_dir={checkpoint_dir}")

    for ep in range(start_ep, episodes):
        obs = env.reset()
        ep_reward = 0

        for t in range(HP["steps_per_rollout"]):
            v_data   = obs['v_rep'].reshape(1, 6, 8)
            state_ts = torch.tensor(v_data, dtype=torch.float32).to(device)
            mask_ts  = torch.tensor(env.get_action_mask(), dtype=torch.bool).to(device)

            with torch.no_grad():
                logits, value = model(state_ts)
                logits[0][~mask_ts] = -1e10
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done, info = env.step(action.item())
            buffer.store(state_ts.squeeze(0), action, reward, value.item(), log_prob.item())
            obs = next_obs
            ep_reward += reward

            if done:
                buffer.finish_path(last_val=0)
                reward_history.append(ep_reward)
                obs = env.reset()
                ep_reward = 0
            elif t == HP["steps_per_rollout"] - 1:
                v_data_end    = obs['v_rep'].reshape(1, 6, 8)
                v_tensor_end  = torch.tensor(v_data_end, dtype=torch.float32).to(device)
                with torch.no_grad():
                    _, last_val = model(v_tensor_end)
                buffer.finish_path(last_val.item())

        if len(reward_history) >= HP["moving_avg_window"]:
            current_moving_avg = np.mean(reward_history[-HP["moving_avg_window"]:])
            if current_moving_avg > best_moving_avg:
                best_moving_avg = current_moving_avg
                best_model_weights = copy.deepcopy(model.state_dict())
                print(f"*** NEW BEST V-REP MODEL (Avg: {best_moving_avg:.2f}) at Ep {ep} ***")

        data = buffer.get()
        indices = np.arange(HP["steps_per_rollout"])
        for _ in range(HP["ppo_epochs"]):
            np.random.shuffle(indices)
            for start in range(0, HP["steps_per_rollout"], HP["batch_size"]):
                mb_idx = indices[start:start + HP["batch_size"]]
                mb_states       = data['states'][mb_idx]
                mb_actions      = data['actions'][mb_idx]
                mb_adv          = data['advantages'][mb_idx]
                mb_ret          = data['returns'][mb_idx]
                mb_old_logprobs = data['log_probs'][mb_idx]

                logits, values = model(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio  = torch.exp(new_log_probs - mb_old_logprobs)
                surr1  = ratio * mb_adv
                surr2  = torch.clamp(ratio, 1.0 - HP["clip_eps"], 1.0 + HP["clip_eps"]) * mb_adv
                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values.squeeze(-1), mb_ret)
                loss = actor_loss + HP["critic_coef"] * critic_loss - HP["entropy_coef"] * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), HP["max_grad_norm"])
                optimizer.step()

        buffer.clear()

        if ep % 100 == 0:
            avg = np.mean(reward_history[-10:]) if reward_history else 0
            print(f"V-Rep Ep {ep} | Recent Avg: {avg:.2f} | Best Avg: {best_moving_avg:.2f}")

        if checkpoint_interval > 0 and ep > 0 and ep % checkpoint_interval == 0:
            _save_checkpoint(checkpoint_dir, ep, model, optimizer,
                             reward_history, best_moving_avg, best_model_weights)

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'vrep_final.pth'))
    _save_checkpoint(checkpoint_dir, episodes - 1, model, optimizer,
                     reward_history, best_moving_avg, best_model_weights)
    print(f"[V-rep] Done. Models saved to {checkpoint_dir}/")

    best_model = DeepSetActorCritic(input_dim=8, num_pieces=6, num_actions=24)
    best_model.load_state_dict(best_model_weights)
    return model, best_model
