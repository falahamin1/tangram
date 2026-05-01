import os
import argparse
from vrep import train_v_rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train V-rep PPO agent (server)')
    ap.add_argument('--episodes',             type=int, default=40000)
    ap.add_argument('--checkpoint-dir',       default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'vrep'))
    ap.add_argument('--checkpoint-interval',  type=int, default=500,
                    help='Save checkpoint every N episodes (0 = off)')
    args = ap.parse_args()

    train_v_rep(
        episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
