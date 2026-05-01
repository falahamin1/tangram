import os
import argparse
from hrep import train_h_rep

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Train H-rep PPO agent (server)')
    ap.add_argument('--episodes',             type=int, default=40000)
    ap.add_argument('--checkpoint-dir',       default=os.path.join(os.path.dirname(__file__), 'checkpoints', 'hrep'))
    ap.add_argument('--checkpoint-interval',  type=int, default=500,
                    help='Save checkpoint every N episodes (0 = off)')
    args = ap.parse_args()

    train_h_rep(
        episodes=args.episodes,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
    )
