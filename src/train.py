import argparse

from trainer import run
from utils import format_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--emnist_split", type=str, default="balanced")
    parser.add_argument("--img_height", type=int, required=True)
    parser.add_argument("--img_width", type=int, required=True)
    parser.add_argument("--in_channels", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_train_size", type=int, required=True)
    parser.add_argument("--min_train_size", type=int, required=True)
    parser.add_argument("--model_path", type=str, default="model_ckpt/")
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--notebook", default=False, action="store_true")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dim", type=int, required=True)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--step_train_size", type=int, required=True)
    parser.add_argument("--target_dir", type=str, default=".")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()
    output = run(args)
    format_output(output)
