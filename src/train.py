import argparse

from trainer import run
from utils import format_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration")
    # Standard training args
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step_train_size", type=int, required=True)
    parser.add_argument("--target_dir", type=str, default="results/")
    parser.add_argument("--verbose", type=int, default=0)
    # Adv train/test args
    parser.add_argument("--adv_test_mode", default=False, action="store_true")
    parser.add_argument("--adv_train_mode", default=False, action="store_true")
    parser.add_argument("--attack_type", type=str, default="fgsm")
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--eps_iter", type=float, default=0.01)
    parser.add_argument("--nb_iter", type=float, default=10)
    args = parser.parse_args()
    output = run(args)
    print(format_output(output))
