"""Split the deduplicated gold-standard corpus into train and test sets.

Shuffles the corpus with a fixed seed for reproducibility and holds out a
fixed-size test set, writing both splits back to disk as JSON.

Example:
    python scripts/split_data.py --test-size 500 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DEFAULT_INPUT = DATA_DIR / "gold_standard_deduplicated.json"
DEFAULT_TRAIN = DATA_DIR / "train_split.json"
DEFAULT_TEST = DATA_DIR / "test_split.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT,
                        help="Path to the deduplicated gold-standard corpus.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN,
                        help="Output path for the training split.")
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST,
                        help="Output path for the test split.")
    parser.add_argument("--test-size", type=int, default=500,
                        help="Number of samples to hold out for the test set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used to shuffle the corpus.")
    return parser.parse_args()


def split_dataset(args):
    print(f"Loading dataset from {args.input_file}...")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: could not find {args.input_file}")
        return

    total_samples = len(data)
    print(f"Loaded {total_samples} pairs.")

    if total_samples <= args.test_size:
        print(f"ERROR: not enough data. You need more than {args.test_size} samples.")
        return

    args.train_file.parent.mkdir(parents=True, exist_ok=True)

    print("Shuffling the dataset...")
    random.seed(args.seed)
    random.shuffle(data)

    test_data = data[:args.test_size]
    train_data = data[args.test_size:]

    print(f"Saving {len(train_data)} training samples to {args.train_file}...")
    with open(args.train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4)

    print(f"Saving {len(test_data)} evaluation samples to {args.test_file}...")
    with open(args.test_file, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4)

    print("\nSplit complete.")


if __name__ == "__main__":
    split_dataset(parse_args())
