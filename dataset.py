import os
from datasets import load_dataset

SPLIT_FILES = ["train_split.txt", "val_split.txt", "test_split.txt"]
VOCAB_FILE = "vocab.txt"


def splits_and_vocab_exist():
    return all(os.path.exists(f) for f in SPLIT_FILES + [VOCAB_FILE])


def atomic_write(filename, lines):
    tmp_filename = filename + ".tmp"
    try:
        with open(tmp_filename, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        os.replace(tmp_filename, filename)
    except Exception as e:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)
        raise e


def save_split(filename, data):
    atomic_write(filename, data)


def save_vocab(train_data, val_data, test_data):
    all_text = "".join(train_data) + "".join(val_data) + "".join(test_data)
    vocab = set(all_text)
    atomic_write(VOCAB_FILE, sorted(vocab))


def process_and_save_minipile():
    try:
        ds = load_dataset("JeanKaddour/minipile")
        train_data = ds["train"]["text"]
        val_data = ds["validation"]["text"]
        test_data = ds["test"]["text"]

        save_split("train_split.txt", train_data)
        save_split("val_split.txt", val_data)
        save_split("test_split.txt", test_data)
        save_vocab(train_data, val_data, test_data)
        print(
            "Splits and vocab saved: train_split.txt, val_split.txt, test_split.txt, vocab.txt"
        )
    except Exception as e:
        print(f"Error during dataset processing: {e}")


def prepare_dataset():
    if splits_and_vocab_exist():
        resp = input(
            "Splits and vocab already exist. Download and process again? (y/N): "
        ).strip().lower()
        if resp not in ("y", "yes"):
            print("Using existing splits and vocab.")
            return
        print("Overwriting existing splits and vocab...")
    process_and_save_minipile()


if __name__ == "__main__":
    prepare_dataset()