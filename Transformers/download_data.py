from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train")
out_path = "/home/Transformers/DATA/tinystories.txt"

with open(out_path, "w", encoding="utf-8") as f:
    for example in dataset:
        text = example["text"].strip()
        if text:  # skip empty
            f.write(text + "\n")

print(f"✅ Saved TinyStories dataset to {out_path}")
