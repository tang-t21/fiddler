import datasets
import subprocess
from fiddler import FiddlerMixtral


if __name__ == "__main__":
    data = datasets.load_dataset(
        "json",
        data_files="https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/raw/main/raw/question.jsonl",
        split="train",
    )
    categories = [
        "writing",
        "roleplay",
        "reasoning",
        "math",
        "coding",
        "extraction",
        "stem",
        "humanities",
    ]
    num_per_category = len(data) // len(categories)
    print(num_per_category)

    # for i, category in enumerate(categories):
    #     with open(f"../../results/{category}.txt", "w") as f:
