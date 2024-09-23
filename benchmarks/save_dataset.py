import json
from datasets import load_dataset

dataset = load_dataset("lmsys/lmsys-chat-1m")
print(dataset.keys())
print(len(dataset["train"]))
with open("lmsys_chat.jsonl", "w") as jsonl_file:
    for item in dataset["train"]:
        jsonl_file.write(json.dumps(item) + "\n")