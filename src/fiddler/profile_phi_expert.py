"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import numpy as np

sys.path.append("../src")
from phi import FiddlerPhi

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )

    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")
    parser.add_argument("--torch_threads", type=int, default=16, help="Torch threads.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["LMSYS", "ShareGPT"],required=True)
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples.")
    args = parser.parse_args()

    path_json = args.data_path
    dataset_name=args.dataset
    texts=[]
    if dataset_name == "LMSYS":
        with open(path_json, "r") as f:
            data = [json.loads(line)["conversation"][0]["content"] for line in f]
        dataset_name="LMSYS"
        for d in data:
            if len(d) == 0:
                continue
            # the input of the first round
            texts.append(" ".join(d.split()))
    elif dataset_name == "ShareGPT":
        with open(path_json, "r") as f:
            data = json.load(f)
        for d in data:
            if len(d["conversations"]) == 0:
                continue
            # the input of the first round
            texts.append(" ".join(d["conversations"][0]["value"].split()))
    else:
        raise ValueError("UNSUPPORTED DATASET!")

    random.seed(0)
    random.shuffle(texts)
    model = FiddlerPhi(args)
    for input_token in [16]:
        for output_token in [64]:
            print(f"input_token: {input_token}, output_token: {output_token}")
            for _ in range(2):
                idx_text = 0
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_token:
                        # enough input length
                        break
                # print("text:", text)
                prefill_time, decode_time, hit_rate = model.generate(
                    [text], output_token=output_token, input_token=input_token
                )
                print(
                    "prefill_time:",
                    prefill_time,
                    "decode_time:",
                    decode_time,
                )
    model.reset_popular_experts()
    file_name = f"./expert_popularity-{dataset_name}.txt"
    num_eval_samples = 0
    output_token = 128
    for text in texts:
        if len(text.split(" ")) > 1024:
            continue
            # print("text:", text)
        prefill_time, decode_time, expert_token_cnt = model.generate([text], output_token=output_token)
        print(
            "prefill_time:",
            prefill_time,
            "decode_time:",
            decode_time,
        )
        num_eval_samples += 1
        if num_eval_samples >= args.num_samples:
            break

    model.write_popular_experts(file_name)
            