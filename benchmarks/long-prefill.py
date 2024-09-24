"""Microbenchmarking for CPU offloading"""

import argparse
import json
import os
import random
import sys
import numpy as np

sys.path.append("../src")
from fiddler import FiddlerMixtral

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mixtral-8x7B-v0.1",
        help="Model path. default `mistralai/Mixtral-8x7B-v0.1`.",
    )
    parser.add_argument(
        "--cpu-offload",
        type=int,
        default=1,
        choices=[0, 1],
        help="0: exeute at GPU (baseline), 1: offload to CPU.",
    )

    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")
    parser.add_argument("--torch_threads", type=int, default=16, help="Torch threads.")
    parser.add_argument("--cpp_threads", type=int, default=44, help="C++ threads.")

    args = parser.parse_args()

    # path_json = "./lmsys_chat.jsonl"
    # with open(path_json, "r") as f:
    #     data = [json.loads(line) for line in f]

    # texts = []
    # for d in data:
    #     if len(d["conversation"]) == 0:
    #         continue
    #     # the input of the first round
    #     texts.append(" ".join(d["conversation"][0]["content"].split()))
    
    path_json = "./ShareGPT_V3_unfiltered_cleaned_split.json"
    dataset_name="ShareGPT"
    with open(path_json, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        if len(d["conversations"]) == 0:
            continue
        # the input of the first round
        texts.append(" ".join(d["conversations"][0]["value"].split()))

    random.seed(0)
    random.shuffle(texts)
    model = FiddlerMixtral(args)
    n_sample = 3
    for input_token in [16]:
        for output_token in [64]:
            idx_text = 0
            prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
            print(f"input_token: {input_token}, output_token: {output_token}")
            model.reset_expert_loc((output_token+input_token))
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
    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    file_name = f"./results/prefill-long-latency-{dataset_name}-{args.torch_threads}-{args.cpp_threads}.txt"
    with open(file_name, "a") as f:
        f.write("input_length,prefill_time(s),throughput(token/s)\n")
    for input_token in [4096]:
        idx_text = 0
        input_text = None
        for text in texts:
            if len(text.split()) >= input_token:
                # enough input length
                input_text = text
                break
        if input_text is None:
            print(f"No enough input length for length larger than {input_token}")
            break
        output_token=1
        n_sample = 3 if output_token < 1024 else 1
        print(f"input_token: {input_token}, output_token: {output_token}")
        model.reset_expert_loc((output_token+input_token-1))
        prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
        for _ in range(n_sample):

            # print("text:", text)
            prefill_time, decode_time, hit_rate = model.generate(
                [input_text], output_token=1, input_token=input_token
            )
            print(
                "prefill_time:",
                prefill_time,
                'hit_rate:', hit_rate
            )
            prefill_time_sum += prefill_time
            hit_rate_sum += hit_rate
        # write to file
        with open(file_name, "a") as f:
            f.write(
                f"{input_token}, "
                f"{prefill_time_sum / n_sample}, "
                f"{input_token *n_sample/ (prefill_time_sum):.2f}\n"
            )
