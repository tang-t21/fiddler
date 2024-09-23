import subprocess
import json
import random
import sys
import argparse

sys.path.append("../src")
from fiddler import FiddlerMixtral

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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

    path_json = "./ShareGPT_V3_unfiltered_cleaned_split.json"
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

    input_lengths = [2**i for i in range(5, 9)]
    output_lengths = [2**i for i in range(5, 9)]
    beam_widths = [1] + [4 * i for i in range(1, 4)]
    for input_length in input_lengths:
        for output_length in output_lengths:
            for beam_width in beam_widths:
                print(
                    f"Running input_length={input_length}, output_length={output_length}, beam_width={beam_width}"
                )
                idx_text = 0
                while True:
                    text = texts[idx_text]
                    idx_text += 1
                    if len(text.split()) >= input_length:
                        # enough input length
                        break
                prefill_time_sum, decode_time_sum, hit_rate_sum = 0, 0, 0
                for i in range(n_sample):
                    prefill_time, decode_time, hit_rate = model.generate(
                        [text],
                        output_token=output_length,
                        input_token=input_length,
                        beam_width=beam_width,
                    )
                    prefill_time_sum += prefill_time
                    decode_time_sum += decode_time
                    hit_rate_sum += hit_rate
                with open("./beam-bench-results.txt", "a") as f:
                    f.write(
                        f"{input_length},{output_length},{beam_width},{prefill_time_sum/n_sample:.3f},{decode_time_sum/n_sample:.3f},{hit_rate_sum/n_sample:.4f}\n"
                    )
