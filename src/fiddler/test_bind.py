import torch

from bfloat16_expert import cpu_expert
from mixtral import FiddlerMixtral
import time
import numpy as np
import argparse
import os


def format_output(array):
    return f"mean: {np.mean(array) * 1000:.2f} ms, std: {np.std(array) * 1000:.2f} ms"


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
    parser.add_argument(
        "--input",
        type=str,
        default="University of Washington is",
        help="Input text to generate.",
    )
    parser.add_argument(
        "--n-token",
        type=int,
        default=20,
        help="Number of tokens to generate.",
    )
    parser.add_argument("--torch_threads", type=int, default=16, help="Torch threads.")
    parser.add_argument("--cpp_threads", type=int, default=44, help="C++ threads.")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam search width.")
    parser.add_argument(
        "--token_num", type=int, default=128, help="Number of tokens to process."
    )
    parser.add_argument("--repeat", type=int, default=1, help="Repeat times.")

    args = parser.parse_args()
    model = FiddlerMixtral(args)
    # exit(0)
    N_DIM = 4096
    M_DIM = 14336
    token_num = 2
    inps = torch.randn((token_num, N_DIM), dtype=torch.bfloat16, device="cuda:0")
    w1 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
    w2 = torch.randn((N_DIM, M_DIM), dtype=torch.bfloat16)
    w3 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
    torch.set_num_threads(32)
    num_threads = [2 * i for i in range(1, 28)]
    times = []
    # expert = cpu_expert(w1, w2, w3)
    # warm up run
    # out = expert(inps.to("cpu"), 44)
    # inps = torch.randn((token_num, N_DIM), dtype=torch.bfloat16, device="cuda:0")
    # w1 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
    # w2 = torch.randn((N_DIM, M_DIM), dtype=torch.bfloat16)
    # w3 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
    for i in range(5):
        w1 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
        w2 = torch.randn((N_DIM, M_DIM), dtype=torch.bfloat16)
        w3 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
        w1.to("cuda:0")
        w2.to("cuda:0")
        w3.to("cuda:0")
    del w1, w2, w3
    torch.cuda.empty_cache()
    for i in range(model.n_layer):
        for j in range(model.n_expert):
            if not model.is_expert_in_gpu(i, j):
                # pin memory

                torch.cuda.synchronize()
                tick = time.time()

                # for name in ["w1", "w2", "w3"]:
                #     w = getattr(model.model.layers[i].block_sparse_moe.experts[j], name)
                #     src_weight_data_tensor = w.weight.data
                #     pinned = src_weight_data_tensor.pin_memory()
                #     w.weight.data = pinned

                # # copy["w1.weight", "w2.weight", "w3.weight"]
                # for name in ["w1", "w2", "w3"]:
                #     dst = getattr(model.expert_placeholder, name).weight.data
                #     src = getattr(
                #         model.model.layers[i].block_sparse_moe.experts[j], name
                #     ).weight.data
                #     dst.copy_(src)

                model.expert_placeholder.load_state_dict(
                    model.model.layers[i].block_sparse_moe.experts[j].state_dict()
                )
                torch.cuda.synchronize()
                times.append(time.time() - tick)

    print(format_output(times))
    # torch.cuda.synchronize()
    # start_time = time.time()
    # w1.to("cuda:0")
    # w2.to("cuda:0")
    # w3.to("cuda:0")
    # torch.cuda.synchronize()
    # transfer_time = (time.time() - start_time) / 3
    # print(f"Transfer time: {transfer_time*10**3:.2f} ms")
    times = []

    # for j in range(10):
    #     inps = torch.randn((token_num, N_DIM), dtype=torch.bfloat16, device="cuda:0")
    #     # w1 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
    #     # w2 = torch.randn((N_DIM, M_DIM), dtype=torch.bfloat16)
    #     # w3 = torch.randn((M_DIM, N_DIM), dtype=torch.bfloat16)
    #     # expert = cpu_expert(w1, w2, w3)
    #     start_time = time.time()
    #     out = expert(inps.to("cpu"), 44)
    #     times.append(time.time() - start_time)
    #     # print(out)
    # print(f"Time: {sum(times)/len(times)*10**6:.2f} us")
    # print(f"Variation:{np.var(times)*10**6:.2f} us")
