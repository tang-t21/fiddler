import matplotlib.pyplot as plt
import numpy as np

sys_name = "Twiddler"

result_dir="/home/ubuntu/fiddler-Tian/benchmarks/results"

envs=["rtx6000", "A6000"]
datasets=["sharegpt","lmsys"]

def read_normal_results(env, dataset, framework):
    throughputs = []
    with open(f"{result_dir}/latency-{env}-{dataset}-{framework}.txt", "r") as f:
        for line in f[1:]:
            if int(line.split(",")[1]) <= 1024 and int(line.split(",")[0]) <= 1024:
                throughputs.append(float(line.split(",")[-1]))
    return throughputs


def read_long_results(env, dataset, framework):
    throughputs = []
    with open(f"{result_dir}/latency-{env}-{dataset}-{framework}.txt", "r") as f:
        for line in f[1:]:
            if int(line.split(",")[1]) > 512 or int(line.split(",")[0]) >= 512:
                throughputs.append(float(line.split(",")[-1]))
    return throughputs

def read_refill_results(env, dataset, framework):
    latency = []
    with open(f"{result_dir}/prefill-{env}-{dataset}-{framework}.txt", "r") as f:
        for line in f[1:]:
            latency.append(float(line.split(",")[-2]))
    return latency


def normal_e2e():
    # Sample data (replace with your actual data)
    input_output_tokens = [
        "[32,64]",
        "[32,128]",
        "[32,256]", 
        '[32,512]',
        "[32,1024]",
        "[64,64]",
        "[64,128]",
        "[64,256]",  
        '[64,512]',
        "[64,1024]",
        "[128,64]",
        "[128,128]",
        "[128,256]",  
        '[128,512]',
        "[128,1024]",
        "[256,64]",
        "[256,128]",
        "[256,256]",  
        '[256,512]',
        "[256,1024]",
        "[512,64]",
        "[512,128]",
        "[512,256]",  
        '[512,512]',
        "[512,1024]",
        "[1024,64]",
        "[1024,128]",
        "[1024,256]",
        '[1024,512]',
        "[1024,1024]",
    ]
    n_bar = len(input_output_tokens)
    tokens_per_second = {
        "sharegpt":{
        "env0": {
            "DeepSpeed-MII": read_normal_results(envs[0],datasets[0], "deepspeed-mii"),
            "Eliseev & Mazur": read_normal_results(envs[0], datasets[0], "mixtral-offload"),
            "llama.cpp": read_normal_results(envs[0],datasets[0], "llamacpp"),
            sys_name: read_normal_results(envs[0],datasets[0], "twiddler"),
        },
        "env1": {
            # This is fake data
            'DeepSpeed-MII': read_normal_results(envs[1],datasets[0], "deepspeed-mii"),
            'Eliseev & Mazur': read_normal_results(envs[1], datasets[0], "mixtral-offload"),
            'llama.cpp': read_normal_results(envs[1],datasets[0], "llamacpp"),
            sys_name: read_normal_results(envs[1],datasets[0], "twiddler"),
        },
        },
        "lmsys":{
             "env0": {
            "DeepSpeed-MII": read_normal_results(envs[0],datasets[1], "deepspeed-mii"),
            "Eliseev & Mazur": read_normal_results(envs[0], datasets[1], "mixtral-offload"),
            "llama.cpp": read_normal_results(envs[0],datasets[1], "llamacpp"),
            sys_name: read_normal_results(envs[0],datasets[1], "twiddler"),
            },
            "env1": {
            # This is fake data
            'DeepSpeed-MII': read_normal_results(envs[1],datasets[1], "deepspeed-mii"),
            'Eliseev & Mazur': read_normal_results(envs[1], datasets[1], "mixtral-offload"),
            'llama.cpp': read_normal_results(envs[1],datasets[1], "llamacpp"),
            sys_name: read_normal_results(envs[1],datasets[1], "twiddler"),
            }
        }
    }
    return tokens_per_second, input_output_tokens

def plot_e2e(dataset, tokens_per_second, input_output_tokens, output_dir):
    # append each list with mean value
    print(f"e2e of {dataset}")
    tokens_per_second=tokens_per_second[dataset]
    for env in tokens_per_second[dataset].keys():
        for key in tokens_per_second[env].keys():
            tokens_per_second[env][key].append(np.mean(tokens_per_second[env][key]))
            print(
                "dataset:", dataset, "env:", env, "key:", key, "mean:", np.mean(tokens_per_second[env][key])
            )
    mean_label = "Mean"
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("Paired").colors)

    # Creating subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 4))

    # Plot data for Environment 1
    bar_width = 0.2
    index = np.arange(len(input_output_tokens) + 1)  # Adding one for the mean column

    for i in range(2):
        axes[i].axvspan(
            -0.5 + len(input_output_tokens),
            0.5 + len(input_output_tokens),
            color="gray",
            alpha=0.3,
        )
        axes[i].bar(
            index - 1.5 * bar_width,
            tokens_per_second[f"env{i}"]["DeepSpeed-MII"],
            bar_width * 0.8,
            label="DeepSpeed-MII",
            edgecolor="black",
        )
        axes[i].bar(
            index - 0.5 * bar_width,
            tokens_per_second[f"env{i}"]["Eliseev & Mazur"],
            bar_width * 0.8,
            label="Eliseev & Mazur",
            edgecolor="black",
        )
        axes[i].bar(
            index + 0.5 * bar_width,
            tokens_per_second[f"env{i}"]["llama.cpp"],
            bar_width * 0.8,
            label="llama.cpp",
            edgecolor="black",
            hatch="//",
        )
        axes[i].bar(
            index + 1.5 * bar_width,
            tokens_per_second[f"env{i}"][sys_name],
            bar_width * 0.8,
            label=sys_name,
            edgecolor="black",
            hatch="\\",
        )

        # write a vertical line
        axes[i].axvline(x=3 - 0.5, color="black", linestyle="-")
        axes[i].axvline(x=6 - 0.5, color="black", linestyle="-")
        axes[i].axvline(x=9 - 0.5, color="black", linestyle="-")
        axes[i].axvline(x=12 - 0.5, color="black", linestyle="-")
        axes[i].axvline(x=15 - 0.5, color="black", linestyle="-")

        axes[i].set_xlim(-0.5, len(input_output_tokens) + 0.5)
        axes[i].grid(
            which="major", axis="y", color="gray", linestyle="--", linewidth=1.0
        )
        # remove xticks label
        axes[i].set_xticks([])
        axes[i].tick_params(axis="x", which="minor", length=0)
        axes[i].tick_params(axis="x", which="major", length=0)

    fig.supxlabel("[Input Length, Output Length]", fontsize=12)
    fig.text(
        0.01,
        0.5,
        "Inference Speed (token/s) ↑",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=12,
    )

    # Adjust layout to make room for the shared labels and avoid overlap
    # fig.tight_layout(rect=[-0.5, 0, 1, 1])
    # set y-axis limit
    axes[0].set_ylim(0, 4)
    # axes[1].set_ylim(0, 4)
    axes[1].set_ylim(0, 9)

    axes[0].set_title("Environment 1 (Quadro RTX 6000 GPU)")
    axes[1].set_title("Environment 2 (RTX A6000 GPU)")
    # axes[2].set_title('Environment 3 (RTX 6000 Ada GPU)')
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(input_output_tokens + [mean_label], rotation=0)

    # Add legends
    axes[0].legend(ncol=4)
    # axes[2].legend()
    # plt.ylabel("Inference Speed (token/s) ↑", fontsize=12)
    # plt.xlabel("[Input Length, Output Length]", fontsize=12)

    plt.tight_layout(rect=[0.02, 0, 1, 1])
    plt.savefig(f"{output_dir}/e2e-{dataset}.png")


def long_context():
    # Sample data (replace with your actual data)
    input_tokens = [
        "512",
        "1024",
        "2048",
        "4096",
        "8192"
    ]

    mean_label = "Mean"
    prefill_latency = {
        "env0": {
            "DeepSpeed-MII": read_refill_results(envs[0],datasets[0], "deepspeed-mii"),
            "Eliseev & Mazur": read_refill_results(envs[0], datasets[0], "mixtral-offload"),
            "llama.cpp": read_refill_results(envs[0],datasets[0], "llamacpp"),
            sys_name: read_refill_results(envs[0],datasets[0], "twiddler"),
        },
        "env1": {
            "DeepSpeed-MII": read_refill_results(envs[1],datasets[0], "deepspeed-mii"),
            "Eliseev & Mazur": read_refill_results(envs[1], datasets[0], "mixtral-offload"),
            "llama.cpp":  read_refill_results(envs[1],datasets[0], "llamacpp"),
            sys_name: read_refill_results(envs[1],datasets[0], "twiddler"),
        },
    }

    speed_ups = {"env0": [], "env1": []}
    for env in prefill_latency.keys():
        for i in range(len(input_tokens)):
            speed_up_length = []
            for key in prefill_latency[env].keys():
                speed_up_length.append(
                    prefill_latency[env][key][i] / prefill_latency[env][sys_name][i]
                )
            speed_ups[env].append(min(speed_up_length[:-1]))
    print("env0 speed up", np.mean(speed_ups["env0"]))
    print("env1 speed up", np.mean(speed_ups["env1"]))
    print(
        "total average speed up", np.mean(list(speed_ups["env0"] + speed_ups["env1"]))
    )

    # append each list with mean value
    print("long_context")
    for env in prefill_latency.keys():
        for key in prefill_latency[env].keys():
            prefill_latency[env][key].append(np.mean(prefill_latency[env][key]))
            print("env:", env, "key:", key, "mean:", np.mean(prefill_latency[env][key]))

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("Paired").colors)

    # Creating subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot data for Environment 1
    bar_width = 0.2
    index = np.arange(len(input_tokens) + 1)  # Adding one for the mean column
    for i in range(2):
        axes[i].axvspan(
            -0.5 + len(input_tokens), 0.5 + len(input_tokens), color="gray", alpha=0.3
        )
        axes[i].bar(
            index - 1.5 * bar_width,
            prefill_latency[f"env{i}"]["DeepSpeed-MII"],
            bar_width * 0.8,
            label="DeepSpeed-MII",
            edgecolor="black",
        )
        axes[i].bar(
            index - 0.5 * bar_width,
            prefill_latency[f"env{i}"]["Eliseev & Mazur"],
            bar_width * 0.8,
            label="Eliseev & Mazur",
            edgecolor="black",
        )
        axes[i].bar(
            index + 0.5 * bar_width,
            prefill_latency[f"env{i}"]["llama.cpp"],
            bar_width * 0.8,
            label="llama.cpp",
            edgecolor="black",
            hatch="//",
        )
        axes[i].bar(
            index + 1.5 * bar_width,
            prefill_latency[f"env{i}"][sys_name],
            bar_width * 0.8,
            label=sys_name,
            edgecolor="black",
            hatch="\\",
        )

        # write a vertical line
        # axes[i].axvline(x=3 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=6 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=9 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=12 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=15 - 0.5, color='black', linestyle='-')

        axes[i].set_xlim(-0.5, len(input_tokens) + 0.5)
        axes[i].grid(
            which="major", axis="y", color="gray", linestyle="--", linewidth=1.0
        )
        # remove xticks label
        axes[i].set_xticks(index)
        axes[i].set_xticklabels(input_tokens + [mean_label], rotation=0)
        axes[i].tick_params(axis="x", which="minor", length=0)
        axes[i].tick_params(axis="x", which="major", length=0)
        axes[i].set_xlabel("Input Length", fontsize=12)
        axes[i].set_ylabel("Time To First Token (s) ↓", fontsize=12)

    # add text to environment 2 saying OOM in vertical
    # axes[1].text(2.9, 3, 'Out Of Memory', fontsize=10, color='red', ha='center', rotation=90)

    axes[0].set_title("Environment 1 (Quadro RTX 6000 GPU)")
    # axes[1].set_title('Environment 2 (L4 GPU)')
    axes[1].set_title("Environment 2 (RTX 6000 Ada GPU)")

    # axes[0].set_ylabel("Time To First Token (s) ↓")

    # set y-axis limit
    axes[0].set_ylim(0, 60)
    # axes[1].set_ylim(0, 40)
    axes[1].set_ylim(0, 20)

    # Add legends
    axes[0].legend(ncol=1, loc="upper left")
    # axes[2].legend()

    plt.tight_layout()
    plt.savefig("/home/tian21/fiddler/asset/long_context.png")


def beam():
    # Sample data (replace with your actual data)
    input_tokens = [
        # '1',
        "4",
        "8",
        "12",
        "16",
    ]

    mean_label = "Mean"
    prefill_latency = {
        "env0": {
            "llama.cpp": [
                # 1.83,
                0.14,
                0.04,
                0.03,
                0.02,
            ],
            sys_name: [
                # 2.05,
                0.76,
                0.47,
                0.36,
                0.29,
            ],
        },
        "env1": {
            "llama.cpp": [
                # 7.11,
                0.30,
                0.10,
                0.06,
                0.04,
            ],
            sys_name: [
                # 6.82,
                1.22,
                0.92,
                0.91,
                0.82,
            ],
        },
    }
    speed_up0=[]
    speed_up1=[]
    for i in range(len(input_tokens)):
        speed_up0.append(prefill_latency["env0"][sys_name][i]/prefill_latency["env0"]["llama.cpp"][i])
        speed_up1.append(prefill_latency["env1"][sys_name][i]/prefill_latency["env1"]["llama.cpp"][i])

    # append each list with mean value
    print("beam")
    print("env0 speed up", np.mean(speed_up0))
    print("env1 speed up", np.mean(speed_up1))
    print("total average speed up", np.mean(speed_up0+speed_up1))
    for env in prefill_latency.keys():
        for key in prefill_latency[env].keys():
            prefill_latency[env][key].append(np.mean(prefill_latency[env][key]))
            print("env:", env, "key:", key, "mean:", np.mean(prefill_latency[env][key]))

    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.get_cmap("Paired").colors
    )[2:]

    # Creating subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Plot data for Environment 1
    bar_width = 0.2
    index = np.arange(len(input_tokens) + 1)  # Adding one for the mean column
    for i in range(2):
        axes[i].axvspan(
            -0.5 + len(input_tokens), 0.5 + len(input_tokens), color="gray", alpha=0.3
        )
        # axes[i].bar(
        #     index - 1.5 * bar_width,
        #     prefill_latency[f'env{i}']['DeepSpeed-MII'],
        #     bar_width * 0.8,
        #     label='DeepSpeed-MII',
        #     edgecolor="black",
        # )
        # axes[i].bar(
        #     index - 0.5 * bar_width,
        #     prefill_latency[f'env{i}']['Eliseev & Mazur'],
        #     bar_width * 0.8,
        #     label='Eliseev & Mazur',
        #     edgecolor="black",
        # )
        axes[i].bar(
            index - bar_width,
            prefill_latency[f"env{i}"]["llama.cpp"],
            bar_width * 0.8,
            label="llama.cpp",
            edgecolor="black",
            hatch="//",
        )
        axes[i].bar(
            index + bar_width,
            prefill_latency[f"env{i}"][sys_name],
            bar_width * 0.8,
            label=sys_name,
            edgecolor="black",
            hatch="\\",
        )

        # write a vertical line
        # axes[i].axvline(x=3 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=6 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=9 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=12 - 0.5, color='black', linestyle='-')
        # axes[i].axvline(x=15 - 0.5, color='black', linestyle='-')

        axes[i].set_xlim(-0.5, len(input_tokens) + 0.5)
        axes[i].grid(
            which="major", axis="y", color="gray", linestyle="--", linewidth=1.0
        )
        # remove xticks label
        axes[i].set_xticks(index)
        axes[i].set_xticklabels(input_tokens + [mean_label], rotation=0)
        axes[i].tick_params(axis="x", which="minor", length=0)
        axes[i].tick_params(axis="x", which="major", length=0)
        axes[i].set_xlabel("Beam Search Width", fontsize=12)
        axes[i].set_ylabel("Inference Speed (token/s) ↑", fontsize=12)

    axes[0].set_title("Environment 1 (Quadro RTX 6000 GPU)")
    # axes[1].set_title('Environment 2 (L4 GPU)')
    axes[1].set_title("Environment 2 (RTX 6000 Ada GPU)")

    # set y-axis limit
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 2)
    # axes[2].set_ylim(0, 2)

    # axes[0].set_ylabel("Inference Speed (token/s) ↑")

    # Add legends
    axes[0].legend(ncol=2, loc="upper left")
    # axes[2].legend()

    plt.tight_layout()
    plt.savefig("/home/tian21/fiddler/asset/beam.png")


def microbench():
    # Sample data (replace with your actual data)
    labels = [
        "W copy",
        "A copy",
        "CPU 1",
        "CPU 2",
        "CPU 4",
        "CPU 8",
        "CPU 16",
        "CPU 32",
        "GPU 1",
        "GPU 2",
        "GPU 4",
        "GPU 8",
        "GPU 16",
        "GPU 32",
    ]

    mean_label = "Mean"
    latency = {
        "env0": [
            28.93,
            0.03,
            15.37,
            11.70,
            17.20,
            23.75,
            48.63,
            78.71,
            0.73,
            4.94,
            4.97,
            4.97,
            4.97,
            5.00,
        ],
        # "env1": [
        #     0.0 for _ in range(len(labels))
        # ],
        "env1": [
            14.22,
            0.01,
            3.94,
            7.32,
            10.65,
            18.81,
            38.54,
            53.91,
            0.46,
            0.48,
            0.48,
            0.48,
            0.49,
            0.51,
        ],
    }

    std = {
        "env0": [
            0.11,
            0.01,
            4.65,
            3.68,
            5.73,
            15.17,
            17.82,
            26.92,
            0.05,
            0.05,
            0.06,
            0.05,
            0.06,
            0.06,
        ],
        # "env1": [
        #     0.0 for _ in range(len(labels))
        # ],
        "env1": [
            0.03,
            0.01,
            0.03,
            0.10,
            2.03,
            0.19,
            0.85,
            0.93,
            0.02,
            0.01,
            0.01,
            0.01,
            0.00,
            0.01,
        ],
    }

    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("Paired").colors)

    # Creating subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))

    # Plot data for Environment 1
    bar_width = 0.6
    index = np.arange(len(labels))  # Adding one for the mean column
    for i in range(2):
        axes[i].bar(
            index,
            latency[f"env{i}"],
            bar_width * 0.8,
            yerr=std[f"env{i}"],
            # label='llama.cpp',
            edgecolor="black",
            hatch="//",
        )

        # write a vertical line
        axes[i].axvline(x=2 - 0.5, color="black", linestyle="-")
        axes[i].axvline(x=8 - 0.5, color="black", linestyle="-")

        # set y-axis to log scale
        axes[i].set_yscale("log")

        axes[i].set_xlim(-0.5, len(labels) - 0.5)
        axes[i].grid(
            which="major", axis="y", color="gray", linestyle="--", linewidth=1.0
        )
        # remove xticks label
        axes[i].set_xticks(index)
        axes[i].set_xticklabels(labels, rotation=90)
        axes[i].tick_params(axis="x", which="minor", length=0)
        axes[i].tick_params(axis="x", which="major", length=0)
        # axes[i].set_xlabel('Input Length')

    axes[0].set_title("Environment 1 (Quadro RTX 6000 GPU)")
    # axes[1].set_title('Environment 2 (L4 GPU)')
    axes[1].set_title("Environment 2 (RTX 6000 Ada GPU)")

    axes[0].set_ylabel("Latency (s)")

    # set y-axis limit
    # axes[0].set_ylim(0, 2)
    # axes[1].set_ylim(0, 2)
    # axes[2].set_ylim(0, 2)

    # Add legends
    # axes[0].legend(ncol=2, loc='upper left')
    # axes[2].legend()

    plt.tight_layout()
    plt.savefig("/home/tian21/fiddler/asset/microbench.png")


def e2e():
    tokens_per_second, input_output_tokens = normal_e2e()
    for dataset in tokens_per_second.keys():
        plot_e2e(dataset, tokens_per_second, input_output_tokens, "./fig/")

e2e()
long_context()
beam()
# microbench()
