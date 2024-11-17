import copy
import concurrent.futures
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import transformers
# from bfloat16_expert import cpu_expert
from transformers.models.phimoe.modeling_phimoe import sparsemixer

class FiddlerPhi:
    def __init__(self, args):
        self.dtype = torch.bfloat16
        # kwargs = {"use_flash_attention_2": True}
        self.dev = torch.device("cuda:0")
        self.model = transformers.PhimoeForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            device_map="cpu",
            use_cache=True,
            attn_implementation = 'eager'
        )
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.vocab_size = self.model.config.vocab_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.beam_width = args.beam_width
        self.torch_threads = args.torch_threads
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
        self.processed_tokens = 0
        self.expert_token_num = np.zeros((self.n_layer, self.n_expert), dtype=int)

        self.cpu_offload = args.cpu_offload
        self.cpu_layer_num = []

        self.torch_threads = args.torch_threads
        self.cpp_threads = args.cpp_threads

        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.cpu_expert_time = []
        self.attention_time = []
        self.selection_time = []
        self.gpu_expert_time = []
        self.search_config_time = []
        self.one_token_time = []
        self.expert_pattern = []
        self.expert_counts = np.zeros(self.n_layer * self.n_expert, dtype=int)

        # self.cpu_experts = [[] for i in range(self.n_layer)]
        # self.cpu_experts = [[] for i in range(self.n_layer)]
        # self.init_cpu_expert()
        # self.test_cpu_expert()
        # self.test_cpu_expert()
        self.gpu_latency = np.mean(self.expert_gpu(n_expert=1, batch_size=1)) * 10**3
        self.copy_latency = np.mean(self.weight_copy()) * 10**3
        self.cpu_latency = np.mean(self.expert_cpu(1, 1)) * 10**3
        # self.gpu_latency = 0.46
        # self.copy_latency = 14.14
        # self.cpu_latency = 2.81

        print(f"CPU latency: {self.cpu_latency:.2f} ms")
        print(f"Copy latency: {self.copy_latency:.2f} ms")
        print(f"GPU latency: {self.gpu_latency:.2f} ms")
        # self.cpu_latency = 7
        # self.copy_latency = 40
        # self.gpu_latency = 0.6
        # self.init_attention()

        self.bring_non_expert_to_gpu()
        self.non_expert_alloc_mem = torch.cuda.memory_allocated(self.dev)
        print(f"Non-expert memory: {self.non_expert_alloc_mem}")
        self.default_max_len = 128
        # 0: CPU, 1: GPU
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        n_expert_on_gpu = self.calc_n_expert_on_gpu(self.default_max_len)
        print(
            f"Number of experts on GPU: {n_expert_on_gpu}/{self.n_layer * self.n_expert}"
        )
        self.popular_experts = []
        with open("/home/cc/fiddler/src/fiddler/expert-popularity-phi.txt", "r") as file:
            for line in file.readlines():
                layer_id = int(line.split(",")[0].strip('('))
                expert_id = int(line.split(",")[1].strip(')'))
                self.popular_experts.append((layer_id,expert_id))
        self.set_expert_loc(n_expert_on_gpu, self.popular_experts)
        # print(self.expert_loc)

        self.bring_expert_to_gpu()
        # self.cpu_experts = [[] for i in range(self.n_layer)]
        # self.init_cpu_expert()
        self.pin_expert_in_cpu()
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        print(f"Total memory: {total_mem//1024**3} GB")
        print(
            f"Total memory allocated: {torch.cuda.memory_allocated(self.dev)//1024**3} GB"
        )
        print("Free GPU memory:", torch.cuda.memory_reserved(self.dev) // 1024**3, "GB")
        print("Model is ready.")
    
    def weight_copy(self):
        """Time to copy weights of an expert"""
        torch.set_num_threads(self.torch_threads)
        ret_time = []

        expert_placeholder = copy.deepcopy(
            self.model.layers[0].block_sparse_moe.experts[0]
        ).to(self.dev)
        for i in range(self.n_layer):
            self.model.layers[i].block_sparse_moe.experts[0].to("cpu")
            for name in ["w1", "w2", "w3"]:
                w = getattr(self.model.layers[i].block_sparse_moe.experts[0], name)
                src_weight_data_tensor = w.weight.data
                pinned = src_weight_data_tensor.pin_memory()
                w.weight.data = pinned
            torch.cuda.synchronize()
            tick = time.time()
            # expert_placeholder.load_state_dict(
            #     model.model.layers[i].block_sparse_moe.experts[0].state_dict()
            # )
            for name in ["w1", "w2", "w3"]:
                dst = getattr(expert_placeholder, name).weight.data
                src = getattr(
                    self.model.layers[i].block_sparse_moe.experts[0], name
                ).weight.data
                dst.copy_(src)
            torch.cuda.synchronize()
            ret_time.append(time.time() - tick)
            self.model.layers[i].block_sparse_moe.experts[0].to("cpu")
        return np.array(ret_time)

    def expert_gpu(self, n_expert=1, batch_size=1):
        """Time to execute an expert at GPU"""
        torch.set_num_threads(self.torch_threads)
        ret_time = []

        # warm up
        self.model.layers[0].block_sparse_moe.experts[7].to(self.dev)
        inps = torch.randn((batch_size, 4096), dtype=self.dtype, device=self.dev)
        weights = torch.ones((batch_size, 1), dtype=self.dtype, device=self.dev)
        inps = self.model.layers[0].block_sparse_moe.experts[7](inps) *weights
        self.model.layers[0].block_sparse_moe.experts[7].to("cpu")
        del inps, weights
        torch.cuda.synchronize()

        for i in range(self.n_layer):
            for j in range(n_expert):
                self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)
                inps = torch.randn(
                    (batch_size, 4096), dtype=self.dtype, device=self.dev
                )
                weights = torch.randn(
                    (batch_size, 1), dtype=self.dtype, device=self.dev
                )
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                tick = time.time()
                inps = self.model.layers[i].block_sparse_moe.experts[j](inps) *weights
                torch.cuda.synchronize()
                ret_time.append(time.time() - tick)
                self.model.layers[i].block_sparse_moe.experts[j].to("cpu")
                del inps, weights
        return np.array(ret_time)

    def expert_cpu(self, n_expert=1, batch_size=1, multithreading=False):
        """Time to execute an expert at CPU"""
        torch.set_num_threads(self.torch_threads)
        ret_time = []
        # warm up
        self.model.layers[0].block_sparse_moe.experts[7].to("cpu")
        inps = torch.randn((batch_size, 4096), dtype=self.dtype, device="cpu")
        weights = torch.ones((batch_size, 1), dtype=self.dtype, device="cpu")
        torch.cuda.synchronize()
        tick = time.time()
        inps = self.run_expert_at_cpu(0, 7, inps, weights)
        del inps, weights
        torch.cuda.synchronize()

        for i in range(self.n_layer):
            for j in range(n_expert):
                self.model.layers[i].block_sparse_moe.experts[j].to("cpu")
                inps = torch.randn((batch_size, 4096), dtype=self.dtype, device="cpu")
                weights = torch.randn((batch_size, 1), dtype=self.dtype, device="cpu")
                torch.cuda.synchronize()
                tick = time.time()
                inps = self.run_expert_at_cpu(i, j, inps, weights)
                torch.cuda.synchronize()
                ret_time.append(time.time() - tick)
                del inps, weights
        return np.array(ret_time)
    
    def set_expert_loc(self, n_expert_on_gpu, popular_experts=None):
        """Set the location of experts"""
        for i in range(n_expert_on_gpu):
            i_layer, i_expert = popular_experts[i]
            self.expert_loc[i_layer, i_expert] = 1

    def bring_non_expert_to_gpu(self):
        """Bring non-expert layers to GPU"""
        self.lm_head.to(self.dev)
        self.model.embed_tokens.to(self.dev)
        self.model.norm.to(self.dev)
        for i in range(len(self.model.layers)):
            self.model.layers[i].self_attn.to(self.dev)
            self.model.layers[i].input_layernorm.to(self.dev)
            self.model.layers[i].block_sparse_moe.gate.to(self.dev)
            self.model.layers[i].post_attention_layernorm.to(self.dev)
            # only model.layers[i].block_sparse_moe.experts is on CPU
    
    def bring_expert_to_gpu(self):
        """Bring part of expert layers to GPU"""
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if self.is_expert_in_gpu(i, j):
                    self.model.layers[i].block_sparse_moe.experts[j].to(self.dev)
                else:
                    self.model.layers[i].block_sparse_moe.experts[j].to("cpu")
    
    def pin_expert_in_cpu(self):
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                if not self.is_expert_in_gpu(i, j):
                    for name in ["w1", "w2", "w3"]:
                        w = getattr(
                            self.model.layers[i].block_sparse_moe.experts[j], name
                        )
                        src_weight_data_tensor = w.weight.data
                        pinned = src_weight_data_tensor.pin_memory()
                        w.weight.data = pinned

    def is_expert_in_gpu(self, i_layer, i_expert):
        """Determine if the expert is in GPU"""
        return self.expert_loc[i_layer, i_expert] == 1


    def calc_n_expert_on_gpu(self, max_len):
        """Get the number of experts that we can put on GPU"""
        # get the number of parameters of one expert
        n_param = sum(
            p.numel()
            for p in self.model.layers[0].block_sparse_moe.experts[0].parameters()
        )
        # get the amount of free memory on GPU
        total_mem = torch.cuda.get_device_properties(self.dev).total_memory
        kv_cache_mem = self.n_layer * max_len * self.model.config.hidden_size * 2 * 2
        attn_weight_mem = max_len**2 * 2 * 32
        free_mem = total_mem*0.98 - self.non_expert_alloc_mem - kv_cache_mem - attn_weight_mem*2
        return int((free_mem) // (n_param * 2))

    def initial_beam_tensor(self, input_tensor):
        # transfer tensor of shape (batch_size*beam_width, seq_len, beam_width) to (batch_size*beam_width, 1) properly
        assert input_tensor.shape[-1] == self.beam_width
        input_tensor = input_tensor[:, -1]
        row_idx = torch.tensor(
            [
                i * self.beam_width
                for i in range(input_tensor.shape[0] // self.beam_width)
            ]
        )
        output_tensor = input_tensor[row_idx].view(-1, 1)
        return output_tensor

    def clear_cache(self):
        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0

    def reset_expert_loc(self,max_len):
        n_expert_on_gpu = self.calc_n_expert_on_gpu(max_len)
        print(f'Number of experts on GPU:{n_expert_on_gpu}/{self.n_layer*self.n_expert}')
        self.expert_loc = np.zeros((self.n_layer, self.n_expert), dtype=int)
        self.set_expert_loc(n_expert_on_gpu,self.popular_experts)
        self.clear_cache()
        self.bring_expert_to_gpu()
        self.pin_expert_in_cpu()

    def generate(
        self,
        texts=None,
        output_token=20,
        input_token=None,
        input_ids=None,
        beam_width=1,
        verbose=False,
    ):
        torch.set_num_threads(self.torch_threads)
        self.clear_cache()
        # input_ids.shape: (batch_size, seq_len)
        # position_ids.shape: (1,seq_len)
        self.beam_width = beam_width
        input_ids, position_ids = self.tokenize(texts)

        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            position_ids = position_ids[:, :input_token]
        self.processed_tokens += input_ids.shape[-1] + output_token
        is_decode = False
        prefill_time, decode_time = 0, 0
        decode_strings = ["" for _ in range(input_ids.shape[0])]
        search_start = False
        probs = torch.full((input_ids.shape[0], 1), 1.0)
        tick = time.time()
        for i_token in range(output_token):
            start_time = time.time()
            if is_decode:
                for i in range(input_ids.shape[0]):
                    decode_strings[i] += " " + self.tokenizer.decode(input_ids[i, :])
                    # print("--------------------")
                    # print(f"beam[{i}]: {decode_strings[i]}")

            logits = self.phi_forward(
                input_ids,
                position_ids,
                is_decode,
            )

            logits = logits.to("cpu")
            # logits.shape: (batch_size, seq_len, vocab_size)

            # normalize logits
            logits = F.softmax(logits, dim=-1)
            self.past_key_values_length += logits.shape[1]
            # greedy search:
            if self.beam_width == 1:
                output = torch.argmax(logits, dim=-1)
                input_ids = output[:, -1].unsqueeze(0).view(-1, 1).to(self.dev)
            else:
                # beam_search:

                if search_start:
                    new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                    new_probs = new_probs[:, -1].flatten().view(-1, 1)
                    for i in range(probs.shape[0]):
                        for j in range(probs.shape[0]):
                            new_probs[i * self.beam_width + j] *= probs[i]
                    topk_probs, topk_idx = torch.topk(new_probs, self.beam_width, dim=0)
                    probs = topk_probs
                    output = output.flatten().view(-1, 1)[
                        topk_idx.view(-1, self.beam_width).flatten()
                    ]
                    # print(output)
                    # exit(0)
                    input_ids = output.to(self.dev)
                else:
                    new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                    new_probs = self.initial_beam_tensor(new_probs)
                    output = self.initial_beam_tensor(output)
                    search_start = True
                    probs = probs * new_probs
                    input_ids = output[:, -1].flatten().view(-1, 1).to(self.dev)
                # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            # input_ids.shape: (batch_size, seq_len=1)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.dev,
                )
                .unsqueeze(0)
                .view(-1, 1)
            )
            # position_ids.shape: (1, 1)
            if not is_decode:
                torch.cuda.synchronize()
                prefill_time += time.time() - tick
                tick = time.time()
            is_decode = True
        torch.cuda.synchronize()
        decode_time = time.time() - tick
        if verbose:
            if self.beam_width == 1:
                for i in range(input_ids.shape[0]):
                    print("--------------------")
                    print(f"Input: {texts[i]}")
                    print(f"Output: {decode_strings[i]}")
            else:
                probs = probs.view(-1, self.beam_width)
                max_ids = torch.argmax(probs, dim=-1)
                for i in range(max_ids.shape[0]):
                    print("--------------------")
                    print(f"Input: {texts[i]}")
                    print(f"Output: {decode_strings[i * self.beam_width + max_ids[i]]}")

        return (
            prefill_time,
            decode_time,
            self.cnt_expert_hit/self.cnt_expert_all
        )

    def tokenize(self, texts):
        input_ids = []
        for text in texts:
            encodings = self.tokenizer(text, return_tensors="pt")
            input_id = encodings.input_ids.to(self.dev)
            for i in range(self.beam_width):
                input_ids.append(input_id[0])

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.dev)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.dev
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.shape[-1])

        return input_ids, position_ids

    @torch.no_grad()
    def phi_forward(
        self,
        input_ids,
        position_ids,
        is_decode,
    ):
        hidden_dim = self.model.config.hidden_size
        inps = input_ids.to(self.dev)
        inps = self.model.embed_tokens(inps)

        past_seen_tokens = self.past_key_value.get_seq_length() if self.past_key_value is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inps.shape[1], device=inps.device
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.model._update_causal_mask(
            None, inps, cache_position, self.past_key_value, self.model.config.output_attentions
        )
        position_embeddings = self.model.rotary_emb(inps, seq_len=cache_position[-1] + 1)
        # cpu_layer_num = 0
        # outliner_num = 0
        # outliners = []

        for i_layer, layer in enumerate(self.model.layers):
            # print(
            #     f"Layer {i_layer}: {torch.cuda.memory_allocated(self.dev)/1024**3} GB"
            # )
            original_inps_shape = inps.shape
            inps_residual = inps
            inps = layer.input_layernorm(inps)
            # print(f"{torch.cuda.memory_allocated(self.dev)/1024**3} GB")
            inps, self_attn_weights, present_key_value = layer.self_attn(
                inps,
                position_ids=position_ids,
                past_key_value=self.past_key_value,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                use_cache=True,
            )

            # exit(0)
            # inps.shape: (batch_size, seq_len/token_num, embed_dim)
            inps = inps_residual + inps
            inps_residual = inps
            inps = layer.post_attention_layernorm(inps)
            # torch.cuda.synchronize()
            # self.attention_time.append((time.time() - start_time) * 10**6)
            inps = inps.view(-1, hidden_dim)
            # start_time = time.time()
            # print(f"Attention time:{(time.time()-start_time)*10**3}")
            # inps.shape: (batch_size*seq_len*embed_dim/hidden_dim, hidden_dim)
            router_logits = layer.block_sparse_moe.gate(inps)
            routing_weights = F.softmax(router_logits, dim=1)
            # routing_weights.shape: (batch_size*seq_len, num_experts)
            routing_weights, selected_experts = sparsemixer(
                router_logits,
                jitter_eps=layer.block_sparse_moe.router_jitter_noise,
                training=layer.block_sparse_moe.training,
            )
            inps_after_experts = torch.zeros_like(inps, device=inps.device)
            experts = layer.block_sparse_moe.experts
            if self.cpu_offload == 0:
                # baseline: do everything at GPU
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                for i_expert in range(len(experts)):
                    is_cuda = self.is_expert_in_gpu(i_layer, i_expert)
                    idx, top_2 = torch.where(expert_mask[i_expert])

                    if top_2.shape[0] == 0:
                        # print(f"Expert {i_expert}: has no tokens")
                        continue

                    # torch.cuda.synchronize()

                    current_state = inps[None, top_2].reshape(-1, hidden_dim)
                    if not is_cuda:
                        self.expert_placeholder.load_state_dict(
                            experts[i_expert].state_dict()
                        )
                        current_state = self.expert_placeholder(current_state) * routing_weights[top_2, idx, None]
                    else:
                        current_state = experts[i_expert](current_state) *routing_weights[top_2, idx, None]

                    inps_after_experts.index_add_(
                        0, top_2, current_state.to(inps.dtype)
                    )

                    if not is_cuda:
                        experts[i_expert] = experts[i_expert].to("cpu")

                    # end of one expert

            else:
                # prefill stage with offloading
                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=self.n_expert
                ).permute(2, 1, 0)

                # first, calculate the number of tokens for each expert
                idxs, top_2s = [], []
                # cost_per_expert = np.zeros(
                #     (len(experts), 2), dtype=float
                # )  # 0: CPU, 1: GPU
                # hit_cnt = self.cnt_expert_hit
                cpu_experts = []
                gpu_experts = []
                for i_expert in range(len(experts)):
                    idx, top_2 = torch.where(expert_mask[i_expert])
                    idxs.append(idx)
                    top_2s.append(top_2)
                    # expected latency at CPU: number of token * cost_at_cpu
                    # expected latency at GPU: cost_at_gpu (constant)
                    cpu_cost = self.cpu_latency * top_2.shape[0]
                    gpu_cost = self.copy_latency + self.gpu_latency
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        # if the expert is in GPU, the latency at GPU is
                        # approximately 0
                        gpu_cost = self.gpu_latency
                        self.cnt_expert_hit += top_2.shape[0]
                    self.cnt_expert_all += top_2.shape[0]
                    if cpu_cost <= gpu_cost:
                        cpu_experts.append(i_expert)
                    else:
                        gpu_experts.append(i_expert)
                # print("hit number of this layer:", self.cnt_expert_hit - hit_cnt)
                # print("Number of tokens for each expert:", expert_tokens)

                # second, partition experts processing between CPU and GPU so that we can minimize:
                # max(sum of cost at CPU, sum of cost at GPU)
                # print(cpu_experts, gpu_experts)
                for i_expert in gpu_experts:
                    top_2 = top_2s[i_expert]
                    if top_2.shape[-1] == 0:
                        continue
                    idx = idxs[i_expert]
                    current_state = inps[None, top_2].reshape(-1, hidden_dim)
                    if self.is_expert_in_gpu(i_layer, i_expert):
                        current_state = experts[i_expert](current_state) * routing_weights[top_2, idx, None]
                    else:
                        # self.expert_placeholder.load_state_dict(
                        #     experts[i_expert].state_dict()
                        # )
                        for name in ["w1", "w2", "w3"]:
                            dst = getattr(self.expert_placeholder, name).weight.data
                            src = getattr(experts[i_expert], name).weight.data
                            dst.copy_(src)
                        current_state = self.expert_placeholder(current_state) * routing_weights[top_2, idx, None]
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

                for i_expert in cpu_experts:
                    top_2 = top_2s[i_expert]
                    if top_2.shape[-1] == 0:
                        continue
                    idx = idxs[i_expert]
                    current_state = inps[None, top_2].reshape(-1, hidden_dim)
                    current_state = self.run_expert_at_cpu(
                        i_layer,
                        i_expert,
                        current_state.to("cpu"),
                        routing_weights[top_2, idx, None].to("cpu"),
                    )
                    inps_after_experts.index_add_(
                        0,
                        top_2s[i_expert].to(self.dev, non_blocking=True),
                        current_state.to(self.dev, non_blocking=True),
                    )

            # addition because there's residual connection over moe layer
            inps = inps_residual + inps_after_experts.reshape(original_inps_shape)
            # layer_time = time.time() - layer_start
            # print(f"Layer time: {layer_time}")

            # end of one layer

        # self.cpu_layer_num.append(cpu_layer_num)
        # self.outliner_nums.append(outliner_num)
        # self.outliners.extend(outliners)
        inps = self.model.norm(inps)
        lm_logis = self.lm_head(inps)
        self.past_key_value = present_key_value

        return lm_logis

    def run_expert_at_cpu(self, i_layer, i_expert, inps, routing_weights):
        """Run the expert at CPU"""
        return self.model.layers[i_layer].block_sparse_moe.experts[i_expert](inps) *routing_weights

    def write_popular_experts(self, filename):
        popular_experts = []
        for i in range(self.n_layer):
            for j in range(self.n_expert):
                popular_experts.append((i, j, self.expert_token_num[i][j]))
        popular_experts.sort(key=lambda x: x[2], reverse=True)
        with open(filename, "w") as f:
            for i, j, hit_num in popular_experts:
                f.write(f"({i,j}),{hit_num}/{self.processed_tokens}\n")

    def reset_popular_experts(self):
        self.processed_tokens = 0
        self.expert_token_num = np.zeros((self.n_layer, self.n_expert), dtype=int)
