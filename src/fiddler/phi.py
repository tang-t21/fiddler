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
        self.model = transformers.PhimoeForCausalLM.from_pretrained(
            args.model,
            torch_dtype=self.dtype,
            device_map='auto',
            use_cache=True,
            attn_implementation="flash_attention_2"
        )
        self.device = self.model.device
        self.lm_head = self.model.lm_head
        self.model = self.model.model
        self.vocab_size = self.model.config.vocab_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.past_key_value = transformers.cache_utils.DynamicCache.from_legacy_cache()
        self.past_key_values_length = 0
        self.beam_width = args.beam_width
        self.torch_threads = args.torch_threads
        self.n_layer = len(self.model.layers)
        self.n_expert = len(self.model.layers[0].block_sparse_moe.experts)
        self.processed_tokens = 0
        self.expert_token_num = np.zeros((self.n_layer, self.n_expert), dtype=int)
    

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
                input_ids = output[:, -1].unsqueeze(0).view(-1, 1).to(self.device)
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
                    input_ids = output.to(self.device)
                else:
                    new_probs, output = torch.topk(logits, self.beam_width, dim=-1)
                    new_probs = self.initial_beam_tensor(new_probs)
                    output = self.initial_beam_tensor(output)
                    search_start = True
                    probs = probs * new_probs
                    input_ids = output[:, -1].flatten().view(-1, 1).to(self.device)
                # new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
                probs = probs / probs.sum(dim=-1, keepdim=True)
            # input_ids.shape: (batch_size, seq_len=1)

            position_ids = (
                torch.arange(
                    self.past_key_values_length,
                    self.past_key_values_length + 1,
                    dtype=torch.long,
                    device=self.device,
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
            self.expert_token_num
        )

    def tokenize(self, texts):
        input_ids = []
        for text in texts:
            encodings = self.tokenizer(text, return_tensors="pt")
            input_id = encodings.input_ids.to(self.device)
            for i in range(self.beam_width):
                input_ids.append(input_id[0])

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)

        position_ids = torch.arange(
            0, input_ids.shape[-1], dtype=torch.long, device=self.device
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
        inps = input_ids.to(self.device)
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
            inps = inps_residual.to(inps.device) + inps
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
            # routing_weights.shape: (batch_size*seq_len, 2)
            # selected_experts.shape: (batch_size*seq_len, 2)
            # activated_experts, counts = torch.unique(
            #     selected_experts, return_counts=True
            # )
            # for i in range(selected_experts.shape[0]):
            #     for j in range(selected_experts.shape[1]):
            #         self.expert_counts[i_layer][selected_experts[i, j]] += 1
            # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # print(f"Selection time:{(time.time()-start_time)*10**3}")
            # torch.cuda.synchronize()
            # self.selection_time.append((time.time() - start_time) * 10**6)
            for top_2 in selected_experts:
                for i in top_2:
                    self.expert_token_num[i_layer][i] += 1

            # intermediate variable to store the output of experts
            inps_after_experts = torch.zeros_like(inps, device=inps.device)
            experts = layer.block_sparse_moe.experts

      
            # baseline: do everything at GPU
            expert_mask = torch.nn.functional.one_hot(
                selected_experts, num_classes=self.n_expert
            ).permute(2, 1, 0)

            for i_expert in range(len(experts)):
                idx, top_2 = torch.where(expert_mask[i_expert])

                if top_2.shape[0] == 0:
                    # print(f"Expert {i_expert}: has no tokens")
                    continue

                # torch.cuda.synchronize()

                current_state = inps[None, top_2].reshape(-1, hidden_dim)
                current_state = experts[i_expert](current_state) * routing_weights[top_2, idx, None]
                inps_after_experts.index_add_(
                    0, top_2, current_state.to(inps.dtype)
                )

                    # end of one expert
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
