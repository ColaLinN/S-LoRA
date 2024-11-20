from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm
import random
from transformers import AutoTokenizer

class Request:
    def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time):
        self.req_id = req_id
        self.model_dir = model_dir 
        self.adapter_dir = adapter_dir
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.req_time = req_time

    
    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, " \
               f"prompt_len={self.prompt_len}, output_len={self.output_len}, " \
               f"req_time={self.req_time}"


def dummy_prompt(prompt_len):
    return "Hello " * prompt_len


def generate_requests(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed=42):
    print("generating requests", num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed)
    np.random.seed(seed)

    # tot_req 与 req_rate 和 duration 有关
    tot_req = int(req_rate * duration)

    # generate adapter id
    probs = np.random.power(alpha, tot_req)
    ind = (probs * num_adapters).astype(int)

    # 假设 tot_req 已定义
    alpha = 10.0  # Power distribution parameter
    input_lens = 1 - np.random.power(alpha, tot_req)
    input_lens = np.floor(input_lens * (1016 / input_lens.max()) + 8).astype(int)
    output_lens = 1 - np.random.power(alpha, tot_req)
    output_lens = np.floor(input_lens * (1016 / input_lens.max()) + 8).astype(int)

    # 对 input_lens 和 output_lens 进行计数
    input_counts = Counter(input_lens)
    output_counts = Counter(output_lens)

    # # 获取 input_lens 和 output_lens 中计数最多的前10个值和最少的后10个值
    # input_most_common = input_counts.most_common(10)  # 前10
    # input_least_common = input_counts.most_common()[:-11:-1]  # 后10
    # output_most_common = output_counts.most_common(10)  # 前10
    # output_least_common = output_counts.most_common()[:-11:-1]  # 后10
    # print("total req", tot_req)
    # print("Input Lens - Top 10 Most Common:", input_most_common)
    # print("Input Lens - Top 10 Least Common:", input_least_common)
    # print("Output Lens - Top 10 Most Common:", output_most_common)
    # print("Output Lens - Top 10 Least Common:", output_least_common)
    
    # original
    # input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    # output_lens = np.random.randint(output_range[0], output_range[1], tot_req)
    # return 

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        requests.append(Request(i, adapter_dirs[ind[i]][0], adapter_dirs[ind[i]][1],
                                dummy_prompt(input_lens[i]), int(input_lens[i]), int(output_lens[i]),
                                tic))
    return requests

def generate_requests_v2(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed=42):
    print("generating requests v2", 
                      "num_adapters", num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      "len_adapters", len(adapter_dirs), 
                      seed)
    np.random.seed(seed)

    # input_lens = [2040, 1024, 16, 16, 16, 16, 16, 16]
    # output_lens= [2040, 8, 8, 8, 8, 8, 8, 8]

    # input_lens = [16, 1024, 2040, 16, 16, 16, 16, 16]
    # output_lens= [2040, 8, 8, 8, 8, 8, 8, 8]
    
    # input_lens = [16, 1024, 2040, 16, 16, 16, 16, 16]
    # output_lens= [16, 2040, 8, 8, 8, 8, 8, 8]

    # input_lens = [16]
    # output_lens= [2046]
    
    # input_lens = [8, 8, 8, 8, 8, 8]
    # output_lens= [8, 1022, 8, 1022, 8, 2046]

    # start
    # input_lens = [8]
    # output_lens= [8]

    # input_lens = [1022]
    # output_lens= [1022]

    # input_lens = [8, 8, 8]
    # output_lens= [8, 8, 8]
    
    # input_lens = [8, 8, 8]
    # output_lens= [8, 1022, 8]

    # input_lens = [8, 8, 8]
    # output_lens= [1022, 1022, 1022]
    
    # input_lens = [8, 1022, 8]
    # output_lens= [8, 8, 8]
    
    # input_lens = [8, 2046, 8]
    # output_lens= [8, 8, 8]
    
    # input_lens = [8, 8, 8]
    # output_lens= [8, 2046, 8]

    # input_lens = [8, 8, 8, 8, 8, 8]
    # output_lens= [8, 1022, 8, 1022, 8, 2046]

    # generate adapter id
    # probs = np.random.power(alpha, tot_req)
    # ind = (probs * num_adapters).astype(int)
    # ind = [1]

    
    # input_lens = [8, 2046, 8, 1022]
    # output_lens= [8, 2046, 8, 1022]
    
    # input_lens = [8, 8, 8, 8]
    # output_lens= [8, 2046, 8, 1022]
    
    # input_lens = [8, 8, 8, 8]
    # output_lens= [8, 8, 8, 8]
    
    # input_lens = [8, 2046, 8, 1022]
    # output_lens= [2046, 8, 1022, 8]
    
    ind = [1, 1, 1, 1, 
           1, 1, 1, 1, 
           1, 1, 1]
    
    ind = [1, 2, 3, 4, 
           5, 6, 7, 8, 
           9, 10, 11]

    input_lens = [8, 8, 8, 8, 
                  8, 8, 8, 8, 
                  8, 8, 8]
    # output_lens= [8, 8, 8, 8, 
    #               512, 8, 512, 8, 
    #               8, 8, 8]
    output_lens= [8, 8, 8, 8, 
                  8, 8, 8, 8, 
                  8, 8, 8]

    print(ind)
    print(input_lens, output_lens)
    tot_req = int(len(input_lens))


    
    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    intervals = np.random.gamma(shape, scale, tot_req)
    
    # interval 应该固定，否则这个变量会影响分析 req_time
    
    print("construting req")
    for i in range(tot_req):
        tic += intervals[i]
        base_dir = adapter_dirs[ind[i]][0]
        adapter_dir = adapter_dirs[ind[i]][1]
        # adapter_dir = None
        print(adapter_dirs[ind[i]])
        requests.append(Request(
                                req_id=i, 
                                model_dir=base_dir, 
                                adapter_dir=adapter_dir, 
                                prompt=dummy_prompt(input_lens[i]), 
                                prompt_len=int(input_lens[i]), 
                                output_len=int(output_lens[i]), 
                                req_time=tic
                            )
                        )
    # for req in requests:
    #     print(req.__repr__)
    return requests

def get_real_requests(trace_file, req_rate, duration, base_model, adapter_dirs, input_range, output_range, seed=42):
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    conversations = downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range)
    model_mapping = generate_model_mapping(conversations, adapter_dirs)
    conversations = sort_and_rescale_by_req_time(conversations, duration)
    reqs = parse_into_req(base_model, conversations, model_mapping, tokenizer)
    return model_mapping.values(), reqs

# functions below are used to generate real requests
def downsample(json_file, req_rate, duration, tokenizer, input_range, output_range):
    with open(json_file, "r") as file:
       all_conversations = json.load(file)
    
    more_ratio = 2
    need_num = int(req_rate * duration)
    # sample a bit more than needed
    selected_indicies = np.random.choice(len(all_conversations), more_ratio * need_num, replace=False)
    downsampled_conversations = [all_conversations[idx] for idx in selected_indicies]
    for idx, conv in enumerate(downsampled_conversations):
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        if prompt_len >= input_range[1] or output_len >= output_range[1]:
            # to avoid OOM in some configurations
            downsampled_conversations.pop(idx)
    downsampled_conversations = downsampled_conversations[:need_num]
    print(f"Downsampled {len(downsampled_conversations)}")
    return downsampled_conversations 

def generate_model_mapping(conversations, adapter_dirs):
    model_mapping = {}
    num_ranks = [0] * len(adapter_dirs)
    for conv in conversations:
        model = conv["model"]
        if model not in model_mapping.keys():
            adapter_dir = random.choice(adapter_dirs)
            name = f"{adapter_dir}-{num_ranks[adapter_dirs.index(adapter_dir)]}"
            num_ranks[adapter_dirs.index(adapter_dir)] += 1
            model_mapping[model] = name
    print(model_mapping)
    return model_mapping

def sort_and_rescale_by_req_time(conversations, duration):
    # sort first
    sorted_conversations = sorted(conversations, key=lambda d: d['tstamp']) 
    interval_start = sorted_conversations[0]["tstamp"]
    interval_end = sorted_conversations[-1]["tstamp"]
    # print(f"sorted time step: {[s['tstamp'] for s in sorted_conversations]}")

    for conv in conversations:
        tstamp = conv["tstamp"]
        assert interval_start <= tstamp and tstamp <= interval_end
        rescaled_tstamp = (tstamp - interval_start) / (interval_end - interval_start) * duration
        conv["tstamp"] = rescaled_tstamp
    return sorted_conversations 

def parse_into_req(base_model, conversations, model_mapping, tokenizer):
    reqs = []
    for idx, conv in enumerate(tqdm(conversations, desc="parse into reqs")):
        model = conv["model"]
        name = model_mapping[model]
        # print(conv["conversation"][0]["content"])
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        
        req = Request(req_id=idx, model_dir=base_model, adapter_dir=name, 
              prompt=conv["conversation"][0]["content"], prompt_len=prompt_len,
              output_len=output_len, req_time=conv["tstamp"])
        reqs.append(req)
    # print(reqs)
    return reqs

