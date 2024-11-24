from collections import Counter
import json
import logging
from itertools import groupby
import numpy as np
from typing import List, Tuple, Any
from tqdm import tqdm
import random
from transformers import AutoTokenizer
from trace import Request, dummy_prompt

def generate_average_distribution(alpha, num1, num2, N):
    distribution = []
    for _ in range(N):
        if random.random() < alpha:
            distribution.append(num1)
        else:
            distribution.append(num2)
    return distribution

def generate_long_tail_requests(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed, total_req, hard_code_adapter_id):
    if seed is None:
        seed = 42
    # print("generating requests", num_adapters, alpha, req_rate, cv, duration,
    #                   input_range, output_range,
    #                   adapter_dirs, # (base_dir, adapter_dir)
    #                   seed)
    np.random.seed(seed)

    # tot_req 与 req_rate 和 duration 有关
    duration = total_req / req_rate

    # generate adapter id
    # probs = np.random.power(alpha, total_req)
    # ind = (probs * num_adapters).astype(int)
    ind = [hard_code_adapter_id for _ in range(total_req)]
    
    # original
    output_range = [input_range[0]+2, input_range[1]+2]
    # print("input_range", input_range)
    if input_range[0] != input_range[1]:
        input_lens = generate_average_distribution(alpha, input_range[0], input_range[1], total_req)
        output_lens = [input_lens[i] + 2 for i in range(total_req)]
        # output_lens = generate_average_distribution(alpha, output_range[0], output_range[1], total_req)
        print("input_lens", input_lens)
        print("output_lens", output_lens)
    else:
        input_lens = [input_range[0] for _ in range(total_req)] 
        output_lens = [output_range[0] for _ in range(total_req)] 
    # print("input_lens", input_lens)
    # print("output_lens", output_lens)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, total_req)
    
    for i in range(total_req):
        tic += intervals[i]
        requests.append(Request(i, 
                                adapter_dirs[ind[i]][0], 
                                adapter_dirs[ind[i]][1],
                                dummy_prompt(input_lens[i]), 
                                int(input_lens[i]), 
                                int(output_lens[i]),
                                tic))
    return requests