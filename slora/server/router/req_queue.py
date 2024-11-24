import uuid
import asyncio
import numpy as np
from typing import List
from ..io_struct import Batch, Req
from slora.utils.infer_utils import  calculate_time
import torch
from slora.utils.logging import print_with_timestamp


class ReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size) -> None:
        assert batch_max_tokens is not None
        
        self.max_total_tokens = max_total_tokens
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size

        self.waiting_req_list: List[Req] = []
        # (has_run_len, left_out_len)
        # self.cache_len_list = []
        print_with_timestamp(
            inside_func="ReqQueue.__init__",
            max_total_tokens=self.max_total_tokens,
            batch_max_tokens=self.batch_max_tokens,
            running_max_req_size=self.running_max_req_size,
        )
        
    def append(self, req):
        self.waiting_req_list.append(req)
        return
    
    def _init_cache_list(self, current_batch:Batch, lora_ranks):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                self.cache_len_list.append((req.input_len + len(req.output_ids),
                                           req.max_output_len - len(req.output_ids) - 1))
                if req.adapter_dir not in self.adapters:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req, lora_ranks):
        # lora_ranks is a dict: {'dummy-lora-13b-rank-64-0': 64, 'dummy-lora-13b-rank-32-0': 32}
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1)) # hard to analysis
        # sort by has_run_len in descending order
        self.cache_len_list.sort(key=lambda x: -x[1])
        if req.adapter_dir not in self.adapters:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        # [2048, 8, 8, 8]
        # [2048, 8]
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        
        # [2048, 8, 8, 8]
        # [2048, 8]
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0

        # [2048, 2056, 2064, 2072]
        # [2048, 2056]
        cum_run_len_array = np.cumsum(has_run_len_array)
        
        # [1, 2, 3, 4], the priority of the request
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        
        # [2048, 4112, 6192, 8288]
        # [2048, 4112]
        inter_need_max_token_num = (left_out_len_array * size_array + cum_run_len_array)
        
        # 8288
        # 4112
        need_max_token_num = inter_need_max_token_num.max()
        
        # assuming max_total_tokens = 8192, adapter_size = 64, running_max_req_size = 8
        # 8288 < 8192 - 64 and 4 <= 8
        # 4112 < 4096
        
        print_with_timestamp(
            inside_func="ReqQueue._can_add_new_req",
            to_check_req=req.__repr__,
            cache_len_list=self.cache_len_list,
            has_run_len_array=has_run_len_array,
            left_out_len_array=left_out_len_array,
            cum_run_len_array=cum_run_len_array,
            size_array=size_array,
            inter_need_max_token_num=inter_need_max_token_num,
            need_max_token_num=need_max_token_num,
            max_total_tokens=self.max_total_tokens,
            adapter_size=self.adapter_size,
            batch_max_tokens=self.batch_max_tokens,
            running_max_req_size=self.running_max_req_size,
        )

        if (need_max_token_num < self.max_total_tokens - self.adapter_size and
            len(self.cache_len_list) <= self.running_max_req_size
        ):
            print_with_timestamp(
                inside_func="ReqQueue._can_add_new_req",
                can_add_new_req=req.__repr__,
            )
            # nsys _can_add_new_req
            torch.cuda.nvtx.range_push("can add new req, need_max_token_num_{}".format(need_max_token_num))
            # nsys _can_add_new_req
            torch.cuda.nvtx.range_pop()
            return True
        else:
            print("=========> Cannot add new req, req block!!!!  ", req.__repr__)
            return False
    
    def update_counter(self, req):
        pass 

    # @calculate_time(show=True, min_cost_ms=0.1)
    def generate_new_batch(self, current_batch:Batch, lora_ranks: dict[str, int]):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.aborted:
                aborted_count += 1
                continue
            if (
                self._can_add_new_req(req, lora_ranks) and
                new_batch_total_tokens + req.input_len <= self.batch_max_tokens
            ):
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            print_with_timestamp(
                inside_func="ReqQueue.generate_new_batch",
                new_batch=new_batch,
                len_waiting_req_list=len(self.waiting_req_list),
            )
            # nsys generate_new_batch
            torch.cuda.nvtx.range_push("gen_new_batch_{}".format(len(new_batch.reqs)))
            # nsys generate_new_batch
            torch.cuda.nvtx.range_pop()
            return new_batch
        else:
            return None


    def next_batch(self):
        next_batch = []
        new_batch_total_tokens = 0
        for req in self.waiting_req_list:
            if req.aborted:
                continue
            if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                next_batch.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break
        if len(next_batch) > 0:
            next_batch = Batch(uuid.uuid4().hex, next_batch)
            print_with_timestamp(
                inside_func="ReqQueue.next_batch",
                next_batch=next_batch,
                len_waiting_req_list=len(self.waiting_req_list),
            )
            # nsys next_batch
            torch.cuda.nvtx.range_push("next_batch{}".format(len(next_batch.reqs)))
            # nsys next_batch
            torch.cuda.nvtx.range_pop()
            return next_batch
        else:
            return None


def main():
    # test 
    req_queue = ReqQueue(8192, 4096, 8)
    #TODO: add test case for the ReqQueue class