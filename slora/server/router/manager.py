import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import pickle
import time
import torch
import zmq
import zmq.asyncio
from typing import Dict, List, Optional

from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from slora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

from slora.server.input_params import InputParams
from slora.models.peft.lora_adapter import get_lora_config
from slora.server.router.profiler import AlphaModel, BetaModel
from slora.server.router.abort_req_queue import AbortReqQueue
from slora.server.router.cluster_req_queue import ClusterReqQueue
from slora.server.router.vtc_req_queue import VTCReqQueue
from slora.server.router.pets_req_queue import PETSReqQueue
from slora.server.router.peft_req_queue import PEFTReqQueue
from slora.utils.logging import print_with_timestamp



def get_scheduler(input_params, adapter_dirs):
    if input_params.scheduler == "vtc_fair":
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    else:
        raise Exception("unrecognized scheduler")


class RouterManager:

    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[None] = 0

        self.req_queue = get_scheduler(input_params, adapter_dirs)

        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)


    async def wait_to_model_ready(self):
        # nsys wait_to_model_ready
        torch.cuda.nvtx.range_push("wait_to_model_ready")
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            # nsys init_model
            torch.cuda.nvtx.range_push("init_model(rank_id_{})".format(rank_id))
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                ))
            # nsys init_model
            torch.cuda.nvtx.range_pop()

        await asyncio.gather(*init_model_ret)
        # nsys wait_to_model_ready
        torch.cuda.nvtx.range_pop()
        return
    
    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        # cache_dir = os.path.expanduser("~/.cache/slora")
        cache_dir = os.getenv("SLORA_CACHE_DIR")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir+"/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return


    def add_req(
        self,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        # nsys add_req
        torch.cuda.nvtx.range_push("add_req(prompt_len_{}_req_id_{})".format(len(prompt_ids), request_id))
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        # nsys add_req
        torch.cuda.nvtx.range_pop()
        return

    async def abort(self, request_id):
        # nsys abort
        torch.cuda.nvtx.range_push("abort(request_id_{})".format(request_id))
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        # nsys abort
        torch.cuda.nvtx.range_pop()
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:", self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num)
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self):
        """
        事件处理循环
        """
        # nsys _step
        torch.cuda.nvtx.range_push("_step")
        # 删除所有已经 finished 的 req
        if self.running_batch is None:
            # nsys running_batch is None
            torch.cuda.nvtx.range_push("running_batch is null")
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch

                if not self.input_params.no_lora:
                    # load adapters
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                
                # merge adapter to base model
                if self.input_params.scheduler == "peft":
                    torch.cuda.synchronize()
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].merge_adapter())
                    await asyncio.gather(*ret)
            
                # TODO: needs to be optimized
                torch.cuda.synchronize()
                print_with_timestamp(
                    inside_func="_step",
                    to_run_func="_prefill_batch",
                    running_batch=self.running_batch.__repr__(),
                )
                await self._prefill_batch(self.running_batch)
                await self._filter_runing_batch()
                self.has_wait_tokens = 0
            # nsys running_batch is None
            torch.cuda.nvtx.range_pop()
            return

        if self.has_wait_tokens < self.max_wait_tokens:
            # nsys has_wait_tokens < max_wait_tokens
            torch.cuda.nvtx.range_push("small_hwt_{},mwt_{}".format(self.has_wait_tokens, self.max_wait_tokens))
            self.stats_tool.count_output_tokens(self.running_batch)
            # prefetch
            if (not self.input_params.no_lora and
                self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                next_batch = self.req_queue.next_batch()
                if next_batch is not None:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(
                            next_batch.adapter_dirs, prefetch=True))
                    await asyncio.gather(*ret)
            print_with_timestamp(
                inside_func="_step",
                to_run_func="start _decode_batch",
                running_batch=self.running_batch.__repr__(),
            )
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()

            self.has_wait_tokens += 1
            # nsys has_wait_tokens < max_wait_tokens
            torch.cuda.nvtx.range_pop()
            return
        else:
            # nsys has_wait_tokens > max_wait_tokens
            torch.cuda.nvtx.range_push("large_hwt_{},mwt_{}".format(self.has_wait_tokens, self.max_wait_tokens))
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_mini_batch is not None:
                # nsys new_mini_batch is not None
                torch.cuda.nvtx.range_push("new_mini_batch_not_null")
                self.stats_tool.count_prompt_tokens(new_mini_batch)

                if not self.input_params.no_lora:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                print_with_timestamp(
                    inside_func="_step",
                    to_run_func="_decode_batch",
                    running_batch=self.running_batch.__repr__(),
                )
                await self._prefill_batch(new_mini_batch, minibatch=True)
                if not new_mini_batch.is_clear():
                    print_with_timestamp(
                        inside_func="_step",
                        to_run_func="_merge_batch",
                        running_batch=self.running_batch.__repr__(),
                        new_mini_batch=new_mini_batch.__repr__(),
                    )
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
                # nsys new_mini_batch_not_null
                torch.cuda.nvtx.range_pop()
            else:
                # nsys new_mini_batch is None
                torch.cuda.nvtx.range_push("new_mini_batch_null")
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
                # nsys new_mini_batch is None
                torch.cuda.nvtx.range_pop()
            # nsys has_wait_tokens > max_wait_tokens
            torch.cuda.nvtx.range_pop()
        # nsys _step
        torch.cuda.nvtx.range_pop()

    # @calculate_time(show=True, min_cost_ms=0.1)
    async def _init_batch(self, batch: Batch):
        # nsys _init_batch
        torch.cuda.nvtx.range_push("_init_batch")
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        # nsys _init_batch
        torch.cuda.nvtx.range_pop()
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    async def _prefill_batch(self, batch, minibatch=True):
        # nsys _prefill_batch
        torch.cuda.nvtx.range_push("_prefill_batch")
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        # nsys _prefill_batch
        torch.cuda.nvtx.range_pop()
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    async def _decode_batch(self, batch:Batch):
        # nsys _decode_batch
        torch.cuda.nvtx.range_push("_decode_batch")
        self.req_queue.update_counter(batch)
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        # nsys _decode_batch
        torch.cuda.nvtx.range_pop()
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    async def _filter_batch(self, batch: Batch):
        # nsys _filter_batch
        torch.cuda.nvtx.range_push("_filter_batch")
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    async def _merge_batch(self, batch1, batch2):
        # nsys _merge_batch
        torch.cuda.nvtx.range_push("_merge_batch")
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        # nsys _merge_batch
        torch.cuda.nvtx.range_pop()
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    async def _remove_batch(self, batch):
        # nsys _remove_batch
        torch.cuda.nvtx.range_push("_remove_batch")
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        # nsys _remove_batch
        torch.cuda.nvtx.range_pop()
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        # nsys _handle_finish_req
        torch.cuda.nvtx.range_push("_handle_finish_req")
        if has_new_finished_req:
            batch.filter_finished()

            # unmerge adapter from base model
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)

            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch)
        # nsys _handle_finish_req
        torch.cuda.nvtx.range_pop()
        return

    async def _filter_runing_batch(self):
        # nsys _filter_runing_batch
        torch.cuda.nvtx.range_push("_filter_runing_batch")
        if self.running_batch is not None and self.running_batch.is_clear():
            if not self.input_params.no_lora:
                # offload model and adapters
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters())
                await asyncio.gather(*ret)
            print_with_timestamp(
                inside_func="_filter_runing_batch",
                to_run_func="running_batch=null",
                running_batch=self.running_batch.__repr__(),
            )
            self.running_batch = None
        # nsys _filter_runing_batch
        torch.cuda.nvtx.range_pop()
        return
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        # nsys _add_token_id_to_req
        torch.cuda.nvtx.range_push("_add_token_id_to_req")
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        # nsys _add_token_id_to_req
        torch.cuda.nvtx.range_pop()
        return
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        # nsys _send_to_detokenization_proc
        torch.cuda.nvtx.range_push("_send_to_detokenization_proc")
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
    
        self.send_to_detokenization.send_pyobj(batch_out)
        # nsys _send_to_detokenization_proc
        torch.cuda.nvtx.range_pop()
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                # nsys loop_for_netio_req req
                torch.cuda.nvtx.range_push("loop_for_netio_req recv_req")
                adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                print_with_timestamp(
                    inside_func="loop_for_netio_req",
                    to_run_func="add_req",
                    running_batch=f"{adapter_dir} len_propmt_ids {len(prompt_ids)} sampling_params {sampling_params} request_id {request_id}",
                )
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)
                # nsys loop_for_netio_req req
                torch.cuda.nvtx.range_pop()
            elif isinstance(recv_req, AbortReq):
                # nsys loop_for_netio_req abort_req
                torch.cuda.nvtx.range_push("abort_req")
                abort_req = recv_req
                request_id = abort_req.req_id
                print_with_timestamp(
                    inside_func="loop_for_netio_req",
                    to_run_func="abort",
                    running_batch=f"exp_debugging|loop_for_netio_req try to abort {request_id}",
                )
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
                # nsys loop_for_netio_req abort_req
                torch.cuda.nvtx.range_pop()
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        # nsys clean_up
        torch.cuda.nvtx.range_push("clean_up")
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                              )

    try:
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval,
        )
    
        asyncio.run(router.wait_to_model_ready())
        if input_params.profile:
            asyncio.run(router.profile_prefill())
        if input_params.scheduler == "pets" and input_params.profile:
            router.req_queue.alpha = router.alpha_model
            router.req_queue.beta = router.beta_model
        elif input_params.scheduler == "pets":
            # loading from file
            # cache_dir = os.path.expanduser("~/.cache/slora")
            cache_dir = os.getenv("SLORA_CACHE_DIR")
            router.req_queue.alpha = AlphaModel.from_file(cache_dir+"/profile_results.pkl")
            router.req_queue.beta = BetaModel.from_file(cache_dir+"/profile_results.pkl")
    
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
