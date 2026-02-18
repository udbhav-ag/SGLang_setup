import argparse
import copy
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import sys
import time
from types import SimpleNamespace
from typing import Tuple


def _extract_cli_flag_value(argv, flag):
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _read_device_from_config(config_path):
    try:
        import yaml

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict):
            device = cfg.get("device")
            if isinstance(device, str):
                return device.strip().lower()
    except Exception:
        pass

    # Fallback parser for simple YAML key/value files.
    try:
        with open(config_path, "r") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                if key.strip() == "device":
                    return value.strip().strip("'\"").lower()
    except Exception:
        pass

    return None


def _bootstrap_cpu_engine_from_argv():
    # Must run before importing torch/sglang so backend dispatch picks CPU paths.
    argv = sys.argv[1:]
    device = _extract_cli_flag_value(argv, "--device")

    if device is None:
        config_path = _extract_cli_flag_value(argv, "--config")
        if config_path:
            device = _read_device_from_config(config_path)

    if isinstance(device, str) and device.strip().lower() == "cpu":
        os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


_bootstrap_cpu_engine_from_argv()

import numpy as np
import torch
import torch.distributed as dist


from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_dp_attn_mixin import prepare_mlp_sync_batch_raw
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    get_bool_env_var,
    is_cuda_alike,
    is_xpu,
    kill_process_tree,
    maybe_reindex_device_id,
    require_mlp_sync,
    require_mlp_tp_gather,
    set_gpu_proc_affinity,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

## 
from utils import *


def _read_prompts_from_file(prompt_file, rank_print):
    """Read custom prompts from the file specified by `--prompt-filename`."""
    if not prompt_file:
        return []
    if not os.path.exists(prompt_file):
        rank_print(
            f"Custom prompt file {prompt_file} not found. Using default inputs..."
        )
        return []
    with open(prompt_file, "r") as pf:
        return pf.readlines()


@dataclasses.dataclass
class BenchArgs:
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (1024,)
    output_len: Tuple[int] = (16,)
    prompt_filename: str = "prompts/1.txt"
    result_filename: str = "result.jsonl"
    correctness_test: bool = False
    # This is only used for correctness test
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_activities: Tuple[str] = ("CPU", "GPU")
    profile_stage: str = "all"
    profile_filename_prefix: str = "profile"

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name, help="A name for this benchmark run, used in logging and profiling file names.")
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=BenchArgs.batch_size
            , help="Batch size(s) to benchmark. Can specify multiple values for multiple runs."
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=BenchArgs.input_len
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=BenchArgs.output_len
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=BenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=BenchArgs.result_filename
        )
        parser.add_argument("--correctness-test", action="store_true")
        parser.add_argument("--cut-len", type=int, default=BenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=BenchArgs.log_decode_step,
            help="Log decode latency by step, default is set to zero to disable.",
        )
        parser.add_argument("--profile", action="store_true", help="Enable profiling.")
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record tensor shapes in profiling results.",
        )
        parser.add_argument(
            "--profile-activities",
            type=str,
            nargs="+",
            default=["CPU", "GPU"],
            choices=["CPU", "GPU", "CUDA_PROFILER"],
            help="Profiler activities: CPU, GPU, CUDA_PROFILER. If CPU/GPU, use torch profiler. If CUDA_PROFILER, use CUDA profiler.",
        )
        parser.add_argument(
            "--profile-stage",
            type=str,
            default=BenchArgs.profile_stage,
            choices=["all", "prefill", "decode"],
            help="Which stage to profile: all, prefill, or decode only.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=BenchArgs.profile_filename_prefix,
            help="Prefix of the profiling file names. The full profiling result file(s) be "
            '"[profile_filename_prefix]_batch[batch_size]_input[input_len]_output[output_len].trace.json.gz"',
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        # use the default value's type to cast the args into correct types.
        attrs = [(attr.name, type(attr.default)) for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: attr_type(getattr(args, attr)) for attr, attr_type in attrs}
        )


def load_model(server_args, port_args, gpu_id, tp_rank):
    
    suppress_other_loggers()
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    runtime_device_msg = f"runtime_device={model_runner.device}, gpu_id={gpu_id}, tp_rank={tp_rank}"
    if model_runner.device.startswith("cuda") and torch.cuda.is_available():
        runtime_device_msg += f", cuda_current_device={torch.cuda.current_device()}"
    rank_print(runtime_device_msg)
    rank_print(f"max_total_num_tokens={model_runner.max_total_num_tokens}")
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def prepare_inputs_for_correctness_test(bench_args, tokenizer, custom_prompts):
    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )
    input_ids = [tokenizer.encode(p) for p in prompts] # tokenize the prompts
    sampling_params = SamplingParams(                   # Simple SamplingParams with temperature 0 for deterministic output, and max_new_tokens set to the benchmark's output_len
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(prompts)):
        assert len(input_ids[i]) > bench_args.cut_len

        tmp_input_ids = input_ids[i][: bench_args.cut_len]
        req = Req(                                          # The input and output status of a request
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=tmp_input_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args, input_ids, reqs, model_runner
):
    for i in range(len(reqs)):
        req: Req = reqs[i]
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ].to(req.prefix_indices.dtype)
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    batch_size, input_len, custom_inputs=None
):
    input_ids = (
        custom_inputs
        if custom_inputs
        else np.random.randint(0, 10000, (batch_size, input_len), dtype=np.int32)
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=BenchArgs.output_len,
    )

    reqs = []
    for i in range(len(input_ids)):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(input_ids[i]),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        reqs.append(req)

    return reqs

def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, model_runner):
    if require_mlp_sync(model_runner.server_args):
        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(model_runner.server_args),
            disable_overlap_schedule=model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


class TreeCacheNamespace(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()
    
    
@torch.no_grad
def extend(reqs, model_runner):
    # Create dummy tree_cache for benchmarks (no prefix caching, just allocation)
    dummy_tree_cache = TreeCacheNamespace(
        page_size=model_runner.server_args.page_size,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )

    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad
def extend_and_maybe_persist_kv(
    reqs,
    model_runner,
    kv_save_dir: str = "/Udbhav/cache_gpu",
    rank_print=print,
):
    """
    Run prefill/extend once, and optionally persist KV cache to disk.

    Returns:
        next_token_ids, next_token_logits, batch, saved_files
    """
    next_token_ids, next_token_logits, batch = extend(reqs, model_runner)

    saved_files = []
    if kv_save_dir:
        saved_files = save_batch_kv_cache_to_disk(reqs, model_runner, kv_save_dir)
        rank_print(f"Saved KV cache for {len(saved_files)} request(s) to {kv_save_dir}")

    return next_token_ids, next_token_logits, batch, saved_files


# def load_persisted_kv_for_reqs(
#     reqs,
#     model_runner,
#     kv_load_dir: str,
#     rank_print=print,
# ):
#     """
#     Restore KV cache from disk for requests in `reqs`.

#     Returns:
#         List[int]: restored cached token count for each request.
#     """
#     restored_tokens = load_batch_kv_cache_from_disk(reqs, model_runner, kv_load_dir)
#     rank_print(
#         f"Loaded KV cache for {len(restored_tokens)} request(s) from {kv_load_dir}"
#     )
#     return restored_tokens


def correctness_test(
    server_args,
    port_args,
    bench_args,
    gpu_id,
    tp_rank,
):
    try:
        # Configure the logger
        configure_logger(server_args, prefix=f" TP{tp_rank}")
        rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

        # Load the model
        model_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)

        # Prepare inputs
        custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
        input_ids, reqs = prepare_inputs_for_correctness_test(
            bench_args, tokenizer, custom_prompts
        )
        rank_print(f"\n{input_ids=}\n")

        if bench_args.cut_len > 0:
            # Prefill
            next_token_ids, next_token_logits, batch = extend(reqs, model_runner)
            rank_print(f"prefill logits (first half): {next_token_logits} \n")

        # Prepare extend inputs
        reqs = prepare_extend_inputs_for_correctness_test(
            bench_args, input_ids, reqs, model_runner
        )

        # Extend (prefill w/ KV cache)
        next_token_ids, next_token_logits, batch, saved_files = (
            extend_and_maybe_persist_kv(reqs, model_runner)
        )
        rank_print(f"prefill logits (final): {next_token_logits} \n")
        print(f"Saved KV cache files: {saved_files}")
    finally:
        destroy_distributed_environment()

    # Decode
    # output_ids = [input_ids[i] + [next_token_ids[i]] for i in range(len(input_ids))]
    # for _ in range(bench_args.output_len[0] - 1):
    #     next_token_ids, _ = decode(next_token_ids, batch, model_runner)
    #     next_token_ids_list = next_token_ids.tolist()
    #     for i in range(len(reqs)):
    #         output_ids[i].append(next_token_ids_list[i])

    # # Print output texts
    # for i in range(len(reqs)):
    #     rank_print(f"========== Prompt {i} ==========")
    #     rank_print(tokenizer.decode(output_ids[i]), "\n")
    
    
def main(server_args, bench_args):
    server_args.cuda_graph_max_bs = max(bench_args.batch_size)

    _set_envs_and_config(server_args)

    if server_args.model_path:
        if bench_args.correctness_test:
            work_func = correctness_test
        # else:
            # work_func = latency_test
    else:
        raise ValueError(
            "Provide --model-path for running the tests or "
            "provide --result-filename for plotting the results"
        )

    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        work_func(server_args, port_args, bench_args, 0, 0)
    else:
        workers = []
        for tp_rank in range(server_args.tp_size):
            with maybe_reindex_device_id(tp_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=work_func,
                    args=(
                        server_args,
                        port_args,
                        bench_args,
                        gpu_id,
                        tp_rank,
                    ),
                )
                proc.start()
                workers.append(proc)

        for proc in workers:
            proc.join()

        proc.terminate()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)
    argv = sys.argv[1:]
    if "--config" in argv:
        from sglang.srt.server_args_config_parser import ConfigArgumentMerger

        argv = ConfigArgumentMerger(parser).merge_config_with_args(argv)
    args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)
