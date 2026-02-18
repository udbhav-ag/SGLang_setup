import tiktoken
import sys
import os
import torch
import time

from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

def count_tokens(text, model="cl100k_base"):  # or "cl100k_base" for Llama-3.1
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
    
    # From file
    with open("prompts/2.txt", "r") as f:
        text = f.read()
        
    text = text[:134000]    
    print(f"Tokens in prompt1.txt ({model}): {count_tokens(text, model)}")
    
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


def start_profile(profile_activities, profile_record_shapes=False, rank_print=print):
    """
    Abstracted function to start profiling based on profile_activities.
    Returns profiler object (or None).
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStart()
            rank_print("CUDA Profiler started (nsys will begin capturing)")
        except Exception as e:
            rank_print(f"Failed to start CUDA profiler: {e}")
        return None
    else:
        activities = []
        if "CPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CPU)
        if "GPU" in profile_activities:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        if activities:
            profiler = torch.profiler.profile(
                activities=activities,
                with_stack=True,
                record_shapes=profile_record_shapes,
            )
            profiler.start()
            return profiler
        return None
    
    


def stop_profile(
    profiler,
    profile_activities,
    rank_print=print,
    save_trace=False,
    trace_filename=None,
    stage=None,
):
    """
    Abstracted function to stop profiling based on profile_activities.
    Optionally saves trace results and prints completion messages.
    """
    if "CUDA_PROFILER" in profile_activities:
        try:
            torch.cuda.cudart().cudaProfilerStop()
            rank_print("CUDA Profiler stopped (nsys should dump traces)")
        except Exception as e:
            rank_print(f"Failed to stop CUDA profiler: {e}")
    elif profiler is not None:
        profiler.stop()

    if save_trace:
        if profiler is not None:
            if trace_filename:
                _save_profile_trace_results(profiler, trace_filename)
                stage_desc = f"for {stage}" if stage else ""
                rank_print(
                    f"torch profiler chrome trace {stage_desc} saved to {trace_filename}"
                )
        if "CUDA_PROFILER" in profile_activities:
            rank_print(f"CUDA profiler trace for {stage} completed")
            
            

def _get_torch_profiler_output_dir():
    return os.environ.get("SGLANG_TORCH_PROFILER_DIR", "/tmp")


def _create_torch_profiler_filename(
    profile_filename_prefix, batch_size, input_len, output_len, stage
):
    output_dir = _get_torch_profiler_output_dir()
    filename = f"{profile_filename_prefix}_batch{batch_size}_input{input_len}_output{output_len}_{stage}.trace.json.gz"
    return os.path.join(output_dir, filename)


def _save_profile_trace_results(profiler, filename):
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total"
        )
    )


def _ceil_align(x: int, align: int) -> int:
    if align <= 1:
        return x
    return ((x + align - 1) // align) * align


def _alloc_req_slot(req_to_token_pool, req) -> int:
    if isinstance(req_to_token_pool, HybridReqToTokenPool):
        req_pool_indices = req_to_token_pool.alloc(1, [req])
    else:
        req_pool_indices = req_to_token_pool.alloc(1)

    if req_pool_indices is None:
        raise RuntimeError("Failed to allocate request slot for KV restore.")

    return int(req_pool_indices[0])


def _alloc_cache_indices(token_to_kv_pool_allocator, num_tokens: int, page_size: int):
    alloc_tokens = _ceil_align(num_tokens, page_size)
    cache_indices = token_to_kv_pool_allocator.alloc(alloc_tokens)
    if cache_indices is None:
        raise RuntimeError(
            f"Failed to allocate {alloc_tokens} KV slots (requested {num_tokens})."
        )

    if alloc_tokens > num_tokens:
        token_to_kv_pool_allocator.free(cache_indices[num_tokens:])
        cache_indices = cache_indices[:num_tokens]

    return cache_indices


def save_req_kv_cache_to_disk(req, model_runner, output_path: str) -> str:
    """
    Persist one request's KV cache to disk.

    Notes:
    - Call this after a successful prefill/extend when req.req_pool_idx is valid.
    - The file can be restored only with a compatible model/configuration.
    """
    if req.req_pool_idx is None:
        raise ValueError("Request has no req_pool_idx. Run extend/prefill first.")

    num_cached_tokens = int(getattr(req, "kv_committed_len", 0))
    if num_cached_tokens <= 0 and getattr(req, "fill_ids", None) is not None:
        num_cached_tokens = len(req.fill_ids)
    if num_cached_tokens <= 0:
        num_cached_tokens = req.seqlen - 1
    if num_cached_tokens <= 0:
        raise ValueError("No cached tokens to persist.")

    token_indices = model_runner.req_to_token_pool.req_to_token[
        req.req_pool_idx, :num_cached_tokens
    ].to(torch.int64)

    kv_cache_cpu = model_runner.token_to_kv_pool_allocator.get_cpu_copy(token_indices)

    payload = {
        "rid": req.rid,
        "num_cached_tokens": int(num_cached_tokens),
        "kv_cache_cpu": kv_cache_cpu,
        "page_size": int(model_runner.server_args.page_size),
        "kv_cache_dtype": str(model_runner.kv_cache_dtype),
        "model_path": str(model_runner.server_args.model_path),
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def load_req_kv_cache_from_disk(
    req,
    model_runner,
    input_path: str,
    return_timing: bool = False,
    cuda_only: bool = False,
):
    """
    Restore one request's KV cache from disk.

    Behavior:
    - Allocates a req slot if req.req_pool_idx is missing.
    - Allocates KV slots, loads cached tensors, and writes req_to_token mapping.
    - Sets req.prefix_indices to the restored cache span.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"KV cache file not found: {input_path}")

    t_start = time.perf_counter()
    if cuda_only:
        if not model_runner.device.startswith("cuda"):
            raise RuntimeError(
                "cuda_only=True requires runtime device to be CUDA. "
                f"Got device={model_runner.device}"
            )
        map_location = "cuda"
    else:
        map_location = "cpu"
    payload = torch.load(input_path, map_location=map_location)
    t_after_disk = time.perf_counter()
    num_cached_tokens = int(payload["num_cached_tokens"])
    if num_cached_tokens <= 0:
        raise ValueError("Invalid num_cached_tokens in persisted KV payload.")

    t_mem_start = time.perf_counter()
    req_pool_idx = req.req_pool_idx
    if req_pool_idx is None:
        req_pool_idx = _alloc_req_slot(model_runner.req_to_token_pool, req)
        req.req_pool_idx = req_pool_idx

    cache_indices = _alloc_cache_indices(
        model_runner.token_to_kv_pool_allocator,
        num_cached_tokens,
        model_runner.server_args.page_size,
    )

    model_runner.token_to_kv_pool_allocator.load_cpu_copy(
        payload["kv_cache_cpu"], cache_indices
    )
    model_runner.req_to_token_pool.write(
        (req_pool_idx, slice(0, num_cached_tokens)), cache_indices.to(torch.int32)
    )

    req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
        req_pool_idx, :num_cached_tokens
    ].to(req.prefix_indices.dtype)
    req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

    t_end = time.perf_counter()
    if return_timing:
        timing = {
            "rid": req.rid,
            "disk_s": t_after_disk - t_start,
            "to_memory_s": t_end - t_mem_start,
            "total_s": t_end - t_start,
            "num_cached_tokens": num_cached_tokens,
        }
        return num_cached_tokens, timing
    return num_cached_tokens


def save_batch_kv_cache_to_disk(reqs, model_runner, output_dir: str) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    for req in reqs:
        output_path = os.path.join(output_dir, f"kv_req_{req.rid}.pt")
        saved_files.append(save_req_kv_cache_to_disk(req, model_runner, output_path))
    return saved_files


def load_batch_kv_cache_from_disk(
    reqs,
    model_runner,
    input_dir: str,
    return_timing: bool = False,
    cuda_only: bool = False,
):
    restored_tokens = []
    per_request_timings = []
    for req in reqs:
        input_path = os.path.join(input_dir, f"kv_req_{req.rid}.pt")
        if return_timing:
            restored_num_tokens, timing = load_req_kv_cache_from_disk(
                req,
                model_runner,
                input_path,
                return_timing=True,
                cuda_only=cuda_only,
            )
            restored_tokens.append(restored_num_tokens)
            per_request_timings.append(timing)
        else:
            restored_tokens.append(
                load_req_kv_cache_from_disk(
                    req, model_runner, input_path, cuda_only=cuda_only
                )
            )

    if return_timing:
        summary = {
            "num_requests": len(restored_tokens),
            "num_tokens": int(sum(restored_tokens)),
            "disk_s": sum(x["disk_s"] for x in per_request_timings),
            "to_memory_s": sum(x["to_memory_s"] for x in per_request_timings),
            "total_s": sum(x["total_s"] for x in per_request_timings),
            "per_request": per_request_timings,
        }
        return restored_tokens, summary
    return restored_tokens
