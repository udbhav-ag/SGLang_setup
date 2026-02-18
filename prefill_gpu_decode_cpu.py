#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _resolve_path(base_dir: Path, path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _append_override(cmd: list[str], flag: str, value):
    if value is None:
        return
    cmd.extend([flag, str(value)])


def _has_flag(cmd: list[str], flag: str) -> bool:
    if flag in cmd:
        return True
    return any(arg.startswith(flag + "=") for arg in cmd)


def _run_stage(stage_name: str, cmd: list[str], env: dict):
    print(f"\n=== {stage_name} ===")
    print(" ".join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, env=env, check=True)
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(
        description="Single-file pipeline: prefill on GPU, decode on CPU."
    )
    parser.add_argument("--prefill-config", default="prefill.yaml")
    parser.add_argument("--decode-config", default="decode.yaml")
    parser.add_argument("--prefill-script", default="prefill.py")
    parser.add_argument("--decode-script", default="decode.py")
    parser.add_argument("--kv-cache-dir", default="/Udbhav/cache_gpu")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prompt-filename", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-len", type=int, default=None)
    parser.add_argument("--prefill-cuda-graph-max-bs", type=int, default=None)
    parser.add_argument("--prefill-mem-fraction-static", type=float, default=None)
    parser.add_argument("--prefill-disable-cuda-graph", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    prefill_script = _resolve_path(base_dir, args.prefill_script)
    decode_script = _resolve_path(base_dir, args.decode_script)
    prefill_config = _resolve_path(base_dir, args.prefill_config)
    decode_config = _resolve_path(base_dir, args.decode_config)

    common_overrides: list[str] = []
    _append_override(common_overrides, "--model-path", args.model_path)
    _append_override(common_overrides, "--prompt-filename", args.prompt_filename)
    _append_override(common_overrides, "--batch-size", args.batch_size)
    _append_override(common_overrides, "--output-len", args.output_len)

    prefill_graph_max_bs = args.prefill_cuda_graph_max_bs
    if prefill_graph_max_bs is None:
        prefill_graph_max_bs = args.batch_size if args.batch_size is not None else 2

    prefill_cmd = [
        args.python,
        prefill_script,
        "--config",
        prefill_config,
        "--correctness-test",
        "--device",
        "cuda",
        *common_overrides,
    ]
    if args.prefill_disable_cuda_graph:
        prefill_cmd.append("--disable-cuda-graph")

    decode_cmd = [
        args.python,
        decode_script,
        "--config",
        decode_config,
        "--correctness-test",
        "--device",
        "cpu",
        "--attention-backend",
        "torch_native",
        *common_overrides,
    ]

    prefill_env = os.environ.copy()
    prefill_env["SGLANG_KV_CACHE_DIR"] = args.kv_cache_dir
    prefill_env.pop("SGLANG_USE_CPU_ENGINE", None)
    prefill_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    decode_env = os.environ.copy()
    decode_env["SGLANG_KV_CACHE_DIR"] = args.kv_cache_dir
    decode_env["SGLANG_USE_CPU_ENGINE"] = "1"
    decode_env["CUDA_VISIBLE_DEVICES"] = ""
    decode_env.setdefault("SGLANG_KV_LOAD_CUDA_ONLY", "0")

    try:
        prefill_time_s = _run_stage("Prefill on GPU", prefill_cmd, prefill_env)
    except subprocess.CalledProcessError:
        if _has_flag(prefill_cmd, "--disable-cuda-graph"):
            raise
        retry_cmd = list(prefill_cmd)
        retry_cmd.append("--disable-cuda-graph")
        if not _has_flag(retry_cmd, "--mem-fraction-static"):
            retry_cmd.extend(["--mem-fraction-static", "0.5"])
        print("\nPrefill failed. Retrying with CUDA graph disabled.")
        prefill_time_s = _run_stage(
            "Prefill on GPU (retry: disable cuda graph)", retry_cmd, prefill_env
        )

    print(f"GPU prefill time: {prefill_time_s:.3f}s")
    _run_stage("Decode on CPU", decode_cmd, decode_env)

    print("\nPipeline completed successfully.")
    print(f"Shared KV cache dir: {args.kv_cache_dir}")


if __name__ == "__main__":
    main()
