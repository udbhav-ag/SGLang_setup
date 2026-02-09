# PD Diss

1. Clone Specific Release 

```bash
git clone https://github.com/sgl-project/sglang.git

# Move into the directory
cd sglang

# Checkout the specific version
git checkout v0.5.7
```

2. Install Dependencies

```bash
pip install --upgrade pip
pip install -e "python"
sudo apt update
sudo apt install -y libnuma-dev
sudo apt install -y libnuma1
sudo apt install -y build-essential g++ pkg-config
sudo apt install -y libstdc++-13-dev
sudo apt install nvidia-cuda-toolkit
conda install -y -c conda-forge cmake
```


3. Conda setup

```bash
source "$HOME/Udbhav/miniconda3/etc/profile.d/conda.sh"
conda activate src_com
```

4. Server Start
```bash
 python3 -m sglang.launch_server --config config.yaml > sglang.log 2>&1 
```

3. Testing Command 
```bash
curl "http://127.0.0.1:30000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, my name is",
    "sampling_params": {"max_new_tokens": 16, "temperature": 0}
  }'
```

3. Using HiCacheFile as backend
Using HiCacheFiles to test for persistence of KV cache across runs, 
    1. Write Policy : `write-through`
    2. mem-fraction-static: 0.2 // to force the cache out of the GPU memory
    3. hicache-ratio: 1
```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HOME/Udbhav/source_compile/hicache_persistence"
```

4. 
```bash
Clear the JIT compilation cacheexport SGLANG_DEBUG=1
rm -rf /root/.cache/tvm-ffi/
```

## Tracing the Disk Reload issue?

1. `hicache_storage.py` -> Creates the Hicache controller and loads the cache
2. `hiradix_cache.py` -> 
    ```bash
    func -> prefetch_from_storage is responsible but not running due to conditions not met
    HELLOOOO WORLD FROM PREFETCH 
Prefetching tokens: 7
True
256
Cannot prefetch due to conditions not met

    6. This file runs and calls the functions successfully but

`cache_controller.py`
    ```bash
    [2026-02-05 07:24:35] Revoking prefetch for request dac9f11344814e66b041fa58ce4449fc due to insufficient hits (0).
[2026-02-05 07:24:35] Prefetch dac9f11344814e66b041fa58ce4449fc completed with 0 tokens
    ```

7. Flow 
When a request arrives after restart:

1. `prefetch_from_storage()` is called
2. `HiCacheController.prefetch()` queues the operation `cache_controller.py:761-777`
3. `prefetch_thread_func()` calls `_storage_hit_query()` which returns 0 hits `cache_controller.py:910-959`
  1. `_storage_hit_query`   -> Get no prefix_key # THIS IS THE ISSUE
4. Since 0 < prefetch_threshold (256), prefetch is revoked `cache_controller.py:936-943`
5. System generates new cache and tries to write it, triggering "already exists" messages `hicache_storage.py:249-251`



## Fix
Fixed this to accumulate the Prefix keys
```bash
prefix_keys = list(operation.prefix_keys) if operation.prefix_keys is not None else []
```

Now 
```bash
Aligned prefetch length: 25153
Sending prefetch operation d11d27318479422ca27c3f194df1eb4a with 25153 tokens.
Sending prefetch keys: ['a7d933a1375c236bbd43f144ddca8e85e17bc7f4eeacbb6dcbbccc1ea5ed3eba']
Sending Operation to prefetch buffer <sglang.srt.managers.cache_controller.PrefetchOperation object at 0x76d43ed18550>
Entered _storage_hit_query
Recieved Operation <sglang.srt.managers.cache_controller.PrefetchOperation object at 0x76d43ed18550>
Recieved Tokens to Fetch 25153
Prefix Keys Original ['a7d933a1375c236bbd43f144ddca8e85e17bc7f4eeacbb6dcbbccc1ea5ed3eba']
File : /home/ubuntu/Udbhav/v0.5.7/sglang/python/sglang/srt/managers/cache_controller.py
Prefix Keys Fetched ['a7d933a1375c236bbd43f144ddca8e85e17bc7f4eeacbb6dcbbccc1ea5ed3eba']

[2026-02-05 09:19:07] Prefetch d11d27318479422ca27c3f194df1eb4a completed with 0 tokens
[2026-02-05 09:19:08] Prefetching 25153 pages for request d11d27318479422ca27c3f194df1eb4a.
```