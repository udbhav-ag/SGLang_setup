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