# Mooncake

1. Source Build (Currently not working)
```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
bash dependencies.sh

## Compile
mkdir build
cd build
cmake ..
make -j
sudo make install # optional, make it ready to be used by vLLM/SGLang
```

2. When integrated with SGLang, the system conceptually consists of four key components: the master service, metadata service (Optional), store service (Optional), and the SGLang server. Among them, the master service and metadata service are responsible for object and metadata maintenance. The store service manages a contiguous memory segment that contributes to the distributed KV cache, making its memory accessible to both local and remote SGLang servers. Data transfer occurs directly between the store service and SGLang servers, bypassing the master service.

3. Normal Installation 
```bash
pip install mooncake-transfer-engine
```


4. Start Mooncake
```bash
 mooncake_master --rpc_port=50051 --config_path=mooncake_config.yaml --enable_http_metadata_server=1 -v=1 > mc.log 2>&1 

```