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

sudo apt install -y \
  libibverbs1 librdmacm1 \
  libibverbs-dev librdmacm-dev \
  rdma-core \
  libnuma-dev \
  libaio-dev \
  libssl-dev \
  build-essential \
  cmake \
  pkg-config
sudo ldconfig
pip install mooncake-transfer-engine
```


4. Start Mooncake
```bash
 mooncake_master --rpc_port=50051 --config_path=mooncake_config.yaml --enable_http_metadata_server=1 -v=1 > mc.log 2>&1 

```

5. IMPORTANT
If you configure the SGLang server with a non-zero MOONCAKE_GLOBAL_SEGMENT_SIZE or global_segment_size, the SGLang server itself will act as a store service and contribute its memory to the distributed pool. However, if the SGLang server contributes memory directly, any KV tensors stored in that memory will be lost when the SGLang process exits. To ensure persistence across SGLang restarts, it's recommended to run a separate mooncake_store_service and configure the SGLang server's global_segment_size to 0,


# SOURCE COMPILE
firstly simple cloning of mooncake : 
```
git clone https://github.com/kvcache-ai/Mooncake.git —recursive
```

then installing dependencies : 

- ref : https://kvcache-ai.github.io/Mooncake/getting_started/build.html#manual

sudo apt-get install -y build-essential \
                   cmake \
                   libibverbs-dev \
                   libgoogle-glog-dev \
                   libgtest-dev \
                   libjsoncpp-dev \
                   libnuma-dev \
                   libunwind-dev \
                   libpython3-dev \
                   libboost-all-dev \
                   libssl-dev \
                   pybind11-dev \
                   libcurl4-openssl-dev \
                   libhiredis-dev \
                   pkg-config \
                   patchelf


then `cd Mooncake && mkdir build`

then, this command : 
- for simple TCP only installation. no other bloat at the moment.
```
cmake .. \
      -DUSE_TCP=ON \
      -DUSE_MNNVL=OFF \
      -DUSE_HIP=OFF \
      -DUSE_NVMEOF=OFF \
      -DUSE_CXL=OFF \
      -DUSE_INTRA_NVLINK=OFF \
      -DUSE_EFA=OFF \
      -DUSE_BAREX=OFF \
      -DSTORE_USE_ETCD=ON \
      -DWITH_TE=ON \
      -DWITH_STORE=ON
```

then `make -j`
then `sudo make install`

check with `which mooncake_master` and `which mooncake_client`


## Re-installation

inside : `Mooncake/build`
run : `rm -rf *`
then run : 
```
cmake .. \
      -DUSE_TCP=ON \
      -DUSE_MNNVL=OFF \
      -DUSE_HIP=OFF \
      -DUSE_NVMEOF=OFF \
      -DUSE_CXL=OFF \
      -DUSE_INTRA_NVLINK=OFF \
      -DUSE_EFA=OFF \
      -DUSE_BAREX=OFF \
      -DSTORE_USE_ETCD=ON \
      -DWITH_TE=ON \
      -DWITH_STORE=ON
```
then run : `make -j` and then finall `sudo make install`