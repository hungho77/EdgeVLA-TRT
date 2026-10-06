# Running 0.11 on JetPack 6 (Orin, CUDA 12.6, TensorRT 10.3)

Upstream 0.11 supports Orin only on JetPack 7.2. This branch (`orin-jp6-harness`) is a test
harness that runs export → build → inference on a JetPack 6.2 AGX Orin. It is not meant for `main`.

Three gaps, each reproduced on unmodified upstream 0.11:

| Gap | Symptom | Handled by |
|---|---|---|
| NVRTC 12.6 has no built-in `vector_types.h` etc. | `Failed to NVRTC compile XQA kernel` | `gen_cpp_header.py` embeds the CUDA 12 runtime headers when `CUDART_VERSION < 13000` |
| TRT 10.3 passes an empty `opt` profile to `configurePlugin` | `QkvConcatPlugin: ... must match for every profile` | `cpp/plugins/trt103OptCompat.h`, applied in Attention and QkvConcat |
| SM87 prefill attention is CuTe DSL only; the shipped sm_87 archive is CUDA 13 (cubins fail with `CUDA_ERROR_INVALID_IMAGE` on the 12.6 driver) and needs CUDA ≥ 12.8 runtime APIs (`cudaLibrary*`) | `selected prefill kernel is unavailable (... SM=87)` | Generate sm_87 FMHA with the CuTe DSL cu12 toolchain and link the CUDA 12.9 runtime (minor-version compatible with the 12.6 driver) |

## One-time setup

```bash
# CuTe DSL CUDA 12 toolchain (separate venv)
python3 -m venv ~/cutedsl-cu12-venv
~/cutedsl-cu12-venv/bin/pip install "nvidia-cutlass-dsl==4.7.0" "nvidia-cutlass-dsl-libs-base==4.7.0" \
    "nvidia-cutlass-dsl-libs-cu12==4.7.0" "cupy-cuda12x==12.3.0" "numpy<2.3"

# sm_87 FMHA artifacts (written to the gitignored cpp/kernels/cuteDSLArtifact/aarch64/sm_87)
PATH=~/cutedsl-cu12-venv/bin:/usr/local/cuda/bin:$PATH ~/cutedsl-cu12-venv/bin/python \
    kernelSrcs/build_cutedsl.py --gpu_arch sm_87 --arch aarch64 --kernels fmha --cuda-version 12.6 --clean

# CUDA 12.9 runtime (libcudart.so.12 only)
pip download --no-deps "nvidia-cuda-runtime-cu12==12.9.*" --platform manylinux2014_aarch64 --only-binary=:all: -d /tmp/rt
mkdir -p ~/cuda-12.9-runtime/lib && cd ~/cuda-12.9-runtime/lib \
    && python3 -m zipfile -e /tmp/rt/nvidia_cuda_runtime_cu12-*.whl /tmp/rt/x \
    && cp /tmp/rt/x/nvidia/cuda_runtime/lib/libcudart.so.12 . && ln -sf libcudart.so.12 libcudart.so
```

## Build and run

```bash
cmake -S . -B build-jp6 -DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=12.6 -DENABLE_CUTE_DSL=fmha -DCUDART_LIB=$HOME/cuda-12.9-runtime/lib/libcudart.so.12
cmake --build build-jp6 -j

export LD_LIBRARY_PATH=$HOME/cuda-12.9-runtime/lib:/usr/lib/aarch64-linux-gnu:$PWD/build-jp6
export EDGELLM_PLUGIN_PATH=$PWD/build-jp6/libNvInfer_edgellm_plugin.so
```

Export on CPU (`torch==2.13.0+cpu` from `https://download.pytorch.org/whl/cpu`). INT4 checkpoints need
`--int4-gemm-plugin-version 1`: the V2 plugin is CuTe DSL `int4_fp16_gemm`, which is not generated above.
NVFP4 cannot run here at all (SM87, TensorRT 10.3).
