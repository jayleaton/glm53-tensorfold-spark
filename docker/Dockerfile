# TensorFold serving GLM-5.3-Flash on two DGX Sparks (GB10, sm_121, aarch64).
#
# Base: NVIDIA's PyTorch container — the CUDA toolkit, PyTorch and Triton TensorFold's GLM kernels
# are built and tested with upstream. TensorFold itself comes from the pinned submodule plus our
# patches in patches/, so the image is reproducible and our engine changes ship with it.
FROM nvcr.io/nvidia/pytorch:26.07-py3

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TENSORFOLD_NO_UPDATE_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TORCH_EXTENSIONS_DIR=/cache/torch_extensions \
    TRITON_CACHE_DIR=/cache/triton \
    CUDA_CACHE_PATH=/cache/nv/ComputeCache \
    CUDA_CACHE_MAXSIZE=4294967296
# Every kernel cache lives in the /cache volume (scripts/serve.sh: glm53-tf-cache), so a restart (a new container)
# reuses them: torch cpp extensions (fast prefill, EXL3, KDA), Triton's compiled kernels, and the driver's JIT cache
# of PTX compiled for sm_121 (CUDA_CACHE_PATH; ~/.nv/ComputeCache by default, lost with every `docker rm`; 4 GiB
# instead of the 256 MiB default so PyTorch's and NCCL's PTX-only kernels stay cached). patches/0140.

# patches/0230 (GLM53_TF_COMM_BACKEND=roce) builds its RDMA proxy against libibverbs at first use. The 26.07 base
# already ships libibverbs-dev + ibverbs-providers (rdma-core 63); install them only if a base ever drops them.
RUN test -f /usr/include/infiniband/verbs.h || (apt-get update \
 && apt-get install -y --no-install-recommends libibverbs-dev ibverbs-providers && rm -rf /var/lib/apt/lists/*)

# patches/0610 (GLM53_TF_GRAMMAR, off by default): structured output's grammar engine, xgrammar 0.2.8 (Apache-2.0;
# aarch64 cp312 wheel), and what it imports (apache-tvm-ffi, pydantic, transformers: pure Python here). The base
# image's torch is a pre-release (2.13.0a0+...nv26.07), so pip's resolver cannot match xgrammar's `torch>=1.10` to it
# even with a `torch==<that>` constraint (W19: "conflicting dependencies"). So xgrammar goes in with --no-deps, then its
# one missing requirement (transformers, which does not require torch); the build fails if torch's version changed.
RUN python -c "import torch; print(torch.__version__)" > /tmp/torch-before.txt \
 && pip install --no-deps "xgrammar==0.2.8" \
 && pip install "transformers==5.17.0" \
 && python -c "import torch, xgrammar, transformers, importlib.metadata as m; \
v = open('/tmp/torch-before.txt').read().strip(); assert torch.__version__ == v, (torch.__version__, v); \
print('xgrammar', m.version('xgrammar'), 'transformers', transformers.__version__, 'torch', torch.__version__, '(unchanged)')" \
 && rm /tmp/torch-before.txt

COPY vendor/TensorFold /src/TensorFold
COPY patches /src/patches
# PATCHES: space-separated patch number prefixes to apply (e.g. "0001 0002 0003"); empty = every patch, in order
ARG PATCHES=""
RUN cd /src/TensorFold \
 && for p in $(ls /src/patches/*.patch 2>/dev/null | sort); do \
      n=$(basename "$p" | cut -c1-4); \
      if [ -n "$PATCHES" ] && ! echo " $PATCHES " | grep -q " $n "; then echo "skipping $p"; continue; fi; \
      echo "applying $p"; git apply --verbose "$p" || patch -p1 --forward < "$p" || exit 1; done \
 && pip install --no-deps /src/TensorFold \
 && pip install "huggingface-hub>=0.24" "tokenizers>=0.20" "safetensors>=0.4" "jinja2>=3.1" \
 && python -c "import tensorfold, torch, triton; print('tensorfold', tensorfold.__file__, 'torch', torch.__version__, 'triton', triton.__version__)"

COPY docker/entrypoint.sh /usr/local/bin/glm53-tf-entrypoint
ENTRYPOINT ["/usr/local/bin/glm53-tf-entrypoint"]
