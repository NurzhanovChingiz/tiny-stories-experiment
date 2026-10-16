# ROCm kernel for notebooks under /app/notebooks.
# Python 3.12 and HIP torch 2.9.1 already live in the base image at /opt/venv.
# Install [project] dependencies from pyproject.toml, plus the notebook and inference groups.
FROM ghcr.io/astral-sh/uv:0.10.6 AS uv

FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1

COPY --from=uv /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1 \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MIOPEN_FIND_MODE=2

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --inexact --no-dev --group notebook --group inference \
    && python -c "import torch; assert torch.version.hip, 'expected ROCm/HIP torch from /opt/venv'"

# Stock ROCm 7.2.1 AsyncEventsLoop busy-spins two cores on WSL2: /dev/dxg
# supplies no interrupt-backed HSA event, and the polling fallback never sleeps.
# Rebuild that same commit with a 20us-to-2ms nap in the polling fallback.
RUN <<'EOS'
set -eux
apt-get update
apt-get install -y --no-install-recommends cmake pkg-config libdrm-dev libnuma-dev libelf-dev xxd
rm -rf /var/lib/apt/lists/*
git init /tmp/ROCR-Runtime
git -C /tmp/ROCR-Runtime remote add origin https://github.com/ROCm/ROCR-Runtime.git
git -C /tmp/ROCR-Runtime fetch --depth 1 origin 820d83572bd8a098ba8366b84943c760ecce8ca5
git -C /tmp/ROCR-Runtime checkout --detach FETCH_HEAD
cat > /tmp/hsa-wsl-idle-spin.patch <<'PATCH'
diff --git a/runtime/hsa-runtime/core/runtime/runtime.cpp b/runtime/hsa-runtime/core/runtime/runtime.cpp
index 41b6f75..806dc4c 100644
--- a/runtime/hsa-runtime/core/runtime/runtime.cpp
+++ b/runtime/hsa-runtime/core/runtime/runtime.cpp
@@ -1842,6 +1842,11 @@ void Runtime::AsyncEventsLoop(void* _eventsInfo) {
       bool finish = false;
       bool polling = false;
       bool init_age = true;
+      // WSL/dxg signals have no interrupt event, so the poll below would
+      // busy-spin one core per async-events thread for the life of the GPU
+      // context. Back off from 20us to 2ms while idle; the next wait starts
+      // at 20us again.
+      int poll_nap_us = 20;

       // Mark all signals with a waiting tag
       // @note: Waiting tag must be marked before the signal state check on CPU to
@@ -1894,6 +1899,13 @@ void Runtime::AsyncEventsLoop(void* _eventsInfo) {
         if (interrupt_wait) {
           WaitForInterrupt();
           init_age = false;
+        } else if (polling && !finish) {
+          rocr::os::uSleep(poll_nap_us);
+          if (poll_nap_us < 1000) {
+            poll_nap_us *= 2;
+          } else {
+            poll_nap_us = 2000;
+          }
         }
       }
     }
PATCH
git -C /tmp/ROCR-Runtime apply /tmp/hsa-wsl-idle-spin.patch
mkdir -p /tmp/cmake-clang /tmp/cmake-llvm
cat > /tmp/cmake-clang/ClangConfig.cmake <<'EOF'
if(NOT TARGET clang)
  add_executable(clang IMPORTED GLOBAL)
  set_target_properties(clang PROPERTIES IMPORTED_LOCATION /opt/rocm/llvm/bin/clang)
endif()
EOF
cat > /tmp/cmake-llvm/LLVMConfig.cmake <<'EOF'
if(NOT TARGET llvm-objcopy)
  add_executable(llvm-objcopy IMPORTED GLOBAL)
  set_target_properties(llvm-objcopy PROPERTIES IMPORTED_LOCATION /opt/rocm/llvm/bin/llvm-objcopy)
endif()
EOF
cmake -S /tmp/ROCR-Runtime -B /tmp/ROCR-Runtime/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DIMAGE_SUPPORT=OFF \
    -DClang_DIR=/tmp/cmake-clang \
    -DLLVM_DIR=/tmp/cmake-llvm
cmake --build /tmp/ROCR-Runtime/build --target hsa-runtime64 -j"$(nproc)"
cp "$(find /tmp/ROCR-Runtime/build -name 'libhsa-runtime64.so.1*' -type f | head -n 1)" \
    "$(readlink -f /opt/rocm/lib/libhsa-runtime64.so.1)"
rm -rf /tmp/ROCR-Runtime /tmp/hsa-wsl-idle-spin.patch
EOS

ARG UID=1000
ARG GID=1000

RUN mkdir -p \
        /app/cache \
        /home/app/.config/miopen \
        /tmp \
    && if ! getent group "${GID}" >/dev/null; then groupadd --gid "${GID}" app; fi \
    && if ! getent passwd "${UID}" >/dev/null; then \
         useradd --uid "${UID}" --gid "${GID}" --create-home --home-dir /home/app --shell /bin/bash app; \
       fi \
    && chown -R "${UID}:${GID}" \
        /app/cache \
        /home/app \
        /tmp

ENV HOME=/home/app \
    XDG_CACHE_HOME=/app/cache \
    UV_CACHE_DIR=/app/cache/uv \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    MIOPEN_FIND_MODE=2

USER ${UID}:${GID}

WORKDIR /app/notebooks

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/app/notebooks", "--ServerApp.token=", "--ServerApp.password=", "--ServerApp.disable_check_xsrf=True"]
