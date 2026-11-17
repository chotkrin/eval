#!/usr/bin/env bash
set -u

# Adjust these to match the intended evaluation request.
NEED_GPUS=4
NEED_MEM_MB=49152  # 48 GiB
NEED_CPUS=4
MAX_ATTEMPTS=19
PYTHON=/data/zelin/verl/envs/llmjp-eval/.venv/bin/python

candidates=()

while IFS= read -r node; do
    # gpu00 is already confirmed healthy; gpu03/gpu10 have Error 802.
    case "$node" in llm-jp-noble-gpu00|llm-jp-noble-gpu03|llm-jp-noble-gpu10) continue ;; esac

    info=$(scontrol -o show node "$node" 2>/dev/null) || continue
    [[ "$info" =~ State=([^[:space:]]+) ]] || continue
    state=${BASH_REMATCH[1]}
    [[ "$state" == IDLE* || "$state" == MIXED* ]] || continue

    [[ "$info" =~ CfgTRES=([^[:space:]]+) ]] || continue
    cfg=${BASH_REMATCH[1]}
    [[ "$cfg" =~ gres/gpu=([0-9]+) ]] || continue
    gpu_total=${BASH_REMATCH[1]}

    gpu_alloc=0
    if [[ "$info" =~ AllocTRES=([^[:space:]]*) ]]; then
        alloc=${BASH_REMATCH[1]}
        if [[ "$alloc" =~ gres/gpu=([0-9]+) ]]; then
            gpu_alloc=${BASH_REMATCH[1]}
        fi
    fi

    [[ "$info" =~ RealMemory=([0-9]+) ]] || continue
    mem_total=${BASH_REMATCH[1]}
    [[ "$info" =~ AllocMem=([0-9]+) ]] || continue
    mem_alloc=${BASH_REMATCH[1]}
    [[ "$info" =~ CPUEfctv=([0-9]+) ]] || continue
    cpu_total=${BASH_REMATCH[1]}
    [[ "$info" =~ CPUAlloc=([0-9]+) ]] || continue
    cpu_alloc=${BASH_REMATCH[1]}

    gpu_free=$((gpu_total - gpu_alloc))
    mem_free=$((mem_total - mem_alloc))
    cpu_free=$((cpu_total - cpu_alloc))
    if (( gpu_free >= NEED_GPUS && mem_free >= NEED_MEM_MB && cpu_free >= NEED_CPUS )); then
        printf '%-24s state=%-17s free_GPU=%d free_mem=%.1fGiB free_CPU=%d\n' \
            "$node" "$state" "$gpu_free" "$(awk -v x="$mem_free" 'BEGIN {print x/1024}')" "$cpu_free"
        candidates+=("$node")
    fi
done < <(sinfo -h -N -p gpu -o '%N' | sort -u)

if ((${#candidates[@]} == 0)); then
    echo 'No other nodes currently meet 4 GPU + 48 GiB RAM + 4 CPU requirements.'
    exit 1
fi

attempt=0
for node in "${candidates[@]}"; do
    ((attempt+=1))
    ((attempt <= MAX_ATTEMPTS)) || break
    echo "========== Probing $node ($attempt/$MAX_ATTEMPTS) =========="

    if srun -p gpu --nodes=1 --ntasks=1 --nodelist="$node" \
      --gres="gpu:${NEED_GPUS}" --cpus-per-task="$NEED_CPUS" --mem=48G \
      --time=00:02:00 --immediate=10 --job-name=0001_eval-0810 \
      "$PYTHON" -c '
import ctypes, os, socket, subprocess, sys
required = int(sys.argv[1])
print("node:", socket.gethostname(), flush=True)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
service = subprocess.run(["systemctl", "is-active", "nvidia-fabricmanager"], capture_output=True, text=True)
print("Fabric Manager:", service.stdout.strip() or service.stderr.strip(), flush=True)
driver = ctypes.CDLL("libcuda.so.1")
driver.cuInit.argtypes = [ctypes.c_uint]
driver.cuInit.restype = ctypes.c_int
result = driver.cuInit(0)
print("cuInit(0):", result, flush=True)
if result != 0:
    sys.exit(2)
import torch
print("torch:", torch.__version__, "visible GPUs:", torch.cuda.device_count(), flush=True)
torch.cuda.init()
if torch.cuda.device_count() < required:
    sys.exit("Not enough visible GPUs")
for i in range(required):
    value = torch.ones(1, device=f"cuda:{i}")
    torch.cuda.synchronize(i)
    print(f"cuda:{i}: OK ({value.item()})", flush=True)
print("ALL_GPUS_PASS", flush=True)
' "$NEED_GPUS"; then
        echo "HEALTHY_NODE=$node"
        echo "For evaluation: --nodelist=$node"
        exit 0
    else
        echo "No successful immediate test on $node (could be unavailable, restricted, or CUDA-failing)."
    fi
done

echo 'No healthy immediately allocatable alternative was confirmed.'
exit 1
