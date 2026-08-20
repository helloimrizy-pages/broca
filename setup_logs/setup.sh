#!/bin/bash
set -euo pipefail
REPO=/Users/helloimrizy/Documents/Academic/UNB/Research/broca
cd "$REPO"

echo "=== [1/4] install uv ==="
if ! command -v uv >/dev/null 2>&1; then
  brew install uv
fi
export PATH="/opt/homebrew/bin:$PATH"
uv --version

echo "=== [2/4] create venv (python 3.12) ==="
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "=== [3/4] install deps ==="
uv pip install \
  "torch" \
  "transformers>=4.45" \
  "datasets" \
  "scipy" \
  "accelerate" \
  "safetensors" \
  "huggingface_hub" \
  "numpy" \
  "pytest" \
  "tqdm"

python -c "import torch, transformers, scipy, datasets; print('torch', torch.__version__); print('mps available:', torch.backends.mps.is_available()); print('transformers', transformers.__version__)"

echo "=== [4/4] download OLMoE weights (safetensors only) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "allenai/OLMoE-1B-7B-0924",
    allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json", "*.txt", "*.model"],
    ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5"],
    max_workers=8,
)
print("MODEL_PATH", p)
PY

echo "=== SETUP DONE ==="
