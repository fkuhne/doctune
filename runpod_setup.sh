#!/bin/bash
# ==============================================================================
# RunPod Initialization Script — Phases 3–6 (GPU Required)
# Target: OLMo 2 1B Domain-Specific Fine-Tuning
# ==============================================================================
# This script sets up the remote GPU pod for training, alignment, evaluation,
# and deployment. Phase 2 (data curation) can be done locally — see
# local_setup.sh for the local alternative.
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "--- Starting RunPod Environment Setup ---"

# 1. Navigate to the persistent storage volume
# RunPod provides a /workspace directory that persists across pod restarts.
# NEVER store datasets or model weights outside of this directory.
cd /workspace
mkdir -p /workspace/olmo-finetune
cd /workspace/olmo-finetune

# 2. Update system packages (just in case the base image is stale)
echo "Updating apt repositories..."
apt-get update -y && apt-get install -y git wget tmux jq

# 3. Install uv (much faster dependency resolution than pip)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"

# 4. Install the Hugging Face Fine-Tuning Stack
echo "Installing transformers, peft, trl, and dependencies..."
uv pip install --system "transformers>=4.48" peft trl accelerate datasets bitsandbytes unstructured marker-pdf mlflow

# 5. Install Flash Attention 2
# We use --no-build-isolation to force uv to use the pre-installed PyTorch/CUDA
# environment, which prevents it from taking 45 minutes to compile from scratch.
echo "Compiling and installing Flash Attention 2..."
uv pip install --system flash-attn --no-build-isolation

echo "========================================================================"
echo "Setup Complete! The environment is ready."
echo "Hardware Check:"
nvidia-smi
echo "Directory /workspace/olmo-finetune is prepped for Phase 2 data generation."
echo "========================================================================"

# Launch MLflow UI in the background
mlflow ui --host 0.0.0.0 --port 5000 &
