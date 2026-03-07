#!/bin/bash
# ==============================================================================
# RunPod Initialization Script
# Target: OLMo 2 1B Fine-Tuning for HP Printer QA
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "--- Starting RunPod Environment Setup ---"

# 1. Navigate to the persistent storage volume
# RunPod provides a /workspace directory that persists across pod restarts.
# NEVER store datasets or model weights outside of this directory.
cd /workspace
mkdir -p /workspace/hp-qa-model
cd /workspace/hp-qa-model

# 2. Update system packages (just in case the base image is stale)
echo "Updating apt repositories..."
apt-get update -y && apt-get install -y git wget tmux jq

# 3. Upgrade pip to prevent dependency resolution errors
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# 4. Install the Hugging Face Fine-Tuning Stack
echo "Installing transformers, peft, trl, and dependencies..."
pip install "transformers>=4.48" peft trl accelerate datasets bitsandbytes unstructured marker-pdf mlflow

# 5. Install Flash Attention 2
# We use --no-build-isolation to force pip to use the pre-installed PyTorch/CUDA 
# environment, which prevents it from taking 45 minutes to compile from scratch.
echo "Compiling and installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

echo "========================================================================"
echo "Setup Complete! The environment is ready."
echo "Hardware Check:"
nvidia-smi
echo "Directory /workspace/hp-qa-model is prepped for Phase 2 data generation."
echo "========================================================================"

# Launch MLflow UI in the background
mlflow ui --host 0.0.0.0 --port 5000 &
