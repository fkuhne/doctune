#!/bin/bash
# ==============================================================================
# Local Setup Script — Phase 2 (Data Curation Pipeline)
# ==============================================================================
# This script prepares a local macOS or Linux workstation for Phase 2 only.
# Phase 2 does NOT require a GPU — it uses Docling (CPU), sentence-transformers
# (CPU), and OpenAI API calls.
#
# For GPU-dependent phases (3–6: SFT, DPO, Evaluation, Deployment), use
# runpod_setup.sh on a provisioned GPU pod instead.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "--- Local Environment Setup (Phase 2: Data Curation) ---"

# 1. Create and activate a virtual environment (if not already in one)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Creating Python virtual environment in .venv/ ..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Virtual environment activated: $VIRTUAL_ENV"
else
    echo "Already in a virtual environment: $VIRTUAL_ENV"
fi

# 2. Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# 3. Install base project dependencies (no training/GPU packages)
echo "Installing base dependencies (Docling, sentence-transformers, OpenAI)..."
pip install -e "."

# 4. Create the manuals directory if it doesn't exist
mkdir -p manuals

# 5. Verify the OpenAI API key
echo ""
echo "========================================================================"
echo "Setup Complete!"
echo ""
echo "Next steps:"
echo "  1. Export your OpenAI API key:"
echo "     export OPENAI_API_KEY=\"your_key_here\""
echo ""
echo "  2. Place your PDF files in the ./manuals/ directory"
echo ""
echo "  3. Run the data generation pipeline:"
echo "     python build_dataset.py"
echo ""
echo "  4. (Optional) Generate the golden evaluation set:"
echo "     python generate_golden_eval.py"
echo ""
echo "  5. Transfer the generated .jsonl files to your GPU pod for Phases 3–6"
echo "========================================================================"
