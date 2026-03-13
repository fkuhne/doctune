#!/bin/bash
# ==============================================================================
# Local Setup Script — Phase 2 (Data Curation Pipeline)
# ==============================================================================
# This script prepares a local macOS or Linux workstation for Phase 2 only.
# Phase 2 does NOT require a GPU — it uses Docling (CPU), sentence-transformers
# (CPU), and OpenAI/Anthropic API calls.
#
# For GPU-dependent phases (3–6: SFT, DPO, Evaluation, Deployment), use
# runpod_setup.sh on a provisioned GPU pod instead.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "--- Local Environment Setup (Phase 2: Data Curation) ---"

# 1. Check for uv — install if not found
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the updated PATH so uv is available in this session
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

echo "Using uv: $(uv --version)"

# 2. Create a virtual environment and install dependencies
echo "Creating virtual environment and installing base dependencies..."
uv venv .venv
source .venv/bin/activate
uv pip install -e "."

# 3. Create the manuals directory if it doesn't exist
mkdir -p manuals

# 4. Print next steps
echo ""
echo "========================================================================"
echo "Setup Complete!"
echo ""
echo "To activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Export your API key:"
echo "     export OPENAI_API_KEY=\"your_key_here\""
echo "     # or: export ANTHROPIC_API_KEY=\"your_key_here\""
echo ""
echo "  2. Place your PDF files in the ./manuals/ directory"
echo ""
echo "  3. Run the data generation pipeline:"
echo "     python build_dataset.py"
echo "     # or: python build_dataset.py --model claude-3-5-sonnet-20241022"
echo ""
echo "  4. (Optional) Generate the golden evaluation set:"
echo "     python generate_golden_eval.py"
echo ""
echo "  5. Transfer the generated .jsonl files to your GPU pod for Phases 3–6"
echo "========================================================================"
