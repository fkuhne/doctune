.PHONY: install local-setup data train-sft train-dpo eval eval-baseline merge serve lint clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

install: ## Install all dependencies — runtime + training + dev (GPU required)
	uv pip install -e ".[training,dev]"

local-setup: ## Set up local environment for Phase 2 data generation (no GPU)
	bash local_setup.sh

# ──────────────────────────────────────────────
# Data Pipeline (Phase 2) — runs locally, no GPU needed
# ──────────────────────────────────────────────

data: ## Generate the training dataset from PDFs in ./manuals/ (no GPU)
	python build_dataset.py

# ──────────────────────────────────────────────
# Training (Phases 3–4) — GPU required
# NOTE: You must pass --model-id <your-hf-model-id> to these scripts
# ──────────────────────────────────────────────

train-sft: ## Run Supervised Fine-Tuning (Phase 3) — requires MODEL_ID env var
	python train_sft.py --model-id $(MODEL_ID)

train-dpo: ## Run DPO Preference Alignment (Phase 4) — requires MODEL_ID env var
	python train_dpo.py --model-id $(MODEL_ID)

# ──────────────────────────────────────────────
# Evaluation (Phase 5) — GPU required
# ──────────────────────────────────────────────

eval: ## Evaluate the fine-tuned model — requires MODEL_ID env var
	python evaluate.py --model-id $(MODEL_ID)

eval-baseline: ## Evaluate with base model baseline comparison
	python evaluate.py --model-id $(MODEL_ID) --baseline

# ──────────────────────────────────────────────
# Deployment (Phase 6) — GPU required
# ──────────────────────────────────────────────

merge: ## Merge LoRA adapters into standalone model — requires MODEL_ID env var
	python merge_model.py --model-id $(MODEL_ID)

serve: ## Launch vLLM inference server (requires merged model)
	python -m vllm.entrypoints.openai.api_server \
		--model ./doctune-merged \
		--dtype bfloat16 \
		--max-model-len 2048 \
		--port 8000

# ──────────────────────────────────────────────
# Development
# ──────────────────────────────────────────────

lint: ## Run ruff linter
	ruff check .

clean: ## Remove generated artifacts (datasets, checkpoints)
	rm -rf alignment_dataset.jsonl golden_eval.jsonl
	rm -rf doctune-sft doctune-dpo doctune-merged
	rm -rf mlruns/
