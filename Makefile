.PHONY: install data train-sft train-dpo eval eval-baseline merge serve lint clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

install: ## Install all dependencies (runtime + training + dev)
	pip install -e ".[training,dev]"

# ──────────────────────────────────────────────
# Data Pipeline (Phase 2)
# ──────────────────────────────────────────────

data: ## Generate the training dataset from PDFs in ./manuals/
	python build_dataset.py

# ──────────────────────────────────────────────
# Training (Phases 3–4)
# ──────────────────────────────────────────────

train-sft: ## Run Supervised Fine-Tuning (Phase 3)
	python train_sft.py

train-dpo: ## Run DPO Preference Alignment (Phase 4)
	python train_dpo.py

# ──────────────────────────────────────────────
# Evaluation (Phase 5)
# ──────────────────────────────────────────────

eval: ## Evaluate the fine-tuned model
	python evaluate.py

eval-baseline: ## Evaluate with base model baseline comparison
	python evaluate.py --baseline

# ──────────────────────────────────────────────
# Deployment (Phase 6)
# ──────────────────────────────────────────────

merge: ## Merge LoRA adapters into standalone model
	python merge_model.py

serve: ## Launch vLLM inference server (requires merged model)
	python -m vllm.entrypoints.openai.api_server \
		--model ./olmo2-1b-domain-merged \
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
	rm -rf olmo2-1b-domain-sft olmo2-1b-domain-dpo olmo2-1b-domain-merged
	rm -rf olmo2-dpo-beta*
	rm -rf mlruns/
