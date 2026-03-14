# 🔬 Doctune — PDF Domain Adaptation Pipeline

> End-to-end blueprint for domain-adapting **any HuggingFace causal LM** on PDF document corpora using SFT + DPO

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-≥3.10-green.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Pre--Training-orange.svg)]()

---

## Table of Contents

- [Quick Start](#-quick-start)
- [Architecture Overview](#-architecture-overview)
- [Repository Structure](#-repository-structure)
- [Documentation & Plans](#-documentation--master-plans)
- [Data Pipeline (Phase 2)](#-data-generation-pipeline)
- [Training Configuration](#-training-configuration)
- [Evaluation](#-evaluation)
- [Infrastructure](#-infrastructure--environment)
- [Suggested Execution Order](#-suggested-execution-order)
- [License](#-license)

---

## ⚡ Quick Start

### Option A: Free Local Data Generation (Ollama — No API Key, No GPU)

Run everything locally using free, open-weight LLMs via [Ollama](https://ollama.com):

```bash
# 1. Install Ollama (https://ollama.com/download)
curl -fsSL https://ollama.com/install.sh | sh   # Linux
# macOS: download from https://ollama.com/download

# 2. Pull a model (llama3.1:8b is a good default)
ollama pull llama3.1:8b

# 3. Clone the repo and set up
git clone <your-repo-url> && cd doctune
uv venv .venv && source .venv/bin/activate
uv pip install -e "."

# 4. Generate the dataset — no API key needed!
mkdir -p manuals && cp /path/to/your/*.pdf manuals/
python build_dataset.py --model llama3.1:8b
```

> **Tip:** For better quality output, use a larger model like `llama3.1:70b` or `qwen2.5:14b` if your machine has enough RAM.

### Option B: Paid API Data Generation (OpenAI / Anthropic — No GPU)

Higher quality synthetic data using cloud APIs:

```bash
# 1. Clone the repo
git clone <your-repo-url> && cd doctune

# 2. Install uv (if you don't have it already)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create a virtual environment and install base dependencies
uv venv .venv && source .venv/bin/activate
uv pip install -e "."       # Base deps only (no training/GPU packages)

# 4. Generate the dataset from your PDFs
export OPENAI_API_KEY="your_key_here"
mkdir -p manuals && cp /path/to/your/*.pdf manuals/
python build_dataset.py

# Or use an Anthropic model instead:
export ANTHROPIC_API_KEY="your_key_here"
python build_dataset.py --model claude-3-5-sonnet-20241022

# 5. (Optional) Generate the golden evaluation set
python generate_golden_eval.py
```

Then transfer the generated `alignment_dataset.jsonl` and `golden_eval.jsonl` to your GPU environment for training.

### Option C: Full Pipeline (GPU Required)

```bash
# 1. On the GPU pod, install all dependencies
uv pip install -e ".[training]"

# 2. Generate dataset (or upload the .jsonl files generated locally)
export OPENAI_API_KEY="your_key_here"   # or ANTHROPIC_API_KEY
python build_dataset.py                  # --model claude-3-5-sonnet-20241022 for Anthropic

# 3. Train (SFT → DPO → Evaluate → Merge)
#    Replace <your-model-id> with any HuggingFace model
#    Examples: meta-llama/Llama-3.1-8B, mistralai/Mistral-7B-v0.3, allenai/OLMo-2-0425-1B
python train_sft.py   --model-id <your-model-id>
python train_dpo.py   --model-id <your-model-id>
python evaluate.py    --model-id <your-model-id>
python merge_model.py --model-id <your-model-id>
```

**Hardware Requirements (Phases 3–6):** ≥ 24 GB VRAM (A100, RTX 3090/4090, or RTX 6000 Ada). Flash Attention 2 is auto-detected — if unavailable, the pipeline falls back to eager attention automatically.

---

## 🏗️ Architecture Overview

```
 📄 PDF Documents
       │
       ▼
 🔍 Docling Parser (pdf_extractor.py)
       │
       ▼
 ✂️  Semantic Chunking + Metadata Injection
       │
       ▼
 🤖 Teacher Model Synthesis (teacher_model_synthesis.py)
       │
       ▼
 🧹 Cosine Deduplication (deduplicate_dataset.py)
       │
       ▼
 📊 alignment_dataset.jsonl
       │
       ▼
 🎓 SFT Training (LoRA — auto-detected target modules)
       │
       ▼
 ⚖️  DPO Alignment (β sweep)
       │
       ▼
 🔗 Weight Merge → 🚀 vLLM Serving
```

---

## 📁 Repository Structure

```
doctune/
├── README.md                       # This file
├── MODEL_CARD.md                   # Model card (intended use, method, limitations)
├── DATA_CARD.md                    # Dataset card (schema, source guidance, quality)
├── sft_plan.md                     # Master Execution Blueprint (Phases 1–6)
├── data_engineering_spec.md        # Data pipeline theoretical framework
├── model_utils.py                  # Model-agnostic utilities (LoRA target detection, etc.)
├── build_dataset.py                # Phase 2 orchestrator — runs the full data pipeline
├── pdf_extractor.py                # PDF ingestion via IBM Docling
├── teacher_model_synthesis.py      # GPT-4o synthetic QA generation
├── deduplicate_dataset.py          # Semantic dedup via sentence-transformers
├── generate_golden_eval.py         # Golden evaluation set generator
├── train_sft.py                    # Phase 3 — Supervised Fine-Tuning
├── train_dpo.py                    # Phase 4 — DPO Preference Alignment
├── evaluate.py                     # Phase 5 — Evaluation & Red Teaming (--baseline)
├── merge_model.py                  # Phase 6 — Weight Merging for Deployment
├── Makefile                        # Dev targets (make data, make train-sft, etc.)
├── local_setup.sh                  # Local macOS/Linux setup for Phase 2 (no GPU)
├── runpod_setup.sh                 # RunPod GPU environment setup (Phases 3–6)
├── olmo.ipynb                      # Exploratory notebook
├── examples/
│   └── sample_dataset.jsonl        # 10 representative training examples
├── pyproject.toml                  # Project metadata & pinned dependencies
├── LICENSE                         # Apache 2.0
└── .gitignore
```

---

## 📖 Documentation & Master Plans

* **[`sft_plan.md`](sft_plan.md)** — The Master Execution Blueprint. Contains the exact Python scripts, hyperparameter configurations, and terminal commands for Phases 1–6 (Infrastructure, Data, SFT, DPO, Evaluation, Deployment).

* **[`data_engineering_spec.md`](data_engineering_spec.md)** — Theoretical framework for the data pipeline: PDF chunking rules, metadata injection, multi-angle question synthesis, hard negative protocol for DPO, and cosine deduplication thresholds.

---

## 📦 Data Generation Pipeline

These modular Python scripts handle the conversion of raw PDFs into a training-ready dataset:

| Script | Purpose |
|---|---|
| `build_dataset.py` | **Orchestrator.** Scans `./manuals/`, processes every PDF, and outputs `alignment_dataset.jsonl`. Supports `--model` and `--provider` flags. |
| `pdf_extractor.py` | **Ingestion.** Uses IBM Docling for layout-aware PDF parsing with table preservation. |
| `teacher_model_synthesis.py` | **Synthesis.** Generates SFT + DPO pairs via OpenAI, Anthropic, or Ollama (local). Configurable domain, model, and provider. |
| `deduplicate_dataset.py` | **Quality Control.** Cosine similarity dedup (threshold > 0.85) using `all-MiniLM-L6-v2`. |
| `generate_golden_eval.py` | **Evaluation.** Generates 100 complex multi-step evaluation scenarios. Supports `--model`, `--provider`, and `--count` flags. |

---

## 🔧 Training Configuration

| Parameter | SFT | DPO |
|---|---|---|
| Base Model | User-specified via `--model-id` | Post-SFT adapter |
| LoRA Rank (r) | 16 | — (reuses SFT adapters) |
| LoRA Alpha (α) | 32 | — |
| Target Modules | Auto-detected from model architecture | — |
| Learning Rate | 2e-4 | Sweep: [5e-6, 1e-6] |
| Scheduler | Cosine (warmup 10%) | Cosine (warmup 10%) |
| Epochs | 3 | 1 |
| Effective Batch Size | 32 (4 × 8 accum) | 32 (2 × 16 accum) |
| Precision | BF16 | BF16 |
| DPO β | — | Sweep: [0.1, 0.25] |
| Experiment Tracking | MLflow | MLflow |

> **Note:** LoRA target modules are auto-detected from the model architecture at runtime, so the pipeline works with any HuggingFace causal LM (LLaMA, Mistral, Phi, Gemma, Qwen, OLMo, etc.).

---

## 📈 Evaluation

- **Golden Eval Set:** 100 synthetic multi-step reasoning scenarios (`generate_golden_eval.py`)
- **In-Domain Testing:** Domain-specific QA accuracy
- **Out-of-Domain Testing:** Adversarial boundary enforcement (target > 90% refusal rate)

See Phase 5 in `sft_plan.md` for the full evaluation script.

---

## 🖥️ Infrastructure & Environment

### Local (Phase 2 — Data Generation)
* **`local_setup.sh`** — Sets up a local Python virtual environment with base dependencies for PDF extraction and dataset generation. No GPU required.
* **Requirements:** Python ≥ 3.10, macOS or Linux, OpenAI API key.

### Remote GPU Pod (Phases 3–6 — Training & Deployment)
* **`runpod_setup.sh`** — Initializes the RunPod GPU environment: installs the HF fine-tuning stack, Flash Attention 2, and launches MLflow UI.
* **Hardware:** Minimum 24 GB VRAM, RunPod PyTorch template recommended.
* **Storage:** ≥ 50 GB Container Disk + 100 GB Volume Disk.

---

## 🚀 Suggested Execution Order

### Steps 1–3: Data Curation (can run locally — no GPU)
1. Run `bash local_setup.sh` (or `pip install -e "."` manually).
2. Export your API key: `export OPENAI_API_KEY="your_key_here"`.
3. Place your target PDFs in `./manuals/` and run `python build_dataset.py`.

### Steps 4–6: Training & Deployment (GPU required)
4. Connect to the GPU pod and run `bash runpod_setup.sh`.
5. Transfer `alignment_dataset.jsonl` and `golden_eval.jsonl` to the pod (if generated locally).
6. Run the training pipeline with your chosen model:
   ```bash
   export MODEL_ID="meta-llama/Llama-3.1-8B"  # or any HuggingFace model
   python train_sft.py   --model-id $MODEL_ID
   python train_dpo.py   --model-id $MODEL_ID
   python evaluate.py    --model-id $MODEL_ID
   python merge_model.py --model-id $MODEL_ID
   ```

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
