# 🔬 OLMo 2 1B — PDF Domain Adaptation Pipeline

> End-to-end blueprint for fine-tuning [OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B) on any PDF document corpus using SFT + DPO

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

```bash
# 1. Clone and install dependencies
git clone <your-repo-url> && cd olmo
pip install -e ".[training]"

# 2. Generate the dataset from your PDFs
export OPENAI_API_KEY="your_key_here"
python build_dataset.py

# 3. Follow the training plan
# See sft_plan.md for Phases 3–6 (SFT, DPO, Eval, Deployment)
```

**Hardware Requirements:** ≥ 24 GB VRAM (A100, RTX 3090/4090, or RTX 6000 Ada). If no Ampere GPU is available, use FP16 instead of BF16 and disable Flash Attention 2.

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
 🎓 SFT Training (LoRA r=16, α=32)
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
olmo/
├── README.md                       # This file
├── sft_plan.md                     # Master Execution Blueprint (Phases 1–6)
├── data_engineering_spec.md        # Data pipeline theoretical framework
├── build_dataset.py                # Phase 2 orchestrator — runs the full data pipeline
├── pdf_extractor.py                # PDF ingestion via IBM Docling
├── teacher_model_synthesis.py      # GPT-4o synthetic QA generation
├── deduplicate_dataset.py          # Semantic dedup via sentence-transformers
├── generate_golden_eval.py         # Golden evaluation set generator
├── runpod_setup.sh                 # RunPod GPU environment setup
├── olmo.ipynb                      # Exploratory notebook
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
| `build_dataset.py` | **Orchestrator.** Scans `./manuals/`, processes every PDF, and outputs `alignment_dataset.jsonl`. |
| `pdf_extractor.py` | **Ingestion.** Uses IBM Docling for layout-aware PDF parsing with table preservation. |
| `teacher_model_synthesis.py` | **Synthesis.** Generates SFT + DPO pairs via GPT-4o with Pydantic structured outputs. Configurable domain via the `domain` parameter. |
| `deduplicate_dataset.py` | **Quality Control.** Cosine similarity dedup (threshold > 0.85) using `all-MiniLM-L6-v2`. |
| `generate_golden_eval.py` | **Evaluation.** Generates 100 complex multi-step evaluation scenarios. Configurable via the `DOMAIN` env var. |

---

## 🔧 Training Configuration

| Parameter | SFT | DPO |
|---|---|---|
| Base Model | `allenai/OLMo-2-0425-1B` | Post-SFT adapter |
| LoRA Rank (r) | 16 | — (reuses SFT adapters) |
| LoRA Alpha (α) | 32 | — |
| Target Modules | All 7 linear projections + `lm_head`, `embed_tokens` | — |
| Learning Rate | 2e-4 | Sweep: [5e-6, 1e-6] |
| Scheduler | Cosine (warmup 10%) | Cosine (warmup 10%) |
| Epochs | 3 | 1 |
| Effective Batch Size | 32 (4 × 8 accum) | 32 (2 × 16 accum) |
| Precision | BF16 | BF16 |
| DPO β | — | Sweep: [0.1, 0.25] |
| Experiment Tracking | MLflow | MLflow |

---

## 📈 Evaluation

- **Golden Eval Set:** 100 synthetic multi-step reasoning scenarios (`generate_golden_eval.py`)
- **In-Domain Testing:** Domain-specific QA accuracy
- **Out-of-Domain Testing:** Adversarial boundary enforcement (target > 90% refusal rate)

See Phase 5 in `sft_plan.md` for the full evaluation script.

---

## 🖥️ Infrastructure & Environment

* **`runpod_setup.sh`** — Initializes the RunPod GPU environment: installs the HF fine-tuning stack, Flash Attention 2, and launches MLflow UI.
* **Hardware:** Minimum 24 GB VRAM, RunPod PyTorch template recommended.
* **Storage:** ≥ 50 GB Container Disk + 100 GB Volume Disk.

---

## 🚀 Suggested Execution Order

1. Connect to the GPU Pod and run `bash runpod_setup.sh`.
2. Export the OpenAI API key: `export OPENAI_API_KEY="your_key_here"`.
3. Upload your target PDFs to the `./manuals/` directory.
4. Execute `python build_dataset.py` to generate the JSONL dataset.
5. Follow the Python scripts in `sft_plan.md` for Phase 3 (SFT) and Phase 4 (DPO).

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
