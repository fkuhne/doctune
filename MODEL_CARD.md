# Model Card: Doctune — Domain-Adapted QA

## Model Details

| Field | Value |
|---|---|
| **Model Name** | Doctune Domain QA |
| **Base Model** | User-specified (any HuggingFace causal LM) |
| **Architecture** | Transformer decoder-only |
| **Fine-Tuning Method** | LoRA (auto-detected targets) → SFT + DPO |
| **Status** | Pre-training blueprint (not yet executed) |
| **License** | Apache 2.0 |

## Intended Use

This model is designed for **closed-domain question answering** over technical documentation extracted from PDF manuals. After fine-tuning:

- **Primary Use:** Answering user questions grounded in a specific PDF corpus
- **Secondary Use:** Demonstrating SFT + DPO alignment methodology on any HuggingFace LLM
- **Out-of-Scope:** General knowledge QA, code generation, creative writing, or any task outside the trained domain

## Training Methodology

### Phase 1: Data Synthesis
- PDF documents are parsed via IBM Docling (layout-aware extraction)
- Teacher Model (GPT-4o, Claude, or Ollama) generates SFT question-answer pairs and DPO preference tuples
- Cosine similarity deduplication (threshold > 0.85) enforces dataset diversity

### Phase 2: SFT (Supervised Fine-Tuning)
- LoRA adapters auto-targeting all detected Linear projections + `lm_head` + `embed_tokens`
- 3 epochs, cosine LR schedule (2e-4), BF16 precision
- Dedicated `<|pad|>` token (not reusing EOS)

### Phase 3: DPO (Direct Preference Optimization)
- Aligns the SFT model using chosen vs. rejected response pairs
- Beta sweep: [0.1, 0.25], LR sweep: [5e-6, 1e-6]
- 1 epoch per configuration

### Phase 4: Weight Merge + Deployment
- LoRA adapters merged into base model for standalone inference
- Served via vLLM with PagedAttention

## Limitations

- **Domain-locked:** The model will refuse or produce low-quality answers for out-of-domain queries
- **Synthetic data dependency:** All training data is teacher-model-generated; no human-curated ground truth exists yet
- **Scale constraints:** Smaller models limit complex multi-step reasoning compared to larger ones
- **Language:** English only

## Evaluation

| Metric | Target |
|---|---|
| In-domain accuracy | Assessed via golden eval set (100 synthetic scenarios) |
| Out-of-domain refusal rate | > 90% |
| Baseline comparison | Planned (base model vs. fine-tuned) |

## Ethical Considerations

- The model should **not** be used as a sole source of truth for safety-critical decisions
- Synthetic training data may contain subtle biases inherited from the teacher model
- Users should validate model outputs against original source documentation
- The DPO alignment enforces boundary behavior but cannot guarantee zero hallucination

## Citation

```bibtex
@misc{doctune,
  title={Doctune: PDF Domain Adaptation Pipeline for HuggingFace LLMs},
  year={2026},
  url={https://github.com/your-username/doctune}
}
```
