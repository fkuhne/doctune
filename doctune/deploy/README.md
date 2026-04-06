# `doctune/deploy`

This package is **Phase 6** of the doctune training pipeline: preparing the
fine-tuned model for production inference. It takes the LoRA adapters produced by
`train_dpo.py` and fuses them permanently into the base model weights, producing a
standalone binary that can be served without PEFT or the adapter directory.

```
./doctune-dpo/          (LoRA adapters — best DPO sweep run)
       │
       ▼
 merge_model.py
       │
       ▼
./doctune-merged/       (standalone merged model, ready for deployment)
```

---

## Why merge instead of serving with adapters?

Serving with live adapters (via `PeftModel`) adds overhead at inference time: the
adapter weights are applied on every forward pass. Merging them into the base model
eliminates this overhead and produces a model that any standard HuggingFace inference
stack (vLLM, llama.cpp, TGI, Ollama) can load directly — no PEFT dependency needed
at runtime.

---

## Files

| File | Role |
|---|---|
| `merge_model.py` | CLI entry point for LoRA weight merging |
| `__init__.py` | Package marker (Phase 6 annotation) |

---

## `merge_model.py`

Fuses DPO-aligned LoRA adapters back into the base model weights using PEFT's
`merge_and_unload()`. The merged model and tokenizer are saved to an output directory
as a standard HuggingFace checkpoint.

### `parse_args() → Namespace`

| Flag | Default | Description |
|---|---|---|
| `--model-id` | *(required)* | HuggingFace base model identifier (e.g. `meta-llama/Llama-3.1-8B`) — must match the model used for SFT and DPO |
| `--adapter` | `./doctune-dpo` | Path to the DPO LoRA adapter directory to merge (the best run from `train_dpo.py`) |
| `--output` | `./doctune-merged` | Output directory for the merged standalone model |

---

### `main()` — merge orchestration

| Step | Action | Notes |
|---|---|---|
| 1 | `load_tokenizer(args.model_id)` | Loads the tokenizer |
| 2 | `load_base_model(args.model_id, tokenizer, device_map="cpu")` | Loads the base model onto CPU — avoids VRAM fragmentation from holding both the large base model and adapter in GPU memory simultaneously |
| 3 | `PeftModel.from_pretrained(base_model, args.adapter)` | Loads the LoRA adapter on top of the CPU-resident base model |
| 4 | `model.merge_and_unload()` | Fuses adapter weights into the base model and returns a standard `PreTrainedModel` with no PEFT dependency |
| 5 | `merged_model.save_pretrained(args.output)` + `tokenizer.save_pretrained(args.output)` | Saves the merged model and tokenizer; wrapped in `try/except OSError` so a failed write after a costly merge fails loudly |
| 6 | `del merged_model, model, base_model` + `clear_gpu_cache()` | Memory cleanup |

#### Why `device_map="cpu"`?

Loading the base model to CPU during merging prevents two large tensors (base model +
adapter deltas) from competing for VRAM simultaneously. `merge_and_unload()` is a
pure weight arithmetic operation that does not require GPU. The merged result is saved
directly from CPU to disk.

#### What `merge_and_unload()` does

PEFT's `merge_and_unload()` computes, for each LoRA-adapted layer:

```
W_merged = W_base + (lora_B @ lora_A) * (lora_alpha / r)
```

It then removes the adapter modules and returns a plain `PreTrainedModel` whose
weights are the fully-fused result. The returned model is structurally identical to
the original base model — same layer names, same `config.json` — making it
compatible with any standard HuggingFace loader.

---

### CLI example

```bash
python -m doctune.deploy.merge_model \
    --model-id meta-llama/Llama-3.1-8B \
    --adapter ./llama-3-1-8b-dpo-beta0.1-lr5e-06 \
    --output ./doctune-merged
```

The `--adapter` path should point to the **best run** identified by
`_log_sweep_summary` at the end of the DPO sweep. The sweep banner shows this path:

```
  Best adapter saved to: ./llama-3-1-8b-dpo-beta0.1-lr5e-06
```

---

## Output layout

After a successful merge, `--output` contains a standard HuggingFace checkpoint:

```
./doctune-merged/
    config.json
    generation_config.json
    model.safetensors        (or model-00001-of-NNNNN.safetensors for large models)
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
```

This directory can be loaded directly with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./doctune-merged")
tokenizer = AutoTokenizer.from_pretrained("./doctune-merged")
```

Or pushed to HuggingFace Hub:

```bash
huggingface-cli upload <your-org>/<model-name> ./doctune-merged
```

---

## Pipeline position

```
train_dpo.py        →    ./doctune-dpo/
                              │
                         merge_model.py
                              │
                         ./doctune-merged/
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          vLLM serve     Ollama push    HF Hub upload
```

---

## Future additions

The following steps belong in this package and are expected to be added as the
pipeline matures:

| Planned file | Purpose |
|---|---|
| `push_to_hub.py` | Authenticated upload to HuggingFace Hub with model card generation |
| `quantize.py` | GGUF / AWQ / GPTQ quantization for edge or CPU-only deployment |
| `serve.py` | Local vLLM or llama.cpp inference server wrapper for smoke-testing the merged model before push |
