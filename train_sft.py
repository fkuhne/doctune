"""
train_sft.py — Supervised Fine-Tuning (SFT)

Runs Phase 3 of the training pipeline: injects domain-specific knowledge
into any HuggingFace foundation model using LoRA adapters and the SFTTrainer.

Usage:
    python train_sft.py --model-id <huggingface-model-id> [--dataset DATASET_PATH] [--output OUTPUT_DIR]

Requirements:
    pip install -e ".[training]"
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer

from model_utils import detect_lora_target_modules, detect_attn_implementation, derive_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Doctune SFT Training")
    parser.add_argument("--model-id", type=str, required=True, help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)")
    parser.add_argument("--dataset", type=str, default="alignment_dataset.jsonl")
    parser.add_argument("--eval-dataset", type=str, default="golden_eval.jsonl")
    parser.add_argument("--output", type=str, default="./doctune-sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8,  help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    # 2. Load Base Model
    attn_impl = detect_attn_implementation()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    model.resize_token_embeddings(len(tokenizer))

    # 3. LoRA Configuration — auto-detect target modules from model architecture
    target_modules = detect_lora_target_modules(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        modules_to_save=["lm_head", "embed_tokens"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # 4. Load and Format Dataset
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset, split="train")

    def formatting_prompts_func(example: dict) -> list[str]:
        """Converts the JSON structure into conversational strings using the model's chat template."""
        output_texts = []
        for i in range(len(example["prompt"])):
            messages = [
                {"role": "user", "content": example["prompt"][i]},
                {"role": "assistant", "content": example["chosen"][i]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)
        return output_texts

    # 5. Hyperparameter Configuration
    run_name = derive_run_name(args.model_id, "sft")
    training_args = TrainingArguments(
        output_dir=args.output,
        run_name=run_name,
        report_to="mlflow",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="adamw_torch",
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
    )

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 7. Execute Training
    print("Initiating Supervised Fine-Tuning...")
    trainer.train()

    # 8. Save the Fine-Tuned Adapters
    trainer.save_model(args.output)
    print(f"Training complete. LoRA adapters saved to {args.output}")


if __name__ == "__main__":
    main()
