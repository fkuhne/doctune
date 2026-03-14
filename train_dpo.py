"""
train_dpo.py — Direct Preference Optimization (DPO)

Runs Phase 4 of the training pipeline: aligns the SFT-trained model using
preference tuples (chosen vs. rejected) to penalize hallucinations.

Supports hyperparameter sweeping over beta and learning rate values.

Usage:
    python train_dpo.py --model-id <huggingface-model-id> [--sft-adapter SFT_PATH] [--dataset DATASET_PATH]

Requirements:
    pip install -e ".[training]"
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer

from model_utils import detect_attn_implementation, derive_run_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Doctune DPO Training")
    parser.add_argument("--model-id", type=str, required=True, help="HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)")
    parser.add_argument("--sft-adapter", type=str, default="./doctune-sft")
    parser.add_argument("--dataset", type=str, default="alignment_dataset.jsonl")
    parser.add_argument("--eval-dataset", type=str, default="golden_eval.jsonl")
    parser.add_argument("--output", type=str, default="./doctune-dpo")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.25], help="DPO beta values to sweep")
    parser.add_argument("--lrs", type=float, nargs="+", default=[5e-6, 1e-6], help="Learning rates to sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = "right"

    # 2. Load Base Model and SFT Adapters
    print("Loading Base Model and SFT Adapters...")
    attn_impl = detect_attn_implementation()
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, args.sft_adapter, is_trainable=True)

    # 3. Load and Format Dataset
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_dataset, split="train")

    def format_dpo_dataset(example: dict) -> dict:
        """DPOTrainer expects strings for prompt, chosen, and rejected."""
        prompt_messages = [{"role": "user", "content": example["prompt"]}]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": formatted_prompt,
            "chosen": f"{example['chosen']}{tokenizer.eos_token}",
            "rejected": f"{example['rejected']}{tokenizer.eos_token}",
        }

    dpo_dataset = dataset.map(format_dpo_dataset)
    dpo_eval_dataset = eval_dataset.map(format_dpo_dataset)

    # 4. DPO Training Function
    base_run_name = derive_run_name(args.model_id, "dpo")

    def train_dpo(beta_val: float, lr_val: float) -> None:
        run_name = f"{base_run_name}-beta{beta_val}-lr{lr_val}"
        output_dir = f"./{run_name}"
        print(f"\n--- Starting DPO Sweep: Beta={beta_val}, LR={lr_val} ---")

        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            report_to="mlflow",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=lr_val,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            bf16=True,
            max_grad_norm=0.3,
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=training_args,
            train_dataset=dpo_dataset,
            eval_dataset=dpo_eval_dataset,
            tokenizer=tokenizer,
            beta=beta_val,
            max_length=2048,
            max_prompt_length=1024,
        )

        print(f"Initiating DPO Training for {run_name}...")
        trainer.train()

        trainer.save_model(output_dir)
        print(f"Alignment complete for {run_name}. Saved to {output_dir}")

    # 5. Execute DPO Hyperparameter Sweep
    for beta in args.betas:
        for lr in args.lrs:
            train_dpo(beta, lr)


if __name__ == "__main__":
    main()
