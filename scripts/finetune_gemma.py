"""Fine-tune Gemma-4 with LoRA for slang-to-formal academic English translation.

This script performs supervised fine-tuning (SFT) of a 4-bit quantized
Gemma-4 model using LoRA adapters on the synthetic colloquial-to-formal
dataset. It corresponds to the "student" stage of the knowledge-distillation
pipeline described in the accompanying paper.

Example:
    python scripts/finetune_gemma.py \\
        --dataset-path data/train_split.json \\
        --output-dir models/gemma-4-student-lora
"""

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "train_split.json"
DEFAULT_OUTPUT = REPO_ROOT / "models" / "gemma-4-student-lora"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="google/gemma-4-e2b-it",
                        help="Base model identifier on the Hugging Face Hub.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET,
                        help="Path to the JSON training split.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Directory in which to save the trained LoRA adapters.")
    parser.add_argument("--cuda-device", default="0",
                        help="CUDA device index exposed via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum number of optimization steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Peak learning rate.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device training batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16,
                        help="Number of gradient accumulation steps.")
    return parser.parse_args()


def main():
    args = parse_args()

    # CUDA_VISIBLE_DEVICES must be set before torch is imported.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    print("Loading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.padding_side = "right"

    def format_dataset(example):
        prompt_messages = [
            {"role": "user",
             "content": f"Translate this contemporary internet slang to highly formal academic English: '{example['slang']}'"}
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "completion": example["formal"]}

    print("Pre-processing dataset...")
    dataset = load_dataset("json", data_files=str(args.dataset_path), split="train")
    dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=5,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=args.max_steps,
        warmup_steps=15,
        lr_scheduler_type="cosine",
        completion_only_loss=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Beginning fine-tuning...")
    trainer.train()

    trainer.model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Fine-tuning complete. LoRA adapters saved to {args.output_dir}")


if __name__ == "__main__":
    main()
