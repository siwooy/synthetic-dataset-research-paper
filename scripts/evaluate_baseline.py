"""Compute zero-shot baseline metrics for the slang-to-formal task.

Reports two reference points used in the paper:

1. The original slang inputs (reading level and formality of the raw data).
2. The base Gemma-4 model with no LoRA adapters (zero-shot translation),
   scored on reading level, formality, and BERTScore F1.

By default the report is printed to stdout. Pass --output-path to also write
it to a text file.

Example:
    python scripts/evaluate_baseline.py \\
        --test-data data/test_split.json \\
        --output-path results/baseline_evaluation_results.txt
"""

import argparse
import json
from pathlib import Path

import torch
import textstat
import evaluate
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_DATA = REPO_ROOT / "data" / "test_split.json"


def formality_score(result):
    """Return P(formal) from a formality-ranker classification result."""
    if result["label"].lower() in ("formal", "label_1"):
        return result["score"]
    return 1.0 - result["score"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model-id", default="google/gemma-4-e2b-it",
                        help="Base model identifier on the Hugging Face Hub.")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA,
                        help="Path to the JSON test split.")
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Optional path to write the report as text.")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Maximum number of tokens to generate per example.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading evaluators...")
    bertscore = evaluate.load("bertscore")
    formality_pipeline = pipeline(
        "text-classification",
        model="s-nlp/roberta-base-formality-ranker",
        device=0 if torch.cuda.is_available() else -1,
    )

    print("Loading base model (no LoRA)...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map={"": 0}
    )
    base_model.eval()

    with open(args.test_data, "r", encoding="utf-8") as f:
        raw_test_data = json.load(f)

    # 1. Evaluate the original slang inputs.
    print("\nCalculating original dataset baselines...")
    slang_inputs = [item["slang"] for item in raw_test_data]
    slang_fk = sum(textstat.flesch_kincaid_grade(text) for text in slang_inputs) / len(slang_inputs)
    slang_formality_results = formality_pipeline(slang_inputs)
    avg_slang_formality = sum(formality_score(r) for r in slang_formality_results) / len(slang_formality_results)

    # 2. Evaluate the base model zero-shot.
    base_model_outputs = []
    for item in tqdm(raw_test_data, desc="Base model translating"):
        messages = [{"role": "user",
                     "content": f"Translate this contemporary internet slang to highly formal academic English: '{item['slang']}'"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

        with torch.no_grad():
            outputs = base_model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, temperature=0.3, repetition_penalty=1.1
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        base_model_outputs.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))

    print("\nCalculating base model metrics...")
    base_fk = sum(textstat.flesch_kincaid_grade(text) for text in base_model_outputs) / len(base_model_outputs)
    base_formality_results = formality_pipeline(base_model_outputs)
    avg_base_formality = sum(formality_score(r) for r in base_formality_results) / len(base_formality_results)
    b_scores = bertscore.compute(predictions=base_model_outputs, references=slang_inputs, lang="en")
    avg_base_bert = sum(b_scores["f1"]) / len(b_scores["f1"])

    report = (
        "=" * 50 + "\n"
        f"Original Slang FK: Grade {slang_fk:.1f}\n"
        f"Original Slang Formality: {avg_slang_formality * 100:.2f}%\n\n"
        f"Base Model FK: Grade {base_fk:.1f}\n"
        f"Base Model Formality: {avg_base_formality * 100:.2f}%\n"
        f"Base Model BERTScore: {avg_base_bert:.3f}\n"
        + "=" * 50
    )

    print("\n" + report)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\nReport saved to: {args.output_path}")


if __name__ == "__main__":
    main()
