"""Evaluate the fine-tuned LoRA student model on the held-out test split.

Generates formal-academic translations for every slang input in the test
set and reports three metrics:

* Flesch-Kincaid grade level (reading complexity)
* Formality score (s-nlp/roberta-base-formality-ranker)
* Semantic preservation (BERTScore F1 against the slang input)

Per-example predictions and aggregate metrics are written to a JSON file.

Example:
    python scripts/evaluate_model.py \\
        --adapter-dir models/gemma-4-student-lora \\
        --test-data data/test_split.json \\
        --output-path results/student_evaluation_results.json
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
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = REPO_ROOT / "models" / "gemma-4-student-lora"
DEFAULT_TEST_DATA = REPO_ROOT / "data" / "test_split.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "student_evaluation_results.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model-id", default="google/gemma-4-e2b-it",
                        help="Base model identifier on the Hugging Face Hub.")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER,
                        help="Directory containing the trained LoRA adapters.")
    parser.add_argument("--test-data", type=Path, default=DEFAULT_TEST_DATA,
                        help="Path to the JSON test split.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to write the detailed JSON results.")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Maximum number of tokens to generate per example.")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading evaluation models (BERTScore & RoBERTa formality ranker)...")
    bertscore = evaluate.load("bertscore")
    formality_pipeline = pipeline(
        "text-classification",
        model="s-nlp/roberta-base-formality-ranker",
        device=0 if torch.cuda.is_available() else -1,
    )

    print("Loading fine-tuned LoRA model in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir))
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.eval()

    print(f"Loading test data from {args.test_data}...")
    with open(args.test_data, "r", encoding="utf-8") as f:
        raw_test_data = json.load(f)
    print(f"Found {len(raw_test_data)} test items. Beginning inference...")

    evaluated_data = []
    for item in tqdm(raw_test_data, desc="Translating test set"):
        slang_text = item["slang"]

        messages = [{"role": "user",
                     "content": f"Translate this contemporary internet slang to highly formal academic English: '{slang_text}'"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.3,
                repetition_penalty=1.1,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        evaluated_data.append({
            "slang": slang_text,
            "ground_truth_formal": item.get("formal", ""),
            "model_output": translation,
        })

    print("\nCalculating evaluation metrics...")
    slang_inputs = [item["slang"] for item in evaluated_data]
    model_outputs = [item["model_output"] for item in evaluated_data]

    # Flesch-Kincaid grade level.
    input_fk_scores = [textstat.flesch_kincaid_grade(text) for text in slang_inputs]
    output_fk_scores = [textstat.flesch_kincaid_grade(text) for text in model_outputs]
    avg_input_fk = sum(input_fk_scores) / len(input_fk_scores)
    avg_output_fk = sum(output_fk_scores) / len(output_fk_scores)

    # Formality.
    formality_results = formality_pipeline(model_outputs)
    formality_scores = [res["score"] if res["label"].lower() == "formal" else 1.0 - res["score"]
                        for res in formality_results]
    avg_formality = sum(formality_scores) / len(formality_scores)

    # Semantic preservation.
    b_scores = bertscore.compute(predictions=model_outputs, references=slang_inputs, lang="en")
    avg_bert_f1 = sum(b_scores["f1"]) / len(b_scores["f1"])

    print("Saving detailed results...")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "avg_input_fk": avg_input_fk,
                "avg_output_fk": avg_output_fk,
                "avg_formality": avg_formality,
                "avg_bert_f1": avg_bert_f1,
            },
            "predictions": evaluated_data,
        }, f, indent=4)

    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Total test samples evaluated:         {len(evaluated_data)}")
    print(f"Reading level shift (Flesch-Kincaid): Grade {avg_input_fk:.1f} -> Grade {avg_output_fk:.1f}")
    print(f"Target style accuracy (formality):    {avg_formality * 100:.2f}%")
    print(f"Semantic preservation (BERTScore F1): {avg_bert_f1:.3f}")
    print("=" * 50)
    print(f"Detailed translations saved to: {args.output_path}")


if __name__ == "__main__":
    main()
