"""Interactive demo for the fine-tuned slang-to-formal translator.

Loads the base Gemma-4 model in 4-bit with the trained LoRA adapters
attached and translates one or more slang phrases to formal academic
English. Slang phrases may be passed as positional arguments; if none are
given, a small set of built-in examples is used.

Example:
    python scripts/test_model.py "Bro is cooked." "That take is mid."
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER = REPO_ROOT / "models" / "gemma-4-student-lora"

EXAMPLE_PHRASES = [
    "Bro is a bot.",
    "Drake's hate is so forced.",
    "I'm locking in for this project.",
    "Your rizz is completely washed. You can't even pull one huzz.",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phrases", nargs="*", default=None,
                        help="Slang phrases to translate. Defaults to built-in examples.")
    parser.add_argument("--base-model-id", default="google/gemma-4-e2b-it",
                        help="Base model identifier on the Hugging Face Hub.")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER,
                        help="Directory containing the trained LoRA adapters.")
    parser.add_argument("--max-new-tokens", type=int, default=150,
                        help="Maximum number of tokens to generate per phrase.")
    return parser.parse_args()


def main():
    args = parse_args()
    phrases = args.phrases if args.phrases else EXAMPLE_PHRASES

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir))

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, device_map={"": 0}
    )

    print("Attaching LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model.eval()

    def translate_slang(slang_text):
        messages = [
            {"role": "user",
             "content": f"Translate this contemporary internet slang to highly formal academic English: '{slang_text}'"}
        ]
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

        print(f"User:  {slang_text}")
        print(f"Model: {translation}\n")
        print("-" * 50 + "\n")

    print("\n" + "=" * 50)
    print("ACADEMIC TRANSLATOR")
    print("=" * 50 + "\n")

    for phrase in phrases:
        translate_slang(phrase)


if __name__ == "__main__":
    main()
