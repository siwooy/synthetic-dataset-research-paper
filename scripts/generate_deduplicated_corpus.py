"""Generate the synthetic slang-to-formal corpus via teacher-model distillation.

Repeatedly prompts a teacher LLM (Gemini) to produce batches of
slang-to-formal translation pairs, deduplicating on the slang surface form
and checkpointing progress to disk after every batch. Generation continues
until the target corpus size is reached.

Requires a GOOGLE_API_KEY environment variable (a .env file is supported).

Example:
    python scripts/generate_deduplicated_corpus.py --target-samples 10000
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "gold_standard_deduplicated.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-samples", type=int, default=10000,
                        help="Target number of unique pairs to accumulate.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH,
                        help="Path to the corpus JSON file (read and appended to).")
    parser.add_argument("--model-id", default="gemini-3.1-flash-lite-preview",
                        help="Teacher model identifier.")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Seconds to wait between successful batches.")
    return parser.parse_args()


def get_client():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found")
        raise SystemExit(1)
    return genai.Client(api_key=api_key)


def load_existing_data(data_path):
    if not os.path.exists(data_path):
        return [], set()

    with open(data_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            seen_slang = {item["slang"].lower().strip() for item in data}
            return data, seen_slang
        except json.JSONDecodeError:
            return [], set()


def save_data(data, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def get_generation_prompt():
    """Build a prompt that uses micro-contexts to elicit compound slang phrases."""
    scenarios = [
        "leaving a comment on a chaotic TikTok 'brainrot' edit",
        "arguing in an Instagram Reels comment section about who has negative aura",
        "reacting to a streamer (like Kai Cenat, IShowSpeed, or AMP) doing something crazy live",
        "gossiping in a Snapchat group chat using maximum Gen Alpha vocabulary",
        "explaining 'skibidi' lore or 'sigma' mindset concepts in a YouTube Shorts comment",
        "hyping someone up using strictly TikTok viral audio terminology",
        "describing a situation where someone was 'absolutely cooked' or 'fanum taxed'",
        "using 'looksmaxxing' or 'mewing' terminology to judge someone's appearance",
    ]

    structures = [
        "two-word brainrot colloquial combinations",
        "full sentence reactions relying on extreme Gen Alpha slang",
        "TikTok and Instagram comment section shorthand",
        "exaggerated streamer catchphrases and chat spam",
    ]

    selected_scenario = random.choice(scenarios)
    selected_structure = random.choice(structures)

    return (
        f"Generate exactly 50 unique slang-to-formal translation pairs. "
        f"SCENARIO: Focus entirely on slang used when {selected_scenario}. "
        f"STRUCTURE: The slang should specifically be {selected_structure}. "
        f"CRITICAL: The slang MUST be pure Gen Z / Gen Alpha 'brainrot', TikTok audio trends, or YouTube/Twitch streamer terminology. Do NOT use outdated 2010s internet slang. "
        f"The formal translation must be highly academic, robotic, and excessively formal (like an Oxford dictionary). "
        f"Output strictly as a JSON array of objects with keys 'slang' and 'formal'."
    )


def run_distillation_pipeline(args):
    client = get_client()
    corpus, seen_slang = load_existing_data(args.data_path)

    print("Starting distillation pipeline...")
    print(f"Corpus size: {len(corpus)} / {args.target_samples}")

    while len(corpus) < args.target_samples:
        prompt = get_generation_prompt()

        try:
            print("\nRequesting batch from teacher model...")
            response = client.models.generate_content(
                model=args.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.9,
                ),
            )

            new_batch = json.loads(response.text)
            added_this_batch = 0

            for item in new_batch:
                slang = item.get("slang", "").strip()
                formal = item.get("formal", "").strip()

                if not slang or not formal:
                    continue

                if slang.lower() not in seen_slang:
                    corpus.append({"slang": slang, "formal": formal})
                    seen_slang.add(slang.lower())
                    added_this_batch += 1

            save_data(corpus, args.data_path)
            print(f"Batch processed. Unique pairs added: {added_this_batch}")
            print(f"Total unique pairs: {len(corpus)} / {args.target_samples}")

            time.sleep(args.cooldown)

        except APIError as e:
            if "401" in str(e) or "403" in str(e) or "API_KEY_INVALID" in str(e):
                print(f"\nAPI key rejected: {e}")
                break
            print(f"API error: {e}. Retrying in 15s.")
            time.sleep(15)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in 15s.")
            time.sleep(15)


if __name__ == "__main__":
    run_distillation_pipeline(parse_args())
