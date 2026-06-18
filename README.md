# Brainrot to Brilliance: Distilling a Synthetic Slang→Formal Corpus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code and data accompanying the paper on **synthetic dataset generation for
register transfer** — translating contemporary internet slang ("brainrot")
into highly formal academic English.

We use a large teacher LLM to synthesize a parallel corpus of
slang→formal pairs, then distill that knowledge into a small, locally
runnable student model (Gemma-4 2B) via parameter-efficient LoRA
fine-tuning. The student learns to substantially raise the **formality**
and **reading level** of an utterance while **preserving its semantics**.

> ⚠️ **Citation placeholder.** Update the [Citation](#citation) BibTeX with
> the final arXiv ID, author list, and title before camera-ready submission.

## Method

The pipeline has three stages:

1. **Synthesize** — A teacher model (Gemini) is prompted across rotating
   social-media micro-contexts to generate diverse slang→formal pairs.
   Pairs are deduplicated on the slang surface form (`generate_deduplicated_corpus.py`).
2. **Distill** — A 4-bit quantized Gemma-4 student is fine-tuned with LoRA
   adapters on the synthetic corpus (`finetune_gemma.py`).
3. **Evaluate** — The student and a zero-shot baseline are scored on reading
   level, formality, and semantic preservation
   (`evaluate_model.py`, `evaluate_baseline.py`).

## Dataset

`data/gold_standard_deduplicated.json` contains **10,000** deduplicated
slang→formal pairs, split (seed `42`) into a 9,500-pair training set and a
500-pair held-out test set.

Each record has two fields:

```json
{
  "slang": "touch grass",
  "formal": "The imperative to re-engage with empirical reality through somatic interaction with the terrestrial biosphere."
}
```

| File | Pairs | Description |
| --- | --- | --- |
| `data/gold_standard_deduplicated.json` | 10,000 | Full deduplicated corpus |
| `data/train_split.json` | 9,500 | Training split |
| `data/test_split.json` | 500 | Held-out evaluation split |

## Results

Evaluated on the 500-pair held-out test set. Formality is the probability
assigned by [`s-nlp/roberta-base-formality-ranker`](https://huggingface.co/s-nlp/roberta-base-formality-ranker);
semantic preservation is BERTScore F1 against the slang input; reading level
is the Flesch–Kincaid grade.

| Metric | Input slang | Base model (zero-shot) | **Fine-tuned student** |
| --- | --- | --- | --- |
| Reading level (FK grade) | 6.3 | 16.0 | **14.4** |
| Formality | 39.66% | 90.31% | **94.49%** |
| Semantic preservation (BERTScore F1) | — | 0.828 | **0.864** |

The distilled student surpasses the zero-shot base model on both formality
and semantic preservation while producing slightly more readable output.
Full per-example predictions are in `results/student_evaluation_results.json`
and `results/baseline_evaluation_results.txt`.

## Repository structure

```
.
├── data/                  # Synthetic corpus and train/test splits
├── results/               # Evaluation outputs reported in the paper
├── scripts/
│   ├── generate_deduplicated_corpus.py   # Stage 1: teacher distillation
│   ├── split_data.py                     # Train/test split
│   ├── finetune_gemma.py                 # Stage 2: LoRA fine-tuning
│   ├── evaluate_baseline.py              # Zero-shot baseline metrics
│   ├── evaluate_model.py                 # Student model metrics
│   └── test_model.py                     # Interactive translation demo
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

A CUDA-capable GPU is required for training and 4-bit inference.

```bash
git clone https://github.com/siwooy/synthetic-dataset-research-paper.git
cd synthetic-dataset-research-paper
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Access to the gated `google/gemma-4-e2b-it` model on the Hugging Face Hub is
required; run `huggingface-cli login` after accepting the model license.

## Usage

Every script reads its defaults relative to the repository root and exposes
overridable arguments via `--help`.

**1. Generate the synthetic corpus** (optional — the corpus is included).
Requires a `GOOGLE_API_KEY` (a `.env` file is supported):

```bash
python scripts/generate_deduplicated_corpus.py --target-samples 10000
```

**2. Create the train/test split** (optional — splits are included):

```bash
python scripts/split_data.py --test-size 500 --seed 42
```

**3. Fine-tune the student model:**

```bash
python scripts/finetune_gemma.py \
    --dataset-path data/train_split.json \
    --output-dir models/gemma-4-student-lora
```

**4. Evaluate:**

```bash
python scripts/evaluate_baseline.py --output-path results/baseline_evaluation_results.txt
python scripts/evaluate_model.py   --adapter-dir models/gemma-4-student-lora
```

**5. Try it interactively:**

```bash
python scripts/test_model.py "Bro is cooked." "That take is mid."
```

## Models & data licensing

- **Code** is released under the [MIT License](LICENSE).
- The **synthetic corpus** was generated by a teacher LLM; downstream use is
  subject to the teacher provider's terms of service.
- The student model fine-tunes `google/gemma-4-e2b-it`, which is governed by
  the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## Citation

```bibtex
@misc{yun2026brainrot,
  title        = {TODO: Final Paper Title},
  author       = {Yun, Siwoo},
  year         = {2026},
  eprint       = {TODO.XXXXX},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/TODO.XXXXX}
}
```
