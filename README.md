# EmotionEngine v2

Mechanistic interpretability study of **emotion and intensity representations** in large language models.  
We extract residual-stream activations from a frozen LLM, construct contrastive datasets via GPT-4, and identify linear subspaces encoding emotion category and emotional intensity — then validate them with linear probes and activation steering.

---

## Overview

```
scripts/          ← data-pipeline scripts (run in order below)
dataset/          ← processed datasets (CSVs / JSONL)
activation/       ← extracted residual-stream tensors  [not committed — see below]
analysis/         ← probe results, figures, steering outputs
notebook/         ← Jupyter notebooks for exploration and figures
paper/            ← LaTeX source
model.yaml        ← target model config (default: Llama-3.1-8B-Instruct)
```

---

## Pipeline

### 1  Build datasets

```bash
# Affective-rewrite triples (valence / arousal / dominance axes)
python scripts/build_affective_rewrite_dataset.py

# Emotion-category contrastive rewrites (joy/anger/fear/sadness × 3 intensities)
python scripts/build_emotion_rewrite_dataset.py
```

Both scripts call the **OpenAI API** (set `OPENAI_API_KEY` in `.env`).  
Outputs land in `dataset/affective_rewrites/` and `dataset/emotion_rewrites/`.

### 2  Extract residual-stream activations

```bash
# Affective rewrites → activation/affective_rewrites/residual_stream.pt
python scripts/extract_residual_stream.py \
    --data   dataset/affective_rewrites/affective_rewrites.jsonl \
    --config model.yaml \
    --out    activation/affective_rewrites/residual_stream.pt

# Emotion × intensity rewrites → activation/emotion_rewrites/
python scripts/extract_emotion_intensity_residual_stream.py \
    --data    dataset/activations_emotion_intensity \
    --config  model.yaml \
    --out-dir activation/emotion_rewrites
```

Output tensor shape: `(N, 8, 3, L, D)`  
 `N` sources · 8 emotions · 3 intensity levels · `L` layers · `D` hidden dim

### 3  Merge sharded activations (optional)

If extraction was parallelised into shards:

```bash
python scripts/merge_emotion_intensity_activations.py \
    --base-dir activation/emotion_rewrites \
    --merge-dir activation/emotion_rewrites_shard2
```

### 4  Analysis

Open the notebooks in `notebook/emotion_rewrites/` or `notebook/affective_rewrites/`:

| Notebook | Description |
|---|---|
| `dataset_statistics.ipynb` | Dataset size, label distributions |
| `activation_analysis.ipynb` | PCA / UMAP of residual streams |
| `residual_stream_analysis.ipynb` | Linear probe accuracy by layer |
| `steering_experiment.ipynb` | Activation-addition steering experiments |
| `caa_analysis.ipynb` | Contrastive Activation Addition vectors |
| `separability_analysis.ipynb` | Emotion vs. intensity subspace geometry |

---

## Setup

```bash
# Requires Python ≥ 3.11 and uv
uv sync               # creates .venv and installs all dependencies

# For CUDA 12.8 wheels (PyTorch):
uv sync --extra-index-url https://download.pytorch.org/whl/cu128
```

Copy `.env.example` to `.env` and fill in your API key:

```
OPENAI_API_KEY=sk-...
```

---

## Large files

Activation tensors (`*.pt`, `*.npy`, `*.mmap`) are **not committed** to this repo because they are 10–60 GB each.  
Download or reproduce them with the pipeline above.

External datasets (`dataset/dailydialog/`, `dataset/empatheticdialog/`) should be downloaded separately:

- **DailyDialog** — <https://huggingface.co/datasets/daily_dialog>
- **EmpatheticDialogues** — <https://huggingface.co/datasets/empathetic_dialogues>

---

## Model

Default target: **`meta-llama/Llama-3.1-8B-Instruct`** (see `model.yaml`).  
Requires a Hugging Face access token (`HF_TOKEN` in `.env`) and ~18 GB VRAM in bfloat16 or ~6 GB in 4-bit mode.

---

## License

[LICENSE](LICENSE)
