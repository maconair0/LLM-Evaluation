# LLM-Evaluation

Optical Networking LLM evaluation for models trained for optics.

Evaluation code for comparing base and LoRA fine-tuned LLMs (LLaMA 2, LLaMA 3.1, Mistral 7B) on:

- **OpED** (Optical Network Evaluation Dataset): 89 multiple-choice questions on optical networking. The script reports accuracy and an "honesty" rate (how often the model abstains with IDK instead of guessing).
- **MMLU** and **ARC-Challenge**: general-knowledge baselines, to check that fine-tuning did not degrade general ability.
- **Perplexity (approximate)**: a next-word prediction proxy on a short optical-networking passage.

All models are served locally through [Ollama](https://ollama.com).

## Repository contents

| File | Purpose |
|------|---------|
| `LLM_evaluation_script.py` | Runs every benchmark, prints the results table and plots the OpED figure |
| `OpED.json` | The OpED question set (89 questions) |
| `README.md` | This guide |
| `LICENSE` | License terms |

`mmlu_test.json` and `arc_test.json` are **not** included. You generate them yourself (step 4).

---

## 1. Requirements

- Linux, macOS or WSL2
- Python 3.10+
- An NVIDIA GPU is strongly recommended. Each run makes thousands of Ollama calls: OpED alone makes 8 answer calls plus 8 judge calls per question, per model.
- About 30 GB of free disk space for six 7B/8B models

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install requests numpy matplotlib tqdm datasets
```

## 2. Install and start Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve            # skip this if Ollama already runs as a system service
curl http://localhost:11434/api/tags   # check that the server responds
```

The script expects Ollama at `http://localhost:11434`. To use a different host, edit `OLLAMA_API_URL` near the top of `LLM_evaluation_script.py`.

## 3. Set up the models

The script looks models up by the Ollama names in `ALL_MODELS`. Each model must exist under **exactly** that name:

| Display name | Ollama name | Where it comes from |
|--------------|-------------|---------------------|
| LLaMA 2 Base | `llama-2-7b-chat` | Base model (see 3a) |
| LLaMA 2 FT   | `llama-2-lora`    | Your LoRA GGUF (see 3b) |
| LLaMA 3 Base | `llama3.1`        | `ollama pull llama3.1` |
| LLaMA 3 FT   | `llama-3-lora`    | Your LoRA GGUF (see 3b) |
| Mistral Base | `mistral`         | `ollama pull mistral` |
| Mistral FT   | `mistral-lora`    | Your LoRA GGUF (see 3b) |

`mistral` is also the **judge model** that grades OpED answers (`JUDGE_MODEL`), so it must be installed even if you do not evaluate it.

### 3a. Base models

```bash
ollama pull llama3.1
ollama pull mistral
ollama pull llama2:7b-chat
ollama cp llama2:7b-chat llama-2-7b-chat    # rename to the name the script expects
```

### 3b. Fine-tuned (LoRA) models

The fine-tuned models are GGUF files. They are produced by merging a trained LoRA adapter into its base model and converting the result with `llama.cpp`'s `convert_hf_to_gguf.py`. The weights are not in this repository. Ask the author for them, or build your own.

For each GGUF file, write a `Modelfile`:

```text
FROM /absolute/path/to/llama-2-7b-lora_new.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

Then register the model under the expected name:

```bash
ollama create llama-2-lora -f Modelfile_llama2
ollama create llama-3-lora -f Modelfile_llama3
ollama create mistral-lora -f Modelfile_mistral
ollama list    # check that all six names appear
```

The script sets `temperature` in each request, so that request value takes precedence over the Modelfile's `temperature`.

## 4. Get the MMLU and ARC datasets

The script reads `mmlu_test.json` and `arc_test.json` from the current directory. **If either file is missing, it silently uses placeholder questions and the scores are meaningless.**

Both files come from Hugging Face (`cais/mmlu` and `allenai/ai2_arc`). Run this in the repo directory to create them (100 questions each, fixed seed):

```bash
python3 - <<'EOF'
import json, random
from datasets import load_dataset
L = ["A", "B", "C", "D"]
N, SEED = 100, 42

# MMLU: random sample from the full test split
ds = load_dataset("cais/mmlu", "all", split="test")
idx = random.Random(SEED).sample(range(len(ds)), N)
mmlu = [{"id": i, "category": ds[j]["subject"], "question": ds[j]["question"],
         "options": {L[k]: c for k, c in enumerate(ds[j]["choices"])},
         "answer": L[ds[j]["answer"]]} for i, j in enumerate(idx)]
json.dump(mmlu, open("mmlu_test.json", "w"), indent=2)

# ARC-Challenge: keep only 4-option questions, relabel them A-D
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
arc = []
for r in ds:
    labels, texts = r["choices"]["label"], r["choices"]["text"]
    if len(labels) != 4 or r["answerKey"] not in labels:
        continue
    arc.append({"question": r["question"],
                "options": {L[k]: t for k, t in enumerate(texts)},
                "answer": L[labels.index(r["answerKey"])]})
random.Random(SEED).shuffle(arc)
arc = [{"id": i, "category": "ARC-Challenge", **q} for i, q in enumerate(arc[:N])]
json.dump(arc, open("arc_test.json", "w"), indent=2)
print(len(mmlu), len(arc))
EOF
```

If Hugging Face requires a login on your network, run `huggingface-cli login` first.

### Dataset format

All three datasets use the same structure:

```json
{
  "id": 1,
  "category": "Physical Layer",
  "question": "A span has 0.22 dB/km attenuation over 80 km. What is the approximate span loss?",
  "options": {"A": "11 dB", "B": "17.6 dB", "C": "22 dB", "D": "35 dB"},
  "answer": "B"
}
```

In OpED, `category` is one of `Physical Layer`, `DSP`, `Architecture`, `Hardware` or `SDN/NFV`.

## 5. Run the evaluation

```bash
# Full run: all 6 models, all benchmarks (slow; can take many hours)
python3 LLM_evaluation_script.py

# OpED only
python3 LLM_evaluation_script.py --oped-only

# Quick check with a subset of models and fewer samples
python3 LLM_evaluation_script.py --models mistral mistral-lora \
    --n-shots 3 --mmlu-samples 20 --arc-samples 20
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--models` | all | Ollama or display names to evaluate |
| `--oped-only` | off | Skip MMLU, ARC and perplexity |
| `--n-shots` | 8 | Answers sampled per OpED question (majority vote) |
| `--mmlu-samples` / `--arc-samples` | 100 | Questions used from each benchmark |
| `--oped-file` | `OpED.json` | Path to the OpED dataset |
| `--out-prefix` | `oped_figure_c` | File name of the output figure |

## 6. Outputs

| File | Contents |
|------|----------|
| `eval_checkpoint.json` | Results so far, saved after each model finishes |
| `eval_results_all.json` | Final results for every model |
| `eval_summary_table.txt` | Results table with base-vs-fine-tuned deltas |
| `oped_figure_c.png` | OpED accuracy vs. honesty chart |

The figure layout assumes the default six models (three families, each with a Base and an FT variant).

## Troubleshooting

- **`[ERROR] Ollama call failed`**: Ollama is not running, or a model name does not match. Compare `ollama list` with the table in step 3.
- **Every MMLU/ARC score is about 0.25 or identical**: `mmlu_test.json` or `arc_test.json` is missing, so the script used placeholder questions. Redo step 4.
- **Proxy problems**: the script removes proxy environment variables for localhost calls. Download models and datasets *before* you run it.
- **Timeouts**: each request times out after 180 s. Use a GPU, or reduce `--n-shots`.
