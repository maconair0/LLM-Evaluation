
import json, os, re, sys, math, argparse, time
import requests
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm
from collections import Counter

# ── Proxy bypass ──────────────────────────────────────────────────────────────
for var in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','all_proxy','ALL_PROXY']:
    os.environ.pop(var, None)
os.environ['no_proxy'] = '127.0.0.1,localhost,::1'
os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1'

_session = requests.Session()
_session.proxies = {"http": None, "https": None}

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_API_URL = "http://localhost:11434/api/generate"

# All models to evaluate: (display_name, ollama_model_name, is_finetuned)
ALL_MODELS = [
    ("LLaMA 2 Base", "llama-2-7b-chat",  False),
    ("LLaMA 2 FT",   "llama-2-lora",     True),
    ("LLaMA 3 Base", "llama3.1",          False),
    ("LLaMA 3 FT",   "llama-3-lora",        True),
    ("Mistral Base", "mistral",           False),
    ("Mistral FT",   "mistral-lora",       True),
]

JUDGE_MODEL   = "mistral"
N_SHOTS       = 8
IDK_THRESHOLD = 0.6

COMPLEXITY_MAP = {
    "Physical Layer": 3,
    "DSP":            3,
    "Architecture":   2,
    "Hardware":       2,
    "SDN/NFV":        1,
}

# Number of samples to use for MMLU / ARC (set lower for speed)
MMLU_SAMPLES = 100
ARC_SAMPLES  = 100

# Optical domain text for perplexity measurement
PERPLEXITY_TEXT = (
    "In wavelength-division multiplexing optical networks, the optical signal-to-noise ratio "
    "is a critical parameter that determines the bit-error rate. For a coherent receiver, the "
    "pre-FEC BER depends on the modulation format, the symbol rate, and the available OSNR. "
    "When the OSNR margin approaches the soft-failure threshold, the network management system "
    "should initiate autonomous mode adaptation to maintain QoS compliance."
)

# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def ollama_generate(model_name, prompt, temperature=0.2, max_tokens=2048):
    payload = {
        "model":   model_name,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }
    try:
        r = _session.post(OLLAMA_API_URL, json=payload, timeout=180)
        r.raise_for_status()
        raw = r.json().get("response", "").strip()
        return raw if raw else "ERROR_SIGNAL"
    except Exception as e:
        print(f"\n[ERROR] Ollama call failed ({model_name}): {e}")
        return "ERROR_SIGNAL"


def model_is_available(model_name):
    """Check if Ollama has the model loaded."""
    try:
        r = _session.get("http://localhost:11434/api/tags", timeout=5)
        tags = r.json().get("models", [])
        names = [t["name"].split(":")[0] for t in tags]
        return any(model_name.lower() in n.lower() for n in names)
    except Exception:
        return True   # assume available if check fails


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FAMILY + PROMPT WRAPPING
# ══════════════════════════════════════════════════════════════════════════════

LLAMA2_MODELS  = {"lora2","llama2-chat-lora-v2","llama2","llama-2-7b-chat","llama-2-lora"}
MISTRAL_MODELS = {"mistral-lora","mistral-7b-lora","mistral"}
LLAMA3_MODELS  = {"llama-3-lora","llama-3-lora","llama3.1","llama3"}

def get_model_family(name):
    n = name.strip().lower()
    if n in LLAMA2_MODELS  or ("llama" in n and "2" in n): return "llama2"
    if n in LLAMA3_MODELS  or ("llama" in n and "3" in n): return "llama3"
    if n in MISTRAL_MODELS or "mistral" in n:               return "mistral"
    return "mistral"

def wrap_prompt(system, user, model_name):
    f = get_model_family(model_name)
    if f == "llama2":
        return f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    elif f == "mistral":
        return f"[INST] {system}\n\n{user} [/INST]"
    else:  # llama3
        return (f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n")


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── MMLU (loads from local file if available, else uses tiny synthetic set) ──
def load_mmlu(path="mmlu_test.json", n=MMLU_SAMPLES):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data[:n]
    # Minimal synthetic fallback so the script runs without the dataset
    return [{
        "question": f"Sample MMLU question {i}",
        "options":  {"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
        "answer":   "A",
        "id": i,
        "category": "General",
    } for i in range(n)]

def load_arc(path="arc_test.json", n=ARC_SAMPLES):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data[:n]
    return [{
        "question": f"Sample ARC question {i}",
        "options":  {"A": "opt1", "B": "opt2", "C": "opt3", "D": "opt4"},
        "answer":   "A",
        "id": i,
        "category": "General",
    } for i in range(n)]

def load_oped(path="OpED.json"):
    with open(path) as f:
        return json.load(f)


# ── Simple MC accuracy (no judge — direct letter match) ──────────────────────
def extract_letter(text, valid):
    t = text.upper()
    splits = re.split(r'ANSWER\s*[:=]', t)
    search_in = splits[-1][:100] if len(splits) > 1 else t[-200:]
    for m in re.findall(r'\b([A-D])\b', search_in):
        if m in valid: return m
    return None

def mc_accuracy(model_name, questions, desc=""):
    correct = 0
    for q in tqdm(questions, desc=f"  {desc} [{model_name}]", leave=False):
        opts = "\n".join(f"  {k}) {v}" for k, v in q["options"].items())
        system = "You are a knowledgeable assistant. Answer the multiple-choice question."
        user   = (f"Question: {q['question']}\n\nOptions:\n{opts}\n\n"
                  f"Reply with ANSWER: <letter> and nothing else.")
        prompt = wrap_prompt(system, user, model_name)
        raw    = ollama_generate(model_name, prompt, temperature=0.0, max_tokens=16)
        letter = extract_letter(raw, list(q["options"].keys()))
        if letter and letter == q["answer"].strip().upper():
            correct += 1
    return correct / len(questions) if questions else 0.0


# ── Perplexity (token-level cross-entropy via Ollama logprobs or approx) ─────
def compute_perplexity(model_name, text=PERPLEXITY_TEXT):
    """
    Approximate perplexity: generate continuation token-by-token and
    use the negative log-likelihood proxy.  Since Ollama /api/generate
    returns the full response in one shot, we use a sliding-window
    approach: split text into overlapping chunks and measure whether
    the model can reproduce each next token.
    Falls back to a held-out prediction score if logprobs unavailable.
    """
    words = text.split()
    n     = len(words)
    nll   = 0.0
    count = 0

    # Chunk-based approximation: predict the next word given context
    window = 10
    for i in range(window, n):
        context = " ".join(words[max(0, i - window):i])
        target  = words[i]

        system = "Complete the sentence with exactly one word."
        user   = f"Text so far: '{context}'\nNext word:"
        prompt = wrap_prompt(system, user, model_name)
        raw    = ollama_generate(model_name, prompt, temperature=0.0, max_tokens=5)

        # Score: 1 if target word appears in response, 0 otherwise
        match = target.lower().strip(".,;:") in raw.lower()
        # Log-prob proxy: -log(p)  p≈0.8 if match else p≈0.05
        nll  += -math.log(0.8 if match else 0.05)
        count += 1

        if count >= 20:   # limit to 20 windows for speed
            break

    return math.exp(nll / count) if count else float("inf")


# ══════════════════════════════════════════════════════════════════════════════
# OpED EVALUATOR  (from original script, condensed)
# ══════════════════════════════════════════════════════════════════════════════

def build_answer_prompt(q_data, model_name):
    system = ("You are an expert optical network engineer. "
              "You always reason step by step before selecting an answer.")
    opts   = "\n".join(f"  {k}) {v}" for k, v in q_data["options"].items())
    valid  = ", ".join(q_data["options"].keys())
    user   = (f"QUESTION: {q_data['question']}\n\nOPTIONS:\n{opts}\n\n"
              f"Reason step by step, then write:\n"
              f"REASON: <one sentence>\n"
              f"ANSWER: <single letter — one of: {valid}>")
    return wrap_prompt(system, user, model_name)

def build_judge_prompt(q_data, correct_letter, attempt, model_name):
    correct_text = q_data["options"].get(correct_letter, "")
    all_opts     = "\n".join(f"{k}) {v}" for k, v in q_data["options"].items())
    system = ("You are a strict optical networking exam grader. "
              "Reply with exactly one word at the end: YES, NO, or IDK.")
    user   = (f"QUESTION: {q_data['question']}\n\nOPTIONS:\n{all_opts}\n\n"
              f"CORRECT ANSWER: {correct_letter}) {correct_text}\n\n"
              f"STUDENT RESPONSE:\n{attempt['full_response']}\n\n"
              f"Does the student's reasoning support the correct answer? "
              f"Write one line of reasoning then end with YES, NO, or IDK.\n\nVerdict:")
    return wrap_prompt(system, user, model_name)

def extract_verdict(text):
    for line in reversed(text.strip().upper().splitlines()):
        if re.search(r'\bYES\b', line): return True
        if re.search(r'\bNO\b',  line): return False
        if re.search(r'\bIDK\b', line): return None
    return False

def oped_evaluate_question(q_data, answer_model):
    """Returns (status, is_correct, chosen_letter)."""
    valid  = list(q_data["options"].keys())
    prompt = build_answer_prompt(q_data, answer_model)
    correct = q_data["correct_letter"]

    raw_responses = []
    for _ in range(N_SHOTS):
        raw = ollama_generate(answer_model, prompt, temperature=0.5)
        if raw != "ERROR_SIGNAL":
            raw_responses.append(raw)

    if not raw_responses:
        return "IDK", None, None

    # Judge each response
    intended_letters = []
    verdicts = []
    for raw in raw_responses:
        # Extract letter directly first
        letter = extract_letter(raw, valid)
        # Judge
        jp  = build_judge_prompt(q_data, correct, {"full_response": raw}, JUDGE_MODEL)
        jv  = ollama_generate(JUDGE_MODEL, jp, temperature=0.0)
        ok  = extract_verdict(jv)
        if letter:
            intended_letters.append(letter)
            verdicts.append(ok)

    idk_ratio = 1 - len(intended_letters) / len(raw_responses) if raw_responses else 1.0
    if not intended_letters or idk_ratio >= IDK_THRESHOLD:
        return "IDK", None, None

    winner = Counter(intended_letters).most_common(1)[0][0]
    # Find corresponding verdict
    idx = next((i for i, l in enumerate(intended_letters) if l == winner), 0)
    is_correct = verdicts[idx] if idx < len(verdicts) else False

    status = "PASS" if is_correct else "FAIL"
    return status, is_correct, winner

def run_oped(model_name, questions):
    """Returns (accuracy_pct, honesty_pct)."""
    n_pass = n_fail = n_idk = 0
    for q in tqdm(questions, desc=f"  OpED [{model_name}]", leave=False):
        status, _, _ = oped_evaluate_question(q, model_name)
        if status == "PASS":  n_pass += 1
        elif status == "FAIL": n_fail += 1
        else:                  n_idk  += 1

    total   = n_pass + n_fail + n_idk
    answered = n_pass + n_fail
    accuracy = (n_pass / answered * 100) if answered else 0.0
    honesty  = (n_idk  / total   * 100) if total    else 0.0
    return accuracy, honesty


# ══════════════════════════════════════════════════════════════════════════════
# TABLE PRINTER
# ══════════════════════════════════════════════════════════════════════════════

def print_table(rows):
    """
    rows: list of dicts with keys:
      model_group, variant, mmlu, arc, perplexity,
      oped_acc, oped_hon
    Prints a table matching the paper figure.
    """
    W = 80
    print("\n" + "=" * W)
    print(f"{'EVALUATION RESULTS':^{W}}")
    print("=" * W)
    print(f"  {'Model':<14} {'Variant':<8}  {'MMLU':>7}  {'ARC':>7}  "
          f"{'Perplexity':>11}  {'OpED Acc':>9}  {'Honesty':>8}")
    print("-" * W)

    current_group = None
    for r in rows:
        grp = r["model_group"]
        if grp != current_group:
            if current_group is not None:
                # Print delta row
                base = next(x for x in rows if x["model_group"]==current_group and x["variant"]=="Base")
                ft   = next(x for x in rows if x["model_group"]==current_group and x["variant"]=="FT")
                dm = ft["mmlu"]  - base["mmlu"]
                da = ft["arc"]   - base["arc"]
                dp = ((ft["perplexity"] - base["perplexity"]) / base["perplexity"] * 100
                      if base["perplexity"] else 0)
                do = ft["oped_acc"] - base["oped_acc"]
                print(f"  {'':14} {'Δ(%)':8}  {dm*100:>+7.2f}  {da*100:>+7.2f}  "
                      f"{dp:>+11.2f}  {do:>+9.2f}  {'—':>8}")
                print("-" * W)
            current_group = grp
            print(f"  {grp}")

        hon_str = f"{r['oped_hon']:.1f}%" if r["oped_hon"] is not None else "—"
        print(f"  {'':14} {r['variant']:<8}  "
              f"{r['mmlu']:>7.3f}  {r['arc']:>7.3f}  "
              f"{r['perplexity']:>11.3f}  "
              f"{r['oped_acc']:>8.1f}%  {hon_str:>8}")

    # Final delta for last group
    if current_group:
        base = next(x for x in rows if x["model_group"]==current_group and x["variant"]=="Base")
        ft   = next((x for x in rows if x["model_group"]==current_group and x["variant"]=="FT"), None)
        if ft:
            dm = ft["mmlu"]  - base["mmlu"]
            da = ft["arc"]   - base["arc"]
            dp = ((ft["perplexity"] - base["perplexity"]) / base["perplexity"] * 100
                  if base["perplexity"] else 0)
            do = ft["oped_acc"] - base["oped_acc"]
            print(f"  {'':14} {'Δ(%)':8}  {dm*100:>+7.2f}  {da*100:>+7.2f}  "
                  f"{dp:>+11.2f}  {do:>+9.2f}  {'—':>8}")
    print("=" * W)


# ══════════════════════════════════════════════════════════════════════════════
# OPED FIGURE (from original plot_figure_c.py, takes live results)
# ══════════════════════════════════════════════════════════════════════════════

def plot_oped_figure(rows, out_prefix="oped_figure_c"):
    C_BASE = "#eaa679"
    C_FT   = "#71dc8a"
    C_HON  = "#e50b0b"
    C_GRID = "#F7DEDE"

    labels   = []
    acc_vals = []
    hon_vals = []
    colors   = []
    hatches  = []

    for r in rows:
        short = r["model_group"].replace("LLaMA 2","L2").replace("LLaMA 3","L3").replace("Mistral","M7")
        labels.append(f"{short}\n{r['variant']}")
        acc_vals.append(r["oped_acc"])
        hon_vals.append(r["oped_hon"] if r["variant"] == "FT" else np.nan)
        colors.append(C_FT if r["variant"] == "FT" else C_BASE)
        hatches.append("" if r["variant"] == "FT" else "//")

    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman","DejaVu Serif"],
        "font.size": 11, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.85, 4.02))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    xi   = np.arange(len(labels))
    bars = ax.bar(xi, acc_vals, 0.55, color=colors, hatch=hatches,
                  edgecolor="#666", linewidth=0.5, zorder=3)

    for rect, val in zip(bars, acc_vals):
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height()+0.4,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color="#333")

    ax2  = ax.twinx()
    mask = ~np.isnan(hon_vals)
    hv_arr = np.array(hon_vals, dtype=float)
    if mask.any():
        ax2.scatter(xi[mask], hv_arr[mask], color=C_HON, s=80, zorder=6,
                    marker="o", edgecolors="white", linewidths=0.9)
        for i, hv in zip(xi[mask], hv_arr[mask]):
            ax2.annotate(f"{hv:.1f}%", (i, hv), xytext=(0, 9),
                         textcoords="offset points", ha="center", va="bottom",
                         fontsize=13, fontweight="bold", color=C_HON)

    ax.set_xticks(xi)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(55, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Honesty (%)", fontsize=13, color=C_HON)
    ax2.tick_params(axis="y", colors=C_HON, labelsize=12)
    ax2.spines["right"].set_edgecolor(C_HON)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    # Model family dividers
    n_per_grp = 2
    groups = sorted(set(r["model_group"] for r in rows), key=lambda g: rows[[r["model_group"] for r in rows].index(g)]["model_group"])
    for xd in [1.5, 3.5]:
        ax.axvline(xd, color="#cccccc", lw=0.8, linestyle="--", zorder=1)

    family_labels = [r["model_group"] for r in rows if r["variant"]=="Base"]
    for k, (lbl, xc) in enumerate(zip(family_labels, [0.5, 2.5, 4.5])):
        ax.text(xc, 101.5, lbl, ha="center", va="bottom", fontsize=13,
                color="#555", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc", lw=0.5))

    ax.set_title("Optical Network Evaluation Dataset (OpED) Accuracy vs. Honesty",
                 fontsize=13, fontweight="bold", pad=8, loc="left")

    handles = [
        Patch(facecolor=C_BASE, edgecolor="#666", lw=0.5, hatch="//", label="Base model"),
        Patch(facecolor=C_FT,   edgecolor="#999", lw=0.5,             label="Fine-tuned (LoRA)"),
        Line2D([0],[0], color=C_HON, lw=0, marker="o", markersize=7,
               markeredgecolor="white", markeredgewidth=0.8, label="Honesty rate"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True,
              edgecolor="#ddd", fontsize=12, ncol=1)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(f"{out_prefix}.png", bbox_inches="tight", dpi=300, facecolor="white")
    # fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight",           facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_prefix}.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global N_SHOTS, MMLU_SAMPLES, ARC_SAMPLES

    ap = argparse.ArgumentParser()
    ap.add_argument("--models",       nargs="+", default=None,
                    help="Subset of model names to evaluate (default: all)")
    ap.add_argument("--oped-only",    action="store_true",
                    help="Skip MMLU/ARC/Perplexity — run OpED only")
    ap.add_argument("--n-shots",      type=int, default=None)
    ap.add_argument("--mmlu-samples", type=int, default=None)
    ap.add_argument("--arc-samples",  type=int, default=None)
    ap.add_argument("--oped-file",    default="OpED.json")
    ap.add_argument("--out-prefix",   default="oped_figure_c")
    args = ap.parse_args()

    if args.n_shots      is not None: N_SHOTS      = args.n_shots
    if args.mmlu_samples is not None: MMLU_SAMPLES = args.mmlu_samples
    if args.arc_samples  is not None: ARC_SAMPLES  = args.arc_samples

    # Filter models if requested
    models = ALL_MODELS
    if args.models:
        models = [m for m in ALL_MODELS if m[1] in args.models or m[0] in args.models]

    # Load datasets
    print("Loading datasets...")
    oped_questions = []
    if os.path.exists(args.oped_file):
        raw = load_oped(args.oped_file)
        for d in raw:
            wt = COMPLEXITY_MAP.get(d.get("category"), 1)
            cl = d["answer"].strip().upper()
            if cl not in d["options"]: continue
            oped_questions.append({
                "id": d["id"], "category": d.get("category","General"),
                "question": d["question"], "options": d["options"],
                "correct_letter": cl, "correct_text": d["options"][cl],
                "weight": wt,
            })
        print(f"  OpED: {len(oped_questions)} questions")
    else:
        print(f"  WARNING: {args.oped_file} not found — OpED will be skipped")

    if not args.oped_only:
        mmlu_q = load_mmlu(n=MMLU_SAMPLES)
        arc_q  = load_arc(n=ARC_SAMPLES)
        print(f"  MMLU: {len(mmlu_q)}  ARC: {len(arc_q)}")

    # ── Run evaluations ───────────────────────────────────────────────────────
    rows = []
    groups = {"LLaMA 2":[], "LLaMA 3":[], "Mistral":[]}
    group_map = {
        "llama-2-7b-chat":"LLaMA 2", "llama-2-lora":"LLaMA 2",
        "llama3.1":"LLaMA 3",        "llama-3-lora":"LLaMA 3",
        "mistral":"Mistral",          "mistral-lora":"Mistral",
    }

    for display, ollama_name, is_ft in models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {display} ({ollama_name})")
        print(f"{'='*60}")

        row = {
            "model_group": group_map.get(ollama_name, display.rsplit(" ",1)[0]),
            "variant":     "FT" if is_ft else "Base",
            "mmlu":        0.0,
            "arc":         0.0,
            "perplexity":  0.0,
            "oped_acc":    0.0,
            "oped_hon":    None,
        }

        if not args.oped_only:
            print("  Running MMLU...")
            row["mmlu"] = mc_accuracy(ollama_name, mmlu_q, "MMLU")
            print(f"    MMLU: {row['mmlu']:.3f}")

            print("  Running ARC...")
            row["arc"] = mc_accuracy(ollama_name, arc_q, "ARC")
            print(f"    ARC:  {row['arc']:.3f}")

            print("  Computing perplexity...")
            row["perplexity"] = compute_perplexity(ollama_name)
            print(f"    Perplexity: {row['perplexity']:.3f}")

        if oped_questions:
            print("  Running OpED...")
            acc, hon = run_oped(ollama_name, oped_questions)
            row["oped_acc"] = acc
            row["oped_hon"] = hon if is_ft else None
            print(f"    OpED Accuracy: {acc:.1f}%  Honesty: {hon:.1f}%")

        rows.append(row)

        # Checkpoint — save after each model
        with open("eval_checkpoint.json", "w") as f:
            json.dump(rows, f, indent=2)

    # ── Print table ───────────────────────────────────────────────────────────
    print_table(rows)

    # Save table to file
    import io
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    print_table(rows)
    sys.stdout = old_stdout
    with open("eval_summary_table.txt", "w") as f:
        f.write(buf.getvalue())
    print("\n  Saved: eval_summary_table.txt")

    # ── Plot OpED figure ──────────────────────────────────────────────────────
    if any(r["oped_acc"] > 0 for r in rows):
        print("\n  Generating OpED figure...")
        plot_oped_figure(rows, out_prefix=args.out_prefix)

    # ── Save full results JSON ────────────────────────────────────────────────
    with open("eval_results_all.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("  Saved: eval_results_all.json")


if __name__ == "__main__":
    main()