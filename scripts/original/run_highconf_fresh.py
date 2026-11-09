"""Fresh generation for the main evaluation: greedy decoding with per-step log-probabilities (sealed).

Usage:
  python run_highconf_fresh.py --model mistral7 --pilot 200
  python run_highconf_fresh.py --model mistral7            # full run
  python run_highconf_fresh.py --model qwen8 --pilot 200
  ...

Outputs (all under OUT_DIR):
  prompts/<model>_<ds>.jsonl        # prompts as encoded: text + ids + sha (frozen)
  gen/<model>_<ds>.jsonl            # per line: id / question / gold text / full generation / generated ids / empty-retry flag
  logprobs_sealed/<model>_<ds>.jsonl# per line: id / per-step log-probabilities / pbar placeholder / generated ids (sealed)
  PROMPT_FREEZE.md                  # first prompt per model + sha + row counts
  progress/<model>.done_ids.txt     # resume support

Constraints: write only to OUT_DIR. The default prompt is the bare question; --fewshot and --base add the instruction and examples;
sealed log-probability files are not read by the first-stage analyses (separate file names; no reader is provided here).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

OUT_DIR = Path(os.environ.get("HIGHCONF_OUT", "/home/test/hallu_paper/current_survey"))
BENCH = Path("/home/test/hallu_paper/relation_specificity/benchmark_inputs")
POPQA_KEEP = OUT_DIR / "popqa_keep_idx.json"
POPQA_FROZEN_SHA256 = "f859297d66ad3825df2db5ca9094b91e10f65b9c18a415fa9f39ee6fbcce34ed"

MODELS = {
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen8": "Qwen/Qwen3-8B",
    "llama31": "meta-llama/Llama-3.1-8B-Instruct",
    "nemo12": "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen14": "Qwen/Qwen3-14B",
    # base models (no chat template; completion-style few-shot prompt)
    "qwen3base": "Qwen/Qwen3-8B-Base",
    "mistral7base": "mistralai/Mistral-7B-v0.3",
    "llama31base": "meta-llama/Llama-3.1-8B",  # verified to be the base model: no chat_template, eos=<|end_of_text|>
    "nemo12base": "mistralai/Mistral-Nemo-Base-2407",
    "qwen14base": "Qwen/Qwen3-14B-Base",
    # prompt-control arms (2026-09-20): the same instruct weights with the base completion-style prompt,
    # separating aligned weights from prompt format. Run with --base --base_neg; outputs use separate file names.
    "mistral7cmp": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen8cmp": "Qwen/Qwen3-8B",
    # 2026-11-06: prompt-control arms for the other three families (instruct weights + base completion prompt), same protocol as mistral7cmp/qwen8cmp
    "llama31cmp": "meta-llama/Llama-3.1-8B-Instruct",
    "nemo12cmp": "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen14cmp": "Qwen/Qwen3-14B",
    # SFT/DPO pipeline (2026-09-21): the official Zephyr checkpoints.
    # The base stage runs with --base --base_neg (same frozen protocol as the base arm: BASE_PILOT_DECISION=neg);
    # the sft/dpo stages run with --fewshot (as the instruct arm). sft-beta and zephyr-beta share the same chat_template.
    "ladderBase": "mistralai/Mistral-7B-v0.1",
    "ladderSft": "HuggingFaceH4/mistral-7b-sft-beta",
    "ladderDpo": "HuggingFaceH4/zephyr-7b-beta",
    # Tulu 3 pipeline (2026-09-21): second pipeline, Llama-3.1 architecture. The base stage reuses llama31base
    # (the same Meta-Llama-3.1-8B base checkpoint as in the main experiment). SFT/DPO use the short variant + 32 tokens.
    "tulu3Sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "tulu3Dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    # BDM-DPO training outputs (merged LoRA weights, from 2026-09-22)
    "bdmStd": "/home/test/hallu_paper/current_survey/runs/run_standard_dpo/merged",
    "bdmBdm": "/home/test/hallu_paper/current_survey/runs/run_bdm_dpo/merged",
    # RLVR final checkpoint (2026-09-21): Tulu-3-8B = SFT base + RL with verifiable rewards (RLVR).
    # Compared with DPO (subjective preferences) to test whether RL with rule-based verification also induces confident hallucinations.
    "tulu3Rlvr": "allenai/Llama-3.1-Tulu-3-8B",
    # princeton-nlp SimPO pipeline (2026-09-21): Llama-3-8B (not 3.1), four released stages.
    # base uses --base --base_neg (completion few-shot + negative examples); SFT/DPO/SimPO use --prompt_variant short + 32 tokens.
}

# ---- Dynamically register merged models under runs/models/* (multi-epoch experiments, 2026-09-22) ----
# Needed because --model choices are fixed at startup, while multi-epoch checkpoints are
# merged into runs/models/<name> only after training; without a scan the run exits with "invalid choice".
_DYNAMIC_MODELS = Path("/home/test/hallu_paper/current_survey/runs/models")
if _DYNAMIC_MODELS.is_dir():
    for _d in sorted(_DYNAMIC_MODELS.iterdir()):
        if _d.is_dir() and (_d / "config.json").exists():
            MODELS.setdefault(_d.name, str(_d))
# ---- Dynamically register merged models under runs/<run>/merged (SFT pipeline, 2026-09-23) ----
# Needed because bdm_dpo_train.py --merge in chain_pure_sft.sh writes results to
# runs/tulu_dpo_std/merged (not under runs/models/); scanning only runs/models would
# exit with `invalid choice: 'tulu_dpo_std'` and stop the pipeline at generation.
for _m in sorted(Path("/home/test/hallu_paper/current_survey/runs").glob("*/merged")):
    if (_m / "config.json").exists():
        MODELS.setdefault(_m.parent.name, str(_m))

# Qwen3 instruct models: thinking mode must be disabled explicitly (pre-registered: False); otherwise the template adds a thinking block
QWEN3_THINKING_MODELS = {"qwen8", "qwen14"}

MAX_NEW_TOKENS = 10
RETRY_EMPTY = 3


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_rows(dataset: str):
    """Return [(key, question, gold_list, pop_dict)]; pop is only available for PopQA."""
    if dataset == "triviaqa":
        out = []
        # rows are identical across model files; take the question text from mistral; each model encodes it itself
        d = json.loads((BENCH / "mistral-triviaqa.json").read_text(encoding="utf-8"))
        for r in d["records"]:
            out.append((f"triviaqa::{r['row_id']}", r["question"], list(r["gold_answers"]), {}))
        return out
    # popqa: read the frozen pool file first (offline, reproducible, sha256-checked); fall back to the full HF release only if missing
    frozen = OUT_DIR / "popqa_frozen_12509.jsonl"
    if frozen.exists():
        raw = frozen.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        assert sha == POPQA_FROZEN_SHA256, f"PopQA frozen pool sha256 mismatch: {sha} != {POPQA_FROZEN_SHA256}"
        out = []
        for line in raw.decode("utf-8").splitlines():
            r = json.loads(line)
            out.append((r["key"], r["question"], list(r["gold"]), r["pop"]))
        return out
    from datasets import load_dataset
    keep = set(json.loads(POPQA_KEEP.read_text(encoding="utf-8"))["keep_idx"])
    ds = load_dataset("akariasai/PopQA", split="test")
    out = []
    for i, r in enumerate(ds):
        if i not in keep:
            continue
        pop = {"s_pop": r.get("s_pop"), "o_pop": r.get("o_pop"),
               "subj": r.get("subj"), "prop": r.get("prop"), "obj": r.get("obj")}
        pa = r["possible_answers"]
        if isinstance(pa, str):
            import ast as _ast
            try:
                pa = _ast.literal_eval(pa)
            except Exception:
                pa = [pa]
        out.append((f"popqa::{r['id']}", r["question"], list(pa), pop))
    return out


SYSTEM_INSTRUCTION = ("Return only the requested factual answer entity or short value. "
                      "Use as few tokens as possible. NEVER output more than 8 words. "
                      "NEVER output a full sentence. A single entity or noun phrase only. "
                      "Do not write a sentence, preamble, explanation, reasoning, or anything that is not the direct answer.")

# Few-shot examples (CHOKE/PopQA style): short-entity answers in one format, teaching length, not knowledge.
# General-knowledge questions from several domains, chosen to avoid TriviaQA/PopQA questions (no leakage).
# 8 positive examples: short general-knowledge entities, disjoint from TriviaQA/PopQA, answers of 1-3 tokens.
FEWSHOT_POS_EXAMPLES = [
    ("What is the longest river in Africa?", "Nile"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("What is the chemical symbol for water?", "H2O"),
    ("Who was the 42nd President of the United States?", "Bill Clinton"),
    ("In which city was Marie Curie born?", "Warsaw"),
    ("What is the atomic number of gold?", "79"),
    ("Who won the Nobel Prize in Physics in 1921?", "Albert Einstein"),
    ("What is the largest planet in the Solar System?", "Jupiter"),
]


# 4 negative examples: 4 of the 8 questions above with verbose full-sentence answers, explicitly forbidden.
# Note: the negative answers are deliberately bad; grading only accepts the positive format.
NEGATIVE_EXAMPLES = [
    ("What is the longest river in Africa?",
     "The longest river in Africa is the Nile, which flows through many countries."),
    ("Who wrote Hamlet?",
     "Hamlet was written by Shakespeare, the famous English playwright."),
    ("What is the atomic number of gold?",
     "The atomic number of gold is 79, which is a chemical element."),
    ("Who was the 42nd President of the United States?",
     "Bill Clinton was the 42nd President of the United States."),
]

NEGATIVE_REMEMBER = ("Remember: NEVER answer with a full sentence. FORBIDDEN examples: "
    + " ".join(["Bad: \"" + na + "\"." for _nq, na in NEGATIVE_EXAMPLES])
    + " Output the entity only.")

FEWSHOT_EXAMPLES = FEWSHOT_POS_EXAMPLES


def render_prompt(tok, question: str, template_kwargs: dict | None = None, fewshot: bool = False) -> str:
    msgs = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    if fewshot:
        for q, a in FEWSHOT_POS_EXAMPLES:
            msgs.append({"role": "user", "content": q})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": NEGATIVE_REMEMBER + " Question: " + question})
    else:
        msgs.append({"role": "user", "content": question})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   **(template_kwargs or {}))


SYS_SHORT = ("Answer with only the requested factual entity or short value. "
             "No sentences. No explanations. No prefixes such as 'Answer:'.")


def render_prompt_short(tok, question: str, template_kwargs: dict | None = None) -> str:
    """Short-instruction variant (pipeline experiments): the system message only constrains the format, the user message gives the question, no few-shot.
    Identical for the SFT and DPO stages, so the SFT→DPO comparison is clean."""
    msgs = [{"role": "system", "content": SYS_SHORT},
            {"role": "user", "content": "Question: " + question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   **(template_kwargs or {}))


def render_prompt_base(question: str, with_neg: bool = False) -> str:
    """For base models: completion-style few-shot prompt with the same 8 examples, ending with `Q: …\\nA:`.
    with_neg=True also adds the instruction text and the FORBIDDEN negative examples (see PREREG_STAGE1_BASE §2)."""
    parts = []
    if with_neg:
        parts.append(SYSTEM_INSTRUCTION)
    parts += [f"Q: {q}\nA: {a}" for q, a in FEWSHOT_POS_EXAMPLES]
    if with_neg:
        parts.append(NEGATIVE_REMEMBER)
    parts.append(f"Q: {question}\nA:")
    return "\n\n".join(parts)


def setup_pad(tok) -> bool:
    """Choose a pad token different from eos; return whether the embedding needs resizing.

    Rules (required for left padding):
      1) act only if pad is None or pad == eos;
      2) prefer a reserved token from the vocabulary (e.g., <|finetune_right_pad_id|> for llama31, <|vision_pad|> for qwen3base),
         to avoid changing the embedding size (adding "[PAD]" to llama31 triggered an indexSelectLargeIndex error after resizing);
      3) otherwise add "[PAD]" and resize (grow only; ensured by the caller).
    """
    need_resize = False
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        vocab = tok.get_vocab()  # convert_tokens_to_ids returns the unk id for missing tokens, so check membership
        for cand in ("<|finetune_right_pad_id|>", "<|vision_pad|>", "<|fim_pad|>", "<pad>", "[PAD]"):
            if cand in vocab and vocab[cand] != tok.eos_token_id:
                tok.pad_token = cand
                break
        else:
            if tok.unk_token is not None and tok.unk_token_id != tok.eos_token_id:
                tok.pad_token = tok.unk_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})
                need_resize = True
    assert tok.pad_token_id != tok.eos_token_id, "pad and eos share an id: left padding would act as end of sequence"
    tok.padding_side = "left"
    return need_resize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--dataset", default="all", choices=["all", "triviaqa", "popqa"])
    ap.add_argument("--pilot", type=int, default=0, help="run only the first N items per dataset")
    ap.add_argument("--fewshot", action="store_true", default=False, help="enable the 8-shot short-entity examples")
    ap.add_argument("--qwen_thinking", action="store_true", default=False)
    ap.add_argument("--base", action="store_true", default=False,
                    help="base-model mode: completion-style few-shot prompt, no chat template, cut at newline")
    ap.add_argument("--base_neg", action="store_true", default=False,
                    help="in base mode, also add the instruction and the FORBIDDEN negative examples")
    ap.add_argument("--batch", type=int, default=1, help="batch size (default 1; 8 recommended for full runs)")
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS,
                    help="maximum new tokens (default 10; 32 for the pipeline experiments so that sentence answers are not cut before the entity)")
    ap.add_argument("--prompt_variant", default="fs", choices=["fs", "short"],
                    help="fs = few-shot protocol (default); short = short-instruction variant (pipeline experiments)")
    args = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = args.model
    mp = MODELS[model_name]
    template_kwargs: dict = {}
    if model_name in QWEN3_THINKING_MODELS:
        # the Qwen3 template argument goes to apply_chat_template, not generate; pre-registered: False
        pass  # tokenizer not loaded yet; the assertion comes after loading
    tok = AutoTokenizer.from_pretrained(mp, local_files_only=True, trust_remote_code=True)
    need_resize = setup_pad(tok)

    model = AutoModelForCausalLM.from_pretrained(
        mp, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto")
    if need_resize:
        # grow only: resize to max(len(tok), current rows); shrinking would put special tokens out of range
        cur = model.get_input_embeddings().weight.shape[0]
        model.resize_token_embeddings(max(len(tok), cur))
    assert tok.pad_token_id < model.get_input_embeddings().weight.shape[0], "pad id exceeds the embedding size"
    model.eval()

    # Qwen3 models: thinking mode disabled explicitly (pre-registered: False).
    # Detected by whether the chat_template supports enable_thinking, not by a hard-coded name list:
    # checkpoints produced by training (qwen_dpo_std / qwen_dpo_bdm, etc.) are registered dynamically under new names
    # that are not in QWEN3_THINKING_MODELS; a name list would miss them, and the Qwen3 template enables thinking by default,
    # which would break the protocol and, with max_new_tokens=10, produce only a thinking fragment (found 2026-09-23).
    # Checking the template covers all Qwen3-derived checkpoints; base models have no chat_template and are unaffected.
    if model_name in QWEN3_THINKING_MODELS or "enable_thinking" in (tok.chat_template or ""):
        template_kwargs["enable_thinking"] = bool(args.qwen_thinking)  # pre-registered: False
        assert "enable_thinking" in (tok.chat_template or ""), "qwen chat_template does not support enable_thinking"

    # Stop tokens: besides the tokenizer eos, also stop at <|eot_id|> (Llama-3 chat) if it is in the vocabulary
    # (the princeton-nlp checkpoints keep the base eos=<|end_of_text|> but end turns with <|eot_id|>, so generation would run to the limit; 2026-09-21)
    eos_ids = [tok.eos_token_id]
    for cand in ("<|eot_id|>", "<|end_of_turn|>", "<|im_end|>"):
        cid = tok.convert_tokens_to_ids(cand)
        if isinstance(cid, int) and cid >= 0 and cid not in eos_ids and cand in tok.get_vocab():
            eos_ids.append(cid)
    eos_ids = eos_ids if len(eos_ids) > 1 else tok.eos_token_id
    print(f"  [eos] {eos_ids}", flush=True)

    datasets = ["triviaqa", "popqa"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        rows = load_rows(ds)
        if args.pilot:
            rows = rows[:args.pilot]
        tag = "fs" if (args.fewshot or args.base) else "zs"
        if os.environ.get("HIGHCONF_TAG"):
            tag = os.environ["HIGHCONF_TAG"]
        # Log the protocol explicitly (2026-09-24): HIGHCONF_TAG only sets the output directory name;
        # whether the 8 examples are added is set by --fewshot (base models use the completion few-shot of --base).
        # The unified protocol is the fs template family with zero-shot prompting (decided 2026-09-24):
        # the directory tag stays fs (eval_bdm/gate_a read gen_fs/), but fs refers to the template family.
        # The actual protocol is written to the log so that the directory name and the protocol cannot diverge.
        _proto = ("base-completion-fewshot" if args.base
                  else "fs-template-8shot" if args.fewshot
                  else "fs-template-zeroshot")
        print(f"  [protocol] model={model_name} dir_tag={tag} actual={_proto}", flush=True)
        pdir = OUT_DIR / ("prompts_"+tag); gdir = OUT_DIR / ("gen_"+tag); ldir = OUT_DIR / ("logprobs_"+tag)
        for d in (pdir, gdir, ldir, OUT_DIR / ("progress_"+tag)):
            d.mkdir(parents=True, exist_ok=True)
        done_path = OUT_DIR / ("progress_"+tag) / f"{model_name}_{ds}.done_ids.txt"
        done = set(done_path.read_text(encoding="utf-8").split()) if done_path.exists() else set()
        pf = (pdir / f"{model_name}_{ds}.jsonl").open("a", encoding="utf-8")
        gf = (gdir / f"{model_name}_{ds}.jsonl").open("a", encoding="utf-8")
        lf = (ldir / f"{model_name}_{ds}.jsonl").open("a", encoding="utf-8")
        df = done_path.open("a", encoding="utf-8")
        t0 = time.time(); n_new = 0
        B = max(1, args.batch)
        pending = []
        for key, question, gold, pop in rows:
            rid = f"{model_name}|{key}"
            if rid in done:
                continue
            pending.append((rid, key, question, gold, pop))
        def gen_once(batch):
            """Run greedy generation for one batch and return the results."""
            if args.base:
                texts = [render_prompt_base(q, args.base_neg) for _, _, q, _, _ in batch]
            elif args.prompt_variant == "short":
                texts = [render_prompt_short(tok, q, template_kwargs) for _, _, q, _, _ in batch]
            else:
                texts = [render_prompt(tok, q, template_kwargs, fewshot=args.fewshot)
                         for _, _, q, _, _ in batch]
            enc = tok(texts, padding=True, truncation=False, return_tensors="pt", padding_side="left")
            input_ids = enc["input_ids"].to(model.device)
            attn = enc["attention_mask"].to(model.device)
            in_width = input_ids.shape[1]  # inputs have equal length after left padding: cut the generation at the padded width, not at each row's length
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids, attention_mask=attn, do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True, output_scores=True,
                    pad_token_id=tok.pad_token_id, eos_token_id=eos_ids)
            seqs = out.sequences.tolist()
            # scores[t]: [B, V] at generation step t
            all_lps = []
            for t, sc in enumerate(out.scores):
                all_lps.append(torch.log_softmax(sc.float(), dim=-1))
            n_steps = len(out.scores)
            res = []
            for b, (rid, key, question, gold, pop) in enumerate(batch):
                raw = seqs[b][in_width:]  # seqs = [pad + prompt + gen]; the cut point must be the padded width
                if tok.eos_token_id in raw:
                    gen_ids = raw[:raw.index(tok.eos_token_id)]
                else:
                    gen_ids = raw[:n_steps]
                if args.base:
                    # base models do not stop reliably: answers end with a newline, so cut at the first token containing a newline
                    for t, tid in enumerate(gen_ids):
                        if "\n" in tok.decode([tid]):
                            gen_ids = gen_ids[:t]
                            break
                lps = [round(float(all_lps[t][b][gen_ids[t]].item()), 6)
                       for t in range(min(len(gen_ids), n_steps))]
                gen_ids = gen_ids[:len(lps)]
                res.append((rid, key, question, gold, pop, gen_ids, lps,
                            tok.decode(gen_ids, skip_special_tokens=True)))
            return res

        def flush_batch(batch):
            # batch: list[(rid,key,question,gold,pop)]
            # retry only the row with an empty answer (v2: base models give more empty answers; rerunning the whole batch would multiply the cost)
            nonlocal n_new
            got = {}
            cur = batch
            for _ in range(RETRY_EMPTY + 1):
                res = gen_once(cur)
                for r in res:
                    got[r[0]] = r
                cur = [r[:5] for r in res if not r[7].strip()]
                if not cur:
                    break
            results = [got[r[0]] for r in batch]
            for rid, key, question, gold, pop, gen_ids, lps, text in results:
                prompt_text = (render_prompt_base(question, args.base_neg) if args.base else
                               render_prompt(tok, question, template_kwargs, fewshot=args.fewshot))
                ids = tok.encode(prompt_text, add_special_tokens=False)
                pf.write(json.dumps({"id": rid, "prompt_text": prompt_text,
                                     "prompt_token_ids": ids,
                                     "sha256": sha256_str(json.dumps(ids))}, ensure_ascii=False) + "\n")
                gf.write(json.dumps({"id": rid, "question": question, "gold": gold, "pop": pop,
                                     "gen_text": text, "gen_ids": gen_ids,
                                     "engine_fail": not text.strip()}, ensure_ascii=False) + "\n")
                lf.write(json.dumps({"id": rid, "step_logprobs": lps,
                                     "gen_ids": gen_ids, "sealed": True}, ensure_ascii=False) + "\n")
                df.write(rid + "\n")
                n_new += 1
            if n_new % 20 == 0:
                for f in (pf, gf, lf, df):
                    f.flush(); os.fsync(f.fileno())
        for i in range(0, len(pending), B):
            flush_batch(pending[i:i+B])
        for f in (pf, gf, lf, df):
            f.flush(); os.fsync(f.fileno()); f.close()
        dt = time.time() - t0
        print(f"DONE {model_name}/{ds}: new={n_new} {dt:.0f}s {n_new/max(dt,1e-6):.2f}/s", flush=True)


if __name__ == "__main__":
    main()
