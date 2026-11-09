"""BDM-DPO (bounded dynamic margin DPO) vs. standard DPO: LoRA training and training-curve logging.

Motivation: the standard DPO gradient does not decay with sample difficulty, so logits on unfamiliar long-tail samples keep growing.
BDM-DPO bounds the margin using the reference model's entropy on the prompt:

  Standard:  L = -log σ( β · Δ )                       Δ = log(π_θ(y_w)/π_ref(y_w)) - log(π_θ(y_l)/π_ref(y_l))
  BDM:       L = -log σ( β · min( Δ , M0 · exp(-γ · H(π_ref)/ln|V|) ) )

  β=0.1, M0=3.0, γ=2.0; H(π_ref) = mean per-token entropy of the reference model over the **prompt tokens**.
  Small H (familiar queries) ⇒ bound ≈ M0; large H (unfamiliar queries) ⇒ smaller bound, the gradient saturates.

Implementation notes:
  - log-probabilities and H(π_ref) of the frozen reference model are **precomputed and cached** once; only π_θ (LoRA) is on the GPU,
    which saves memory (two models take 28 GB plus activations, close to the 40 GB limit).
  - Hand-written training loop (TRL's DPOTrainer does not support a custom clamped loss).

Usage:
  python mitigation/bdm_dpo_train.py precompute --n 4000
  python mitigation/bdm_dpo_train.py train --mode standard --out_dir runs/run_standard_dpo
  python mitigation/bdm_dpo_train.py train --mode bdm      --out_dir runs/run_bdm_dpo
Outputs:
  mitigation/data/dpo_pairs_cache.pt          (precomputed cache)
  runs/<name>/adapter/ + merged/              (LoRA adapter and merged weights)
  runs/<name>/train_curve.json                (per-step loss/margin/M/H)
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

D = Path("/home/test/hallu_paper/current_survey")
sys.path.insert(0, str(D))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


rhf = _load("rhf", D / "run_highconf_fresh.py")
BASE = rhf.MODELS["mistral7"]          # Mistral-7B-Instruct-v0.3 (main model in the paper)
BASE_DEFAULT = "mistral7"           # default model; override with --base (MODELS key or local path)
UF = D.parent / "datasets/uf_ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet"
CACHE = D / "mitigation/data/dpo_pairs_cache.pt"
CACHE_DEFAULT = D / "mitigation/data/dpo_pairs_cache.pt"   # override with --cache to avoid overwriting the Mistral cache
MAXLEN = 2048          # measured lengths: p50=647, p95=2007 ⇒ ~5% truncated at 2048 (64% at 512, which fails the check)

def resolve_base(spec: str) -> str:
    """--base accepts a MODELS key or a local path."""
    if spec in rhf.MODELS:
        return rhf.MODELS[spec]
    q = Path(spec)
    if not q.exists():
        raise SystemExit(f"--base is neither a MODELS key nor an existing path: {spec}")
    return str(q)



# ---------------------------------------------------------------- data
def build_pairs(n: int, seed: int, data=None):
    # data=None uses the module constant UF (UltraFeedback, main setting); otherwise the given parquet file.
    # Added 2026-09-26 for the dataset-robustness experiment (HH-RLHF); default behavior unchanged.
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files=str(data or UF), split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    pairs = []
    for r in ds:
        p = r["prompt"]
        if isinstance(p, list):
            p = "\n".join(m.get("content", "") for m in p)
        ch, rj = r.get("chosen"), r.get("rejected")
        if isinstance(ch, list):
            ch = "\n".join(m.get("content", "") for m in ch)
        if isinstance(rj, list):
            rj = "\n".join(m.get("content", "") for m in rj)
        if not (p and ch and rj):
            continue
        pairs.append({"prompt": p.strip(), "chosen": ch.strip(), "rejected": rj.strip()})
    return pairs


def encode(tok, head: str, resp: str, maxlen=MAXLEN):
    """Return (ids, mask); mask=1 marks response positions (used for log-probabilities)."""
    hi = tok(head, add_special_tokens=False)["input_ids"]
    ri = tok(resp, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (hi + ri)[:maxlen]
    n_head = min(len(hi), maxlen)
    mask = [0] * n_head + [1] * max(0, len(ids) - n_head)
    return ids, mask


def collate(batch, pad_id):
    L = max(len(b["ids"]) for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(batch), L), dtype=torch.long)
    lm = torch.zeros((len(batch), L), dtype=torch.float)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"])
        am[i, :n] = 1
        lm[i, :n] = torch.tensor(b["mask"], dtype=torch.float)
    return ids, am, lm


# ---------------------------------------------------------------- precompute π_ref
@torch.no_grad()
def _think_kw(tok) -> dict:
    """Qwen3 models: thinking mode explicitly disabled (pre-registered as False).

    This must match evaluation: run_highconf_fresh.py passes enable_thinking=False when generating
    (the thinking block is prefilled empty and the model answers directly). Without it, the Qwen3 template **enables thinking by default**,
    and training prompts would differ from evaluation prompts by a prefilled empty think block
    (found on 2026-09-23).
    For models whose template has no such argument (e.g., Mistral) an empty dict is returned, so behavior is unchanged.
    """
    return {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}


def precompute(args):
    global BASE, CACHE
    BASE = resolve_base(getattr(args, "base", BASE_DEFAULT))
    CACHE = Path(getattr(args, "cache", str(CACHE_DEFAULT)))
    print(f"[base] {BASE}\n[cache] {CACHE}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _data = getattr(args, "data", None)
    pairs = build_pairs(args.n, args.seed, _data)
    print(f"Data: {len(pairs)} pairs ({_data or UF}, seed={args.seed})", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16, device_map="auto")
    net.eval()
    V = net.config.vocab_size

    recs = []
    t0 = time.time()
    for s in range(0, len(pairs), args.batch):
        chunk = pairs[s:s + args.batch]
        batch = []
        for pr in chunk:
            head = tok.apply_chat_template([{"role": "user", "content": pr["prompt"]}],
                                           tokenize=False, add_generation_prompt=True, **_think_kw(tok))
            iw, mw = encode(tok, head, pr["chosen"])
            il, ml = encode(tok, head, pr["rejected"])
            batch.append({"ids": iw, "mask": mw, "il": il, "ml": ml})
        base = len(recs)
        for tag, k, mk in (("w", "ids", "mask"), ("l", "il", "ml")):
            ids, am, lm = collate([{"ids": b[k], "mask": b[mk]} for b in batch], tok.pad_token_id)
            ids, am, lm = ids.to(net.device), am.to(net.device), lm.to(net.device)
            # Must be no_grad: net.eval() only disables dropout, not autograd. Parameters default to
            # requires_grad=True, which would build graphs and accumulate them across iterations.
            # Observed 2026-09-24: Qwen3-8B (vocabulary 151,936) ran out of memory (39.37 GiB) even with batch=1;
            # Mistral (vocabulary 32,768) has a much smaller graph. Only forward log-probabilities are needed here.
            with torch.no_grad():
                logits = net(input_ids=ids, attention_mask=am).logits.float()
            lp = F.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            m = lm[:, 1:]
            seq_lp = (tok_lp * m).sum(1).cpu()               # (B,) response log-probability per pair
            if tag == "w":
                # reference entropy over the prompt tokens (per row, at each prompt's own positions, avoiding right-padding leakage)
                pm = (1.0 - m) * am[:, 1:]
                pr = lp.exp()
                ent = -(pr * lp).sum(-1)                     # (B, L-1) entropy per position
                H = ((ent * pm).sum(1) / pm.sum(1).clamp(min=1)).cpu()
                for j in range(len(chunk)):                  # one record per pair, same indexing as in training
                    recs.append({"w_logp": seq_lp[j], "H_ref": H[j],
                                 "H_norm": H[j] / math.log(V)})
            else:
                assert len(recs) == base + len(chunk), f"record count mismatch (w): {len(recs)} vs {base + len(chunk)}"
                for j in range(len(chunk)):
                    recs[base + j]["l_logp"] = seq_lp[j]
        if s % (args.batch * 20) == 0:
            print(f"  {s + len(chunk)}/{len(pairs)}  ({time.time()-t0:.0f}s)", flush=True)

    assert len(recs) == len(pairs), f"record count mismatch: {len(recs)} vs {len(pairs)}"
    need = ("w_logp", "l_logp", "H_ref", "H_norm")
    for i, r in enumerate(recs):
        miss = [k for k in need if k not in r]
        assert not miss, f"record {i} is missing fields {miss}"
    torch.save({"recs": recs, "pairs": pairs, "base": BASE, "n": len(pairs)}, CACHE)
    hs = torch.stack([r["H_norm"] for r in recs])
    print(f"WROTE {CACHE}\n  H_norm: min={hs.min():.3f} p25={hs.quantile(.25):.3f} p50={hs.median():.3f} "
          f"p75={hs.quantile(.75):.3f} max={hs.max():.3f}")


# ---------------------------------------------------------------- training
def batch_logp(net, ids, am, lm):
    logits = net(input_ids=ids, attention_mask=am, use_cache=False).logits.float()
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * lm[:, 1:]).sum(1)


def train(args):
    global BASE, CACHE
    BASE = resolve_base(getattr(args, "base", BASE_DEFAULT))
    CACHE = Path(getattr(args, "cache", str(CACHE_DEFAULT)))
    print(f"[base] {BASE}\n[cache] {CACHE}", flush=True)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache = torch.load(CACHE, weights_only=False)
    recs, pairs = cache["recs"], cache["pairs"]
    print(f"Cache: {len(recs)} pairs, base={cache['base']}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    net = AutoModelForCausalLM.from_pretrained(BASE, local_files_only=True, trust_remote_code=True,
                                               torch_dtype=torch.bfloat16, device_map="auto")
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    net = get_peft_model(net, lcfg)
    if args.grad_ckpt:
        net.enable_input_require_grads()
        net.gradient_checkpointing_enable()
        net.config.use_cache = False
        print("  Gradient checkpointing enabled (saves memory, about 30% slower)", flush=True)
    net.print_trainable_parameters()
    net.train()

    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=0.0)
    V = net.config.vocab_size
    idx = list(range(len(recs)))
    random.Random(args.seed).shuffle(idx)
    curve = []
    step = 0
    t0 = time.time()
    micro = 0
    opt.zero_grad(set_to_none=True)

    save_epochs_set = set()
    if getattr(args, "save_epochs", ""):
        if args.save_epochs.strip() == "all":
            save_epochs_set = {"all"}
        else:
            save_epochs_set = {int(x.strip()) for x in args.save_epochs.split(",") if x.strip()}
    print(f"Training plan: {args.epochs} epochs; intermediate adapters saved at epochs: {save_epochs_set or 'final only'}", flush=True)
    # out must be defined before use: saving adapter_ep{N} inside the loop depends on it (fixed 2026-09-22)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)


    for epoch in range(args.epochs):
        for s in range(0, len(idx), args.batch):
            sel = idx[s:s + args.batch]
            batch = []
            for i in sel:
                pr = pairs[i]
                head = tok.apply_chat_template([{"role": "user", "content": pr["prompt"]}],
                                               tokenize=False, add_generation_prompt=True, **_think_kw(tok))
                iw, mw = encode(tok, head, pr["chosen"])
                il, ml = encode(tok, head, pr["rejected"])
                batch.append({"ids": iw, "mask": mw, "il": il, "ml": ml})
            ids, am, lm = collate([{"ids": b["ids"], "mask": b["mask"]} for b in batch], tok.pad_token_id)
            ids, am, lm = ids.to(net.device), am.to(net.device), lm.to(net.device)
            lpw = batch_logp(net, ids, am, lm)
            ids2, am2, lm2 = collate([{"ids": b["il"], "mask": b["ml"]} for b in batch], tok.pad_token_id)
            ids2, am2, lm2 = ids2.to(net.device), am2.to(net.device), lm2.to(net.device)
            lpl = batch_logp(net, ids2, am2, lm2)

            assert max(sel) < len(recs), f"index out of range: max={max(sel)} len={len(recs)}"
            ref_w = torch.stack([recs[i]["w_logp"] for i in sel]).to(net.device)
            ref_l = torch.stack([recs[i]["l_logp"] for i in sel]).to(net.device)
            delta = (lpw - ref_w) - (lpl - ref_l)          # Δ_{θ,ref}
            if args.mode not in ("bdm", "bdm_lit"):   # standard / ipo / cdpo / hinge: no clamp (the last three added on 2026-11-06)
                margin = delta
                M_used = None
            else:
                # bdm_lit (Eq. 6 / Algorithm 1 as typeset in the submission): H is the first-token reference entropy H1_norm (written to the cache by h1_cache.py)
                H = torch.stack([recs[i]["H1_norm" if args.mode == "bdm_lit" else "H_norm"] for i in sel]).to(net.device)
                # Bound M(H): three shapes for the ablation (added 2026-09-23)
                #   dynamic (paper): M0 * exp(-gamma*H)   nonlinear, positive, shrinks with prior entropy
                #   static       : M0                     independent of entropy (fixed hard clamp)
                #   linear       : M0 * max(0, 1-gamma*H)  linear decay, control with the same gamma
                if args.trunc == "static":
                    M = torch.full_like(H, args.M0)
                elif args.trunc == "linear":
                    M = args.M0 * torch.clamp(1.0 - args.gamma * H, min=0.0)
                else:
                    M = args.M0 * torch.exp(-args.gamma * H)
                M_used = M.mean().item()
                margin = torch.minimum(delta, M) if args.mode == "bdm" else delta
                capped = (delta > M) if args.mode == "bdm" else (args.beta * delta > M)
            if args.mode == "bdm_lit":  # Eq. 6 as typeset in the submission (added 2026-11-06): −log σ(min(βΔ, M_max)), β inside the min
                loss = -F.logsigmoid(torch.minimum(args.beta * delta, M)).mean()
            elif args.mode == "ipo":      # IPO (Azar et al. 2024): regress the sequence-level log-ratio difference to 1/(2β)
                loss = ((margin - 1.0 / (2.0 * args.beta)) ** 2).mean()
            elif args.mode == "cdpo":   # cDPO (Mitchell 2023): label smoothing, preference labels flipped with probability ε
                loss = (-(1.0 - args.ls_eps) * F.logsigmoid(args.beta * margin)
                        - args.ls_eps * F.logsigmoid(-args.beta * margin)).mean()
            elif args.mode == "hinge":  # SLiC-HF-style hinge (as TRL loss_type="hinge"): zero gradient once β·Δ ≥ 1
                loss = torch.relu(1.0 - args.beta * margin).mean()
            else:                       # standard / bdm: original loss, unchanged
                loss = -F.logsigmoid(args.beta * margin).mean()
            nll = None
            if getattr(args, "rpo_alpha", 0.0) > 0:   # RPO: anchor the chosen likelihood so that DPO does not push probability onto degenerate outputs
                ntok = lm[:, 1:].sum(1).clamp(min=1)
                nll = (-(lpw / ntok)).mean()
                loss = loss + args.rpo_alpha * nll
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    rec = {"step": step, "loss": float(loss), "margin_mean": float(delta.mean()),
                           "margin_min": float(delta.min()), "margin_max": float(delta.max()),
                           "acc": float((delta > 0).float().mean()),
                           "elapsed": round(time.time() - t0, 1),
                           "w_shift": float((lpw - ref_w).mean()), "l_shift": float((lpl - ref_l).mean())}
                    if nll is not None:
                        rec["nll"] = float(nll)
                    if M_used is not None:
                        rec["M_mean"] = M_used
                        rec["frac_capped"] = float(capped.float().mean())
                    curve.append(rec)
                    print(f"  step {step}: loss={rec['loss']:.4f} Δ={rec['margin_mean']:+.4f} "
                          f"acc={rec['acc']:.2f} w={rec['w_shift']:+.2f} l={rec['l_shift']:+.2f}"
                          + (f" nll={rec['nll']:.3f}" if "nll" in rec else "") + (f" M={M_used:.3f} capped={rec['frac_capped']:.2f}" if M_used else ""),
                          flush=True)

        cur_ep = epoch + 1
        print(f"=== Epoch {cur_ep}/{args.epochs} done (elapsed {time.time()-t0:.1f}s, step={step}) ===", flush=True)
        if save_epochs_set and (cur_ep in save_epochs_set or "all" in save_epochs_set):
            ep_adapter = out / f"adapter_ep{cur_ep}"
            net.save_pretrained(ep_adapter)
            print(f"  [Epoch {cur_ep}] saved LoRA adapter to {ep_adapter}", flush=True)

    net.save_pretrained(out / "adapter")
    (out / "train_curve.json").write_text(json.dumps(
        {"mode": args.mode, "beta": args.beta, "M0": args.M0, "gamma": args.gamma, "lr": args.lr,
         "n_pairs": len(recs), "epochs": args.epochs, "batch": args.batch, "accum": args.accum,
         "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "steps": step,
         "trunc": args.trunc, "rpo_alpha": getattr(args, "rpo_alpha", 0.0),
         "maxlen": MAXLEN, "grad_ckpt": bool(args.grad_ckpt), "curve": curve},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"adapter → {out/'adapter'}, curve → {out/'train_curve.json'}", flush=True)

    if args.merge:
        merged = net.merge_and_unload()
        merged.save_pretrained(out / "merged", safe_serialization=True)
        tok.save_pretrained(out / "merged")
        print(f"merged → {out/'merged'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("precompute")
    p1.add_argument("--n", type=int, default=4000)
    p1.add_argument("--batch", type=int, default=8)
    p1.add_argument("--data", default=None,
                    help="preference data parquet with prompt/chosen/rejected columns; defaults to UltraFeedback")
    p1.add_argument("--seed", type=int, default=20260922)
    p1.add_argument("--base", default=BASE_DEFAULT, help="MODELS key or local path")
    p1.add_argument("--cache", default=str(CACHE_DEFAULT), help="path of the π_ref cache (read/write)")
    p2 = sub.add_parser("train")
    p2.add_argument("--mode", choices=["standard", "bdm", "bdm_lit", "ipo", "cdpo", "hinge"], required=True)
    p2.add_argument("--ls_eps", type=float, default=0.1, help="label smoothing ε for cdpo (added 2026-11-06)")
    p2.add_argument("--rpo_alpha", type=float, default=0.0,
                    help="RPO (Pang et al. 2024): add α times the per-token NLL of the chosen answer; 0 (default) is the original loss (added 2026-11-07)")
    p2.add_argument("--out_dir", required=True)
    p2.add_argument("--beta", type=float, default=0.1)
    p2.add_argument("--M0", type=float, default=3.0)
    p2.add_argument("--gamma", type=float, default=2.0)
    p2.add_argument("--trunc", choices=["dynamic", "static", "linear"], default="dynamic",
                    help="shape of the bound: dynamic=M0*exp(-gamma*H) (paper); "
                         "static=constant M0 (independent of entropy); linear=M0*clamp(1-gamma*H, min=0)")
    p2.add_argument("--lr", type=float, default=5e-6)
    p2.add_argument("--batch", type=int, default=4)
    p2.add_argument("--accum", type=int, default=4)
    p2.add_argument("--epochs", type=int, default=1)
    p2.add_argument("--save_epochs", default="", help="epochs at which to save intermediate adapters, comma-separated, e.g. 1,2,3,5,8,10 or all")
    p2.add_argument("--lora_r", type=int, default=16)
    p2.add_argument("--lora_alpha", type=int, default=32)
    p2.add_argument("--log_every", type=int, default=10)
    p2.add_argument("--merge", action="store_true")
    p2.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (fallback for OOM)")
    p2.add_argument("--seed", type=int, default=20260922)
    p2.add_argument("--base", default=BASE_DEFAULT, help="MODELS key or local path")
    p2.add_argument("--cache", default=str(CACHE_DEFAULT), help="path of the π_ref cache (read/write)")
    a = ap.parse_args()
    if a.cmd == "precompute":
        precompute(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
