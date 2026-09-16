"""
================================================================================
compress_distill.py — Knowledge Distillation to DistillVHF-QWEN
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Trains a SMALLER "student" model to imitate the merged VHF-QWEN "teacher".
The student can be:
  (a) a fresh smaller Qwen model — e.g. Qwen2.5-1.5B-Instruct  (default)
  (b) the pruned VHF-QWEN from compress_prune.py, to recover its quality

The trained result is saved as _models/DistillVHF-QWEN/.

DISTILLATION LOSS
-----------------
For each training example (from the SFT dataset), we compute:

    L = alpha * CE_hard(student_logits, ground_truth_tokens)
      + (1 - alpha) * T^2 * KL( softmax(teacher/T) || softmax(student/T) )

with temperature T = 2.0 and alpha = 0.5 (Hinton et al. 2015 defaults).

Why the temperature?  Softening the distributions carries information about
which "wrong" answers the teacher considers plausible.  Learning that
structure gives the student more signal per token than hard labels alone.

Why train the student with LoRA + QLoRA?
----------------------------------------
Even Qwen2.5-1.5B needs ~6 GB VRAM for full fine-tune with adam.  QLoRA
brings it under 3 GB, leaving room for the teacher (~5 GB in 4-bit NF4) in
the same process.  Total VRAM peak: ~8 GB on RTX 4070.

DEPENDENCIES
------------
Uses only transformers + peft + bitsandbytes (already installed).
No trl needed because the custom KD loss is implemented inline.

USAGE
-----
    python compress_distill.py                          # default: 1.5B student
    python compress_distill.py --student-id Qwen/Qwen2.5-3B-Instruct
    python compress_distill.py --student _models/VHF-QWEN-pruned   # recover pruned
    python compress_distill.py --alpha 0.7 --temperature 3.0

OUTPUT
------
_models/DistillVHF-QWEN/    student model + LoRA adapters (call merge_adapter.py
                            or the built-in merge at the end)
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TDataset
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from core import AgentPaths, load_messages_jsonl

paths = AgentPaths.from_env()
W = paths.workspace
CACHE  = paths.cache_dir
MODELS = paths.models_root
VHF_MODELS = paths.domain_models_dir
os.environ.setdefault("HF_HOME", str(paths.hf_cache_dir))

TEACHER_DIR       = VHF_MODELS / f"{paths.domain}-QWEN"
DEFAULT_STUDENT   = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_OUT       = VHF_MODELS / f"Distill{paths.domain}-QWEN"

# Same combined Track 1 + Track 2 datasets as train_sft.py (see notebook § 12/§12.6)
# -- the student should see everything the teacher was fine-tuned on.
SFT_DATASETS = [
    CACHE / "vhf_sft_direct.jsonl",
    CACHE / "vhf_sft_cot.jsonl",
    CACHE / "vhf_sft_rag.jsonl",
    CACHE / "vhf_multihop.jsonl",
    CACHE / "vhf_conversations.jsonl",
    CACHE / "vhf_colreg_sft_direct.jsonl",
    CACHE / "vhf_colreg_sft_cot.jsonl",
    CACHE / "vhf_colreg_sft_rag.jsonl",
    CACHE / "vhf_colreg_multihop.jsonl",
]


def load_all() -> Dataset:
    parts = [Dataset.from_list(load_messages_jsonl(p)) for p in SFT_DATASETS if p.exists()]
    ds = concatenate_datasets(parts).shuffle(seed=13)
    print(f"Distillation train rows: {len(ds)}")
    return ds


# ── Tokenization to chat template ────────────────────────────────────────
def tokenize_row(row: dict, tok, max_len: int = 2048):
    """Turn messages -> input_ids + labels where labels are -100 on the system
    and user tokens, and equal to input_ids on assistant tokens (standard
    completion-only mask so the student learns to produce the assistant reply).
    """
    text = tok.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    prompt_msgs = [m for m in row["messages"] if m["role"] != "assistant"]
    prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

    full_ids   = tok(text,        add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
    prompt_ids = tok(prompt_text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]

    labels = [-100] * min(len(prompt_ids), len(full_ids)) + list(full_ids[len(prompt_ids):])
    labels = labels[:len(full_ids)]
    return {"input_ids": full_ids, "labels": labels}


class CollatorPad:
    """Right-pad input_ids and labels to the batch max length.
    Labels padded with -100 so padding doesn't contribute to loss.
    """
    def __init__(self, pad_id: int): self.pad_id = pad_id
    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids   = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        labs  = torch.full((len(batch), L), -100,        dtype=torch.long)
        attn  = torch.zeros((len(batch), L),             dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids [i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            labs[i, :n] = torch.tensor(b["labels"],    dtype=torch.long)
            attn[i, :n] = 1
        return {"input_ids": ids, "attention_mask": attn, "labels": labs}


# ── Distillation training loop ───────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    """Run logit-distillation training: frozen 4-bit teacher, LoRA student, KD + CE loss."""
    tok_t = AutoTokenizer.from_pretrained(str(TEACHER_DIR))
    tok_s = AutoTokenizer.from_pretrained(args.student_id if not Path(args.student_id).exists() else args.student_id)
    if tok_s.pad_token is None: tok_s.pad_token = tok_s.eos_token
    # We assume teacher and student share the same tokenizer vocabulary.
    # Both are Qwen2.5-based, so this holds. If you swap the student family
    # you MUST also cross-map token-ids or use logit-projection.
    assert tok_t.vocab_size == tok_s.vocab_size, \
        "Teacher and student tokenizers differ — logit KD requires matching vocab."

    # --- Teacher: 4-bit NF4, frozen, no grad ---
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print(f"Loading teacher from {TEACHER_DIR} in 4-bit NF4 (frozen)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        str(TEACHER_DIR), quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- Student: bf16 (small enough) + QLoRA ---
    print(f"Loading student {args.student_id} in bf16 with LoRA...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    student = prepare_model_for_kbit_training(student, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    student = get_peft_model(student, lora)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student trainable params: {trainable:,}")

    # --- Data ---
    ds = load_all()
    ds = ds.map(lambda r: tokenize_row(r, tok_s, args.max_len), remove_columns=ds.column_names)
    ds.set_format(type=None)
    loader = DataLoader(
        cast(TDataset, ds),
        batch_size=args.batch_size, shuffle=True,
        collate_fn=CollatorPad(tok_s.pad_token_id),
        num_workers=0,
    )

    # --- Optim ---
    optim = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    total_steps = (len(loader) // args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(optim, int(0.03 * total_steps), total_steps)

    T, alpha = args.temperature, args.alpha
    student.train()
    print(f"Starting KD training: T={T}, alpha={alpha}, steps={total_steps}")
    step = 0
    running = {"kd": 0.0, "ce": 0.0, "n": 0}
    optim.zero_grad()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader, 1):
            batch = {k: v.to(student.device) for k, v in batch.items()}
            with torch.no_grad():
                t_out = teacher(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"])
                t_logits = t_out.logits.float()
            s_out = student(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])
            s_logits = s_out.logits.float()

            # Mask: only positions where label != -100 contribute
            mask = (batch["labels"] != -100).float()
            # KD loss (KL of soft distributions). Teacher/student can have a
            # slightly different padded vocab size (e.g. Qwen2.5-7B vs 1.5B,
            # 152064 vs 151936) even with the "same" tokenizer -- the extra
            # rows are unused/reserved embedding slots, so truncating both to
            # the common vocab before softmax is a safe, standard fix.
            common_vocab = min(t_logits.size(-1), s_logits.size(-1))
            kd = F.kl_div(
                F.log_softmax(s_logits[..., :common_vocab] / T, dim=-1),
                F.softmax(   t_logits[..., :common_vocab] / T, dim=-1),
                reduction="none",
            ).sum(-1)   # sum over vocab -> per-token
            kd = (kd * mask).sum() / mask.sum().clamp_min(1) * (T * T)

            # Hard-label CE
            ce_all = F.cross_entropy(
                s_logits.view(-1, s_logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100, reduction="mean",
            )

            loss = alpha * ce_all + (1 - alpha) * kd
            (loss / args.grad_accum).backward()
            running["kd"] += float(kd); running["ce"] += float(ce_all); running["n"] += 1

            if i % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in student.parameters() if p.requires_grad], 1.0,
                )
                optim.step(); sched.step(); optim.zero_grad()
                step += 1
                if step % 5 == 0:
                    kd_m = running["kd"] / running["n"]
                    ce_m = running["ce"] / running["n"]
                    print(f"  epoch {epoch+1}  step {step}/{total_steps}  "
                          f"loss={float(loss):.4f}  kd={kd_m:.4f}  ce={ce_m:.4f}  "
                          f"lr={sched.get_last_lr()[0]:.2e}", flush=True)
                    running = {"kd": 0.0, "ce": 0.0, "n": 0}

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving distilled student adapter to {out_dir} ...")
    student.save_pretrained(str(out_dir), safe_serialization=True)
    tok_s.save_pretrained(str(out_dir))

    if args.merge_final:
        print("Merging LoRA into student base weights...")
        merged = student.merge_and_unload()
        merged.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="4GB")
        print("Merged.  You can now load DistillVHF-QWEN as a plain HF model.")

    (out_dir / "distill_manifest.txt").write_text(
        f"teacher    : {TEACHER_DIR}\n"
        f"student_id : {args.student_id}\n"
        f"epochs     : {args.epochs}\n"
        f"alpha      : {args.alpha}\n"
        f"temperature: {args.temperature}\n"
        f"merged     : {args.merge_final}\n"
    )
    print(f"Done. Evaluate with:  python eval_finetuned.py --model {out_dir} --tag distill")


def main() -> None:
    """CLI entry point: parse distillation args and run train()."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", type=str, default=DEFAULT_STUDENT,
                    help="student HF model id, or path to a local model (e.g. pruned VHF-QWEN)")
    ap.add_argument("--output",     type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--epochs",     type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr",         type=float, default=1e-4)
    ap.add_argument("--max-len",    type=int, default=1024,
                    help="truncation length; lower to save VRAM if student OOMs")
    ap.add_argument("--alpha",      type=float, default=0.5,
                    help="weight on hard-label CE; (1-alpha) on KD loss")
    ap.add_argument("--temperature",type=float, default=2.0,
                    help="softmax temperature for KD; 2-4 is typical")
    ap.add_argument("--merge-final",action="store_true",
                    help="merge LoRA into base at the end for standalone deployment")
    args = ap.parse_args()

    print("=" * 70)
    print("DistillVHF-QWEN — Knowledge Distillation from VHF-QWEN")
    print("=" * 70)

    if not TEACHER_DIR.exists():
        raise SystemExit(f"Teacher {TEACHER_DIR} not found. Run merge_adapter.py first.")

    train(args)


if __name__ == "__main__":
    main()
