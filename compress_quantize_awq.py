"""
================================================================================
compress_quantize_awq.py — AWQ int4 quantization of the merged VHF-QWEN
================================================================================

WHAT THIS SCRIPT DOES
---------------------
Applies Activation-aware Weight Quantization (AWQ) to compress VHF-QWEN from
bf16 (~14 GB) to int4 (~4 GB) while preserving quality.

AWQ vs NF4 (the training-time quantization we used):
  * NF4 is a "blind" 4-bit format — every weight column is quantized the same
    way based on its own statistics.
  * AWQ inspects activation magnitudes from real calibration data and rescales
    weights so the largest-activation channels are protected against
    quantization noise.  Result: ~1-2 pp better accuracy than NF4 at the same
    size, and 1.5-2x faster inference on modern GPUs.

CALIBRATION DATA
----------------
AWQ needs 32-512 short samples that look like typical inputs.  We use a
stratified sample of the RAG-augmented SFT rows (chunks of VHF text), which
is exactly the distribution the model will see at inference.

DEPENDENCIES (install day-of-run)
---------------------------------
    pip install autoawq

autoawq wraps the AWQ authors' reference implementation with a HuggingFace-
compatible loader that other frameworks (vLLM, TGI, transformers) accept.

OUTPUT
------
_models/VHF-QWEN-awq-int4/     ~4 GB, drop-in replacement for VHF-QWEN

USAGE
-----
    python compress_quantize_awq.py --n-calibration 128
    python compress_quantize_awq.py --skip-eval

NOTES
-----
* AWQ needs the FULL fp16 model resident in GPU or shared memory to compute
  the activation statistics.  On 8 GB VRAM this is slow (~30 min for 7B).
  Consider running this step on a bigger GPU if available.
* Once quantized the model is inference-only: you can't fine-tune AWQ-int4
  weights (they're integer).  Do all training in bf16/NF4, then AWQ last.
"""
from __future__ import annotations
import os, json, argparse, random
from pathlib import Path

W = Path(__file__).resolve().parent
CACHE = W / "Data" / "VHF" / "VHF_Agents_Training"
# AUTOPILOT_MODELS_DIR points at a shared cloud location (e.g. /srv/shared-models)
# when set; otherwise falls back to the repo-local _models/ folder (laptop use).
MODELS = Path(os.environ["AUTOPILOT_MODELS_DIR"]) if os.environ.get("AUTOPILOT_MODELS_DIR") else W / "_models"
VHF_MODELS = MODELS / "VHF"
os.environ.setdefault("HF_HOME", str(MODELS / "hf_cache"))

DEFAULT_IN  = VHF_MODELS / "VHF-QWEN"
DEFAULT_OUT = VHF_MODELS / "VHF-QWEN-awq-int4"
SFT_RAG     = CACHE / "vhf_sft_rag.jsonl"


def build_calibration_texts(n: int) -> list[str]:
    """Sample from vhf_sft_rag.jsonl user messages — these already contain the
    long context+question format the model sees at inference.
    """
    rng = random.Random(0)
    lines = SFT_RAG.read_text(encoding="utf-8").splitlines()
    rng.shuffle(lines)
    texts = []
    for line in lines:
        r = json.loads(line)
        user = next((m["content"] for m in r["messages"] if m["role"] == "user"), None)
        if user and 200 <= len(user) <= 3500:
            texts.append(user)
        if len(texts) >= n: break
    print(f"Calibration texts: {len(texts)}  (lengths min={min(len(t) for t in texts)}, "
          f"max={max(len(t) for t in texts)})")
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=str, default=str(DEFAULT_IN))
    ap.add_argument("--output", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--n-calibration", type=int, default=128)
    ap.add_argument("--group-size", type=int, default=128,
                    help="AWQ group size (128 is the standard)")
    ap.add_argument("--skip-eval", action="store_true",
                    help="no-op here (this script never evaluates) -- accepted so "
                         "run_all.sh's uniform 'quantize --skip-eval' call doesn't error out; "
                         "run eval_finetuned.py separately as printed below")
    args = ap.parse_args()

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "autoawq not installed. Run:\n"
            "  pip install autoawq"
        )

    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    if not in_dir.exists():
        raise SystemExit(f"Input model not found: {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_dir} for AWQ quantization...")
    tok = AutoTokenizer.from_pretrained(str(in_dir), trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        str(in_dir), safetensors=True, device_map="auto",
    )

    calib = build_calibration_texts(args.n_calibration)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.group_size,
        "w_bit": 4,
        "version": "GEMM",     # GEMM: fastest; GEMV: better for batch=1
    }
    print(f"Quantizing with group_size={args.group_size}, w_bit=4, GEMM kernel...")
    model.quantize(tokenizer=tok, quant_config=quant_config, calib_data=calib)

    print(f"Saving quantized model to {out_dir}...")
    model.save_quantized(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "quantize_manifest.txt").write_text(
        f"source          : {in_dir}\n"
        f"method          : AWQ int4\n"
        f"group_size      : {args.group_size}\n"
        f"n_calibration   : {len(calib)}\n"
        f"calibration_src : {SFT_RAG.name}\n"
    )
    print("Done. Load with:")
    print(f'  from awq import AutoAWQForCausalLM')
    print(f'  m = AutoAWQForCausalLM.from_quantized(r"{out_dir}")')
    print(f"\nEvaluate with:  python eval_finetuned.py --model {out_dir} --tag vhf_qwen_awq")


if __name__ == "__main__":
    main()
