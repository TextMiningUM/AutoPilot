"""
================================================================================
ragas_metrics.py — RAGAS-faithful metric suite + domain metrics (suite v2)
================================================================================

Implements the RAGAS metric *definitions* (claim decomposition, per-claim
verification, reverse-question relevancy, claim-level correctness F1, context
precision/recall) directly against the project's existing OpenAI client.

WHY NOT THE `ragas` PIP PACKAGE: ragas 0.4.3 force-downgrades `openai` from
3.x to 1.x (via its `instructor<2` pin) and pulls the entire langchain +
langgraph stack, which would break every other pipeline script. The metric
algorithms are published and simple; we implement them 1:1 with
domain-adapted extraction prompts (the migration plan's stated fallback and
its "radio-procedure text may decompose oddly" mitigation).

METRICS
-------
LLM-judged (gpt-4o-mini, cached):
  Faithfulness      RAG configs only. answer→statements, each verified against
                    the retrieved contexts; score = supported fraction.
  CorpusGrounded    closed-book configs. RAGAS-inspired, separately named:
                    same statement decomposition, but each statement is
                    verified against its own top-k retrieval from the full
                    RAG chunk corpus.
  ContextPrecision  RAG configs. Rank-weighted usefulness of each retrieved
                    chunk for reaching the gold answer (average precision).
  ContextRecall     RAG configs. Fraction of gold_claims attributable to the
                    retrieved contexts.
  AnswerRelevancy   All configs. 3 reverse-generated questions from the
                    answer; mean cosine to the original question; 0 if the
                    answer is noncommittal.
  AnswerCorrectness All configs. Answer statements vs gold_claims classified
                    TP/FP/FN → F1, blended RAGAS-style:
                    0.75*F1 + 0.25*SemSim(gold_answer, answer).
Deterministic (free):
  NumericF1         Role-aware numeric precision+recall vs gold number/
                    direction claims. Precision catches INVENTED numbers
                    (the old NumHit was recall-only). Vacuously 1.0 when the
                    gold record has no number/direction claims, so every row
                    keeps the same composite mix.
  LitHit            Fraction of gold literal claims (prowords) present.
                    Vacuously 1.0 when gold has no literal claims.
  Cover             Kept from the old suite (sentence-level embedding recall
                    of expected_points).

COMPOSITE (fixed weights per config kind — never re-weighted per row):
  RAG configs:    AnswerCorrectness .35, Faithfulness .20, ContextRecall .10,
                  NumericF1 .15, Cover .10, AnswerRelevancy .05, LitHit .05
  closed-book:    AnswerCorrectness .35, CorpusGrounded .20, NumericF1 .15,
                  Cover .10, AnswerRelevancy .05, LitHit .05 (÷ 0.90)
  A row with a failed judge call gets Composite = nan and is excluded from
  means/CIs (flagged), instead of silently changing the weight mix.

gold_claims come from the Claude-enriched `<gold stem>_claims.json`
(fallback: `<gold stem>_claims_progress.jsonl` for partial pilots) — see
pipeline.eval.enrich_gold_claims / pipeline.eval.gold_claims.

Every summary produced with this suite must carry `metric_suite`
(METRIC_SUITE_VERSION) + `judge_model` so old and new numbers are never
accidentally compared.

Pilot CLI (old-vs-new on existing answers, no inference):

    python -X utf8 -m pipeline.eval.ragas_metrics --pilot Data/VHF/VHF_Agents_Training/eval_qwen_base.jsonl --n 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

METRIC_SUITE_VERSION = "ragas_v2.0"
JUDGE_MODEL = "gpt-4o-mini"

RAG_METRICS = ["AnswerCorrectness", "Faithfulness", "ContextPrecision",
               "ContextRecall", "AnswerRelevancy", "NumericF1", "LitHit", "Cover"]
CLOSED_METRICS = ["AnswerCorrectness", "CorpusGrounded", "AnswerRelevancy",
                  "NumericF1", "LitHit", "Cover"]
# diagnostic P/R splits, free embedding metrics + ProcOrder: reported in every
# summary, never weighted in the composite (AnswerCorrectness/NumericF1 already
# carry their F1s; SemSim is already blended 1/4 inside AnswerCorrectness)
DETAIL_METRICS = ["ClaimPrec", "ClaimRec", "ClaimF1", "NumericPrec", "NumericRec",
                  "SemSim", "AnsRelCos", "ProcOrder"]

COMPOSITE_WEIGHTS = {
    "rag": {"AnswerCorrectness": 0.35, "Faithfulness": 0.20, "ContextRecall": 0.10,
            "NumericF1": 0.15, "Cover": 0.10, "AnswerRelevancy": 0.05, "LitHit": 0.05},
    "closed": {"AnswerCorrectness": 0.35, "CorpusGrounded": 0.20,
               "NumericF1": 0.15, "Cover": 0.10, "AnswerRelevancy": 0.05, "LitHit": 0.05},
}


# ════════════════════════════════════════════════════════════════════════
# LLM prompts (RAGAS definitions, domain-adapted wording)
# ════════════════════════════════════════════════════════════════════════
STATEMENTS_PROMPT = """Decompose the answer below into atomic, self-contained factual statements.
Rules: one verifiable statement per item; no pronouns (resolve "it"/"this channel" to the actual referent); keep exact numbers, channel numbers, rule numbers, prowords and directions (port/starboard) verbatim; ignore filler, greetings and hedging. Radio call transcripts count: extract the channel used, the prowords used, and the call structure as separate statements.
Return JSON: {{"statements": ["...", "..."]}}
Question (for context only): {q}
Answer to decompose:
{a}"""

VERIFY_PROMPT = """For each numbered statement, decide whether it can be directly inferred from the CONTEXT below (1) or not (0). Judge strictly: a statement with a channel number, rule number, proword or direction that the context does not support scores 0. General maritime knowledge NOT in the context scores 0.
Return JSON: {{"verdicts": [0 or 1, ...]}} with exactly {n} entries, in order.
CONTEXT:
{ctx}
STATEMENTS:
{stmts}"""

CTX_PRECISION_PROMPT = """Given the question and the reference answer, decide for each numbered context chunk whether it was useful for arriving at the reference answer (1) or not (0).
Return JSON: {{"verdicts": [0 or 1, ...]}} with exactly {n} entries, in order.
Question: {q}
Reference answer: {g}
CONTEXT CHUNKS:
{chunks}"""

CTX_RECALL_PROMPT = """For each numbered reference claim, decide whether it can be attributed to (found in) the CONTEXT below (1) or not (0).
Return JSON: {{"verdicts": [0 or 1, ...]}} with exactly {n} entries, in order.
CONTEXT:
{ctx}
REFERENCE CLAIMS:
{claims}"""

REVERSE_Q_PROMPT = """Generate exactly 3 different questions that the answer below would directly and fully answer. Also flag whether the answer is noncommittal (evasive, "I don't know", refuses, or gives no substantive information).
Return JSON: {{"questions": ["...", "...", "..."], "noncommittal": 0 or 1}}
Answer:
{a}"""

CORRECTNESS_PROMPT = """Classify the relationship between the candidate's statements and the reference claims for a marine VHF/COLREG exam question.
- TP: candidate statements supported by at least one reference claim
- FP: candidate statements not supported by any reference claim (including wrong numbers/channels/rules/directions)
- FN: reference claims not covered by any candidate statement
Every candidate statement must land in TP or FP; every reference claim not covered goes to FN. Match on meaning, but numbers/channels/rules/directions must be exact to count as supported.
Return JSON: {{"TP": ["..."], "FP": ["..."], "FN": ["..."]}}
Question: {q}
REFERENCE CLAIMS:
{claims}
CANDIDATE STATEMENTS:
{stmts}"""


# ════════════════════════════════════════════════════════════════════════
# Deterministic domain metrics: role-aware NumericF1, claim-driven LitHit
# ════════════════════════════════════════════════════════════════════════
# role → extraction regexes over the lowercased answer; group(1) = value
ROLE_PATTERNS: dict[str, list[re.Pattern]] = {
    "vhf_channel":   [re.compile(r"\b(?:channels?|ch\.?)\s*(\d{1,2}\s?[ab]?)\b"),
                      re.compile(r"\bvhf\s+(?:channel\s+)?(\d{1,2}\s?[ab]?)\b")],
    "colreg_rule":   [re.compile(r"\brules?\s+(\d{1,2})\b")],
    "distance_nm":   [re.compile(r"(\d+(?:\.\d+)?)\s*(?:nautical\s+miles?|nm\b)"),
                      re.compile(r"(\d+(?:\.\d+)?)\s*miles?\b")],
    "speed_kn":      [re.compile(r"(\d+(?:\.\d+)?)\s*(?:knots?|kts?\b|kn\b)")],
    "bearing_deg":   [re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:°|degrees?)")],
    "course_deg":    [re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:°|degrees?)")],
    "power_watt":    [re.compile(r"(\d+(?:\.\d+)?)\s*(?:watts?|w\b)")],
    "frequency_mhz": [re.compile(r"(\d+(?:\.\d+)?)\s*mhz")],
    "gross_tonnage": [re.compile(r"(\d[\d,]*)\s*(?:gt\b|gross\s+tons?|gross\s+tonnage)")],
    "sea_area":      [re.compile(r"\bsea\s+area\s+a\s*(\d)\b"), re.compile(r"\ba(\d)\b\s+sea\s+area")],
    "time_interval": [re.compile(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|minutes?|mins?|hours?)\b")],
    "repeat_count":  [re.compile(r"\b(\d+)\s+times\b")],
    "mmsi":          [re.compile(r"\b(\d{9})\b")],
}
# roles that are semantically interchangeable when matching answer vs gold
_ROLE_GROUPS = [{"bearing_deg", "course_deg"}]
_WORD_REPEATS = {"once": "1", "twice": "2", "three": "3", "thrice": "3"}


def _role_compatible(a: str, b: str) -> bool:
    if a == b:
        return True
    return any(a in g and b in g for g in _ROLE_GROUPS)


def _norm_value(v: str) -> str:
    v = str(v).strip().lower().replace(",", "").replace(" ", "")
    return v.lstrip("0") or "0"


def _value_match(gold_v: str, pred_v: str) -> bool:
    g, p = _norm_value(gold_v), _norm_value(pred_v)
    if g == p:
        return True
    # gold ranges like "20-30": either endpoint (or the range itself) counts
    if "-" in g or "–" in g:
        return p in {_norm_value(x) for x in re.split(r"[-–]", g)}
    return False


def extract_numbers(answer: str) -> list[tuple[str, str]]:
    """All role-tagged (value, role) numeric mentions found in the answer."""
    low = answer.lower()
    for w, d in _WORD_REPEATS.items():
        low = re.sub(rf"\b{w}\s+times\b", f"{d} times", low)
        low = re.sub(rf"\brepeated\s+{w}\b", f"{d} times", low)
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for role, pats in ROLE_PATTERNS.items():
        for pat in pats:
            for m in pat.finditer(low):
                key = (_norm_value(m.group(1)), role)
                if key not in seen:
                    seen.add(key)
                    found.append((m.group(1).strip(), role))
    return found


def numeric_f1(answer: str, gold_claims: list[dict]) -> dict:
    """Role-aware numeric+direction P/R/F1 vs gold claims.

    Recall: gold number claims hit when a role-compatible extraction matches
    the value (other_number claims: bare word-boundary presence). Direction
    claims hit on whole-word presence of the value.
    Precision: over the answer's role-tagged extractions — each must match a
    gold number claim of compatible role, but ONLY when gold has number
    claims at all (a gold record silent on numbers can't prove an extra
    number wrong). No number/direction claims in gold → vacuous 1.0 so every
    row keeps the same composite mix.
    """
    nums = [c for c in gold_claims if c.get("type") == "number"]
    dirs = [c for c in gold_claims if c.get("type") == "direction"]
    if not nums and not dirs:
        return {"NumericF1": 1.0, "NumericPrec": 1.0, "NumericRec": 1.0}

    extracted = extract_numbers(answer)
    low = answer.lower()

    rec_hits = 0
    for c in nums:
        v, role = str(c.get("value", "")), c.get("role", "other_number")
        if role == "other_number" or role not in ROLE_PATTERNS:
            if re.search(rf"\b{re.escape(_norm_value(v))}\b", low.replace(",", "")):
                rec_hits += 1
        elif any(_value_match(v, pv) and _role_compatible(role, pr) for pv, pr in extracted):
            rec_hits += 1
    for c in dirs:
        if re.search(rf"\b{re.escape(str(c.get('value', '')).lower())}\b", low):
            rec_hits += 1
    recall = rec_hits / (len(nums) + len(dirs))

    if nums and extracted:
        ok = sum(1 for pv, pr in extracted
                 if any(_value_match(str(c.get("value", "")), pv)
                        and _role_compatible(c.get("role", ""), pr) for c in nums))
        precision = ok / len(extracted)
    else:
        precision = 1.0  # nothing extracted, or gold silent on numbers

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"NumericF1": round(f1, 3), "NumericPrec": round(precision, 3),
            "NumericRec": round(recall, 3)}


def lit_hit_claims(answer: str, gold_claims: list[dict]) -> float:
    """Fraction of gold literal claims (exact prowords/phrases) present in the answer.
    Vacuously 1.0 when gold has no literal claims (fixed composite mix)."""
    lits = [str(c.get("value", "")).upper() for c in gold_claims
            if c.get("type") == "literal" and str(c.get("value", "")).strip()]
    if not lits:
        return 1.0
    up = answer.upper()
    return sum(1 for v in lits if v in up) / len(lits)


# ════════════════════════════════════════════════════════════════════════
# Gold-claims loading
# ════════════════════════════════════════════════════════════════════════
def load_gold_claims(gold_file: Path) -> dict[str, list[dict]]:
    """{record id -> gold_claims}. Prefers the final <stem>_claims.json,
    falls back to the enrichment progress JSONL (partial coverage, pilots)."""
    final = gold_file.with_name(gold_file.stem + "_claims.json")
    if final.exists():
        return {str(r["id"]): r["gold_claims"]
                for r in json.loads(final.read_text(encoding="utf-8"))}
    prog = gold_file.with_name(gold_file.stem + "_claims_progress.jsonl")
    if prog.exists():
        out = {}
        for line in prog.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r["id"])] = r["gold_claims"]
        return out
    return {}


def real_claims(claims: list[dict]) -> list[dict]:
    """gold_claims minus QA_FLAG review notes."""
    return [c for c in claims if not str(c.get("claim", "")).startswith("QA_FLAG:")]


# ════════════════════════════════════════════════════════════════════════
# The scorer
# ════════════════════════════════════════════════════════════════════════
class RagasScorer:
    """All suite-v2 metrics for one domain. Judge calls cached on disk by
    content hash, so re-scoring and reruns are almost free."""

    def __init__(self, client, embedder, paths, judge_model: str = JUDGE_MODEL,
                pg_file: Path | None = None):
        self.client = client
        self.embedder = embedder
        self.paths = paths
        self.judge_model = judge_model
        self.cache_path = paths.cache_dir / "ragas_judge_cache.jsonl"
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["k"]] = row["v"]
        self._cache_f = self.cache_path.open("a", encoding="utf-8")
        self._chunks: list[dict] | None = None
        self._chunk_emb: np.ndarray | None = None
        self._pg = None            # ProceduralGraph, loaded lazily
        self._pg_missing = False   # remembered so we don't retry every row
        # Defaults to the merged <domain>_pg.json; pass e.g. oow_pg_scenario.json for a
        # ProcOrder signal scoped to one source (Track 2 scenario answers should almost
        # always match that PG's near-trivial assess->role->execute backbone).
        self._pg_file = pg_file or (paths.cache_dir / f"{paths.domain.lower()}_pg.json")
        self.n_calls = 0

    # ── LLM plumbing ─────────────────────────────────────────────────
    def _judge(self, kind: str, prompt: str) -> dict | None:
        key = hashlib.sha1(f"{kind}|{self.judge_model}|{prompt}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        try:
            r = self.client.chat.completions.create(
                model=self.judge_model, temperature=0.0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
            )
            out = json.loads(r.choices[0].message.content)
        except Exception as e:
            print(f"  [ragas judge error/{kind}] {type(e).__name__}: {e}", flush=True)
            return None
        self.n_calls += 1
        self.cache[key] = out
        self._cache_f.write(json.dumps({"k": key, "v": out}, ensure_ascii=False) + "\n")
        self._cache_f.flush()
        return out

    @staticmethod
    def _verdicts(out: dict | None, n: int) -> list[int] | None:
        if not out:
            return None
        v = out.get("verdicts")
        if not isinstance(v, list) or len(v) != n:
            return None
        return [1 if x in (1, "1", True) else 0 for x in v]

    def statements(self, question: str, answer: str) -> list[str] | None:
        out = self._judge("stmts", STATEMENTS_PROMPT.format(q=question, a=answer))
        if not out or not isinstance(out.get("statements"), list):
            return None
        return [s for s in out["statements"] if isinstance(s, str) and s.strip()]

    # ── corpus retrieval for CorpusGrounded ──────────────────────────
    def _load_corpus(self) -> None:
        if self._chunks is not None:
            return
        f = self.paths.cache_dir / f"{self.paths.domain.lower()}_rag_chunks.json"
        self._chunks = json.loads(f.read_text(encoding="utf-8"))
        emb_f = self.paths.cache_dir / f"{self.paths.domain.lower()}_ragas_chunk_emb.npy"
        expected_dim = self.embedder.get_sentence_embedding_dimension()
        if emb_f.exists():
            emb = np.load(emb_f)
            if emb.shape == (len(self._chunks), expected_dim):
                self._chunk_emb = emb
                return
        print(f"  [ragas] embedding {len(self._chunks)} corpus chunks (one-off)...", flush=True)
        self._chunk_emb = self.embedder.encode(
            [c["text"] for c in self._chunks], normalize_embeddings=True,
            show_progress_bar=False, batch_size=64,
        )
        np.save(emb_f, self._chunk_emb)

    def _retrieve(self, texts: list[str], k: int = 3) -> list[str]:
        """Union of top-k corpus chunk texts per query text."""
        self._load_corpus()
        q = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        sims = q @ self._chunk_emb.T
        idx: set[int] = set()
        for row in sims:
            idx.update(np.argsort(row)[-k:].tolist())
        return [self._chunks[i]["text"] for i in sorted(idx)]

    # ── RAGAS metrics ────────────────────────────────────────────────
    def faithfulness(self, question: str, answer: str, contexts: list[str]) -> float:
        stmts = self.statements(question, answer)
        if stmts is None:
            return math.nan
        if not stmts:
            return 0.0
        stmt_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(stmts))
        out = self._judge("faith", VERIFY_PROMPT.format(
            n=len(stmts), ctx="\n---\n".join(contexts), stmts=stmt_block))
        v = self._verdicts(out, len(stmts))
        return round(sum(v) / len(v), 3) if v is not None else math.nan

    def corpus_grounded(self, question: str, answer: str, k: int = 3) -> float:
        """RAGAS-inspired (NOT RAGAS faithfulness): statements verified against
        their own top-k retrieval from the full training corpus."""
        stmts = self.statements(question, answer)
        if stmts is None:
            return math.nan
        if not stmts:
            return 0.0
        contexts = self._retrieve(stmts, k=k)
        stmt_block = "\n".join(f"{i+1}. {s}" for i, s in enumerate(stmts))
        out = self._judge("cground", VERIFY_PROMPT.format(
            n=len(stmts), ctx="\n---\n".join(contexts), stmts=stmt_block))
        v = self._verdicts(out, len(stmts))
        return round(sum(v) / len(v), 3) if v is not None else math.nan

    def context_precision(self, question: str, gold_answer: str, contexts: list[str]) -> float:
        """Average precision over the ranked chunk usefulness verdicts (RAGAS definition)."""
        if not contexts:
            return math.nan
        chunk_block = "\n".join(f"{i+1}. {c[:1200]}" for i, c in enumerate(contexts))
        out = self._judge("ctxprec", CTX_PRECISION_PROMPT.format(
            n=len(contexts), q=question, g=gold_answer, chunks=chunk_block))
        v = self._verdicts(out, len(contexts))
        if v is None:
            return math.nan
        if sum(v) == 0:
            return 0.0
        ap, hits = 0.0, 0
        for i, rel in enumerate(v, 1):
            if rel:
                hits += 1
                ap += hits / i
        return round(ap / sum(v), 3)

    def context_recall(self, gold_claims: list[dict], contexts: list[str]) -> float:
        claims = real_claims(gold_claims)
        if not claims or not contexts:
            return math.nan
        claim_block = "\n".join(f"{i+1}. {c['claim']}" for i, c in enumerate(claims))
        out = self._judge("ctxrec", CTX_RECALL_PROMPT.format(
            n=len(claims), ctx="\n---\n".join(contexts), claims=claim_block))
        v = self._verdicts(out, len(claims))
        return round(sum(v) / len(v), 3) if v is not None else math.nan

    def answer_relevancy(self, question: str, answer: str) -> float:
        out = self._judge("ansrel", REVERSE_Q_PROMPT.format(a=answer))
        if not out or not isinstance(out.get("questions"), list) or not out["questions"]:
            return math.nan
        if out.get("noncommittal") in (1, "1", True):
            return 0.0
        qs = [q for q in out["questions"] if isinstance(q, str) and q.strip()][:3]
        emb = self.embedder.encode([question] + qs, normalize_embeddings=True,
                                   show_progress_bar=False)
        sims = emb[1:] @ emb[0]
        return round(float(np.mean(sims)), 3)

    def answer_correctness(self, question: str, answer: str, gold_answer: str,
                           gold_claims: list[dict]) -> float:
        """RAGAS answer correctness scalar: claim-level F1 blended with semantic
        similarity (0.75 * F1 + 0.25 * SemSim)."""
        return self.answer_correctness_detail(question, answer, gold_answer,
                                              gold_claims)["AnswerCorrectness"]

    def answer_correctness_detail(self, question: str, answer: str, gold_answer: str,
                                  gold_claims: list[dict]) -> dict:
        """Full claim-level comparison: ClaimPrec (candidate statements supported
        by gold — the hallucination side), ClaimRec (gold claims covered — the
        completeness side), ClaimF1, and the blended AnswerCorrectness scalar."""
        nanrow = {"AnswerCorrectness": math.nan, "ClaimPrec": math.nan,
                  "ClaimRec": math.nan, "ClaimF1": math.nan}
        claims = real_claims(gold_claims)
        if not claims:
            return nanrow
        stmts = self.statements(question, answer)
        if stmts is None:
            return nanrow
        stmt_block = "\n".join(f"- {s}" for s in stmts) if stmts else "- (no substantive statements)"
        claim_block = "\n".join(f"- {c['claim']}" for c in claims)
        out = self._judge("correct", CORRECTNESS_PROMPT.format(
            q=question, claims=claim_block, stmts=stmt_block))
        if not out:
            return nanrow
        tp = len(out.get("TP", []))
        fp = len(out.get("FP", []))
        fn = len(out.get("FN", []))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
        emb = self.embedder.encode([gold_answer, answer], normalize_embeddings=True,
                                   show_progress_bar=False)
        sem = float(np.dot(emb[0], emb[1]))
        return {"AnswerCorrectness": round(0.75 * f1 + 0.25 * sem, 3),
                "ClaimPrec": round(prec, 3), "ClaimRec": round(rec, 3),
                "ClaimF1": round(f1, 3)}

    # ── ProcOrder: step-order concordance against the Procedural Graph ──
    def proc_order(self, answer: str) -> float:
        """Do the answer's procedure steps respect the step ordering attested in
        the domain's Procedural Graph (build_pg.py)? Answer sentences are
        matched to PG nodes; each ordered pair of matched steps counts as
        concordant when the graph has a NEXT-path in that direction, and as a
        violation when it only has the REVERSE path. Pairs the graph knows
        nothing about are neutral. Vacuous 1.0 with < 2 matched steps.
        Reported alongside the suite, NOT in the composite until calibrated."""
        if self._pg is None and not self._pg_missing:
            from pipeline.ingest.pg_guidance import ProceduralGraph
            if self._pg_file.exists():
                self._pg = ProceduralGraph(self._pg_file, self.embedder)
            else:
                self._pg_missing = True
        if self._pg is None:
            return math.nan
        from pipeline.ingest.pg_guidance import split_step_sentences
        steps = split_step_sentences(answer)
        nodes = [n for n in self._pg.match_many(steps) if n is not None]
        # collapse immediate repeats (one step split across sentences)
        nodes = [n for i, n in enumerate(nodes) if i == 0 or n != nodes[i - 1]]
        if len(nodes) < 2:
            return 1.0
        concordant = violations = 0
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            fwd = self._pg.reachable(u, v)
            bwd = self._pg.reachable(v, u)
            if fwd:
                concordant += 1
            elif bwd:
                violations += 1  # graph knows this order -- and it's reversed
        total = concordant + violations
        return round(concordant / total, 3) if total else 1.0

    # ── row scoring ──────────────────────────────────────────────────
    def score_row(self, question: str, answer: str, gold_answer: str,
                  expected_points: list[str] | None, gold_claims: list[dict],
                  contexts: list[str] | None) -> dict:
        """All suite-v2 metrics for one answer. contexts=None → closed-book."""
        from pipeline.eval.eval_finetuned import cover  # reused unchanged
        kind = "rag" if contexts else "closed"
        # free embedding diagnostics (the old suite's SemSim/AnsRel, kept visible)
        emb = self.embedder.encode([gold_answer, answer, question],
                                   normalize_embeddings=True, show_progress_bar=False)
        m: dict[str, float | None] = {
            "SemSim": round(float(np.dot(emb[0], emb[1])), 3),
            "AnsRelCos": round(float(np.dot(emb[2], emb[1])), 3),
            "AnswerRelevancy": self.answer_relevancy(question, answer),
            **self.answer_correctness_detail(question, answer, gold_answer, gold_claims),
            "Cover": (round(cover(self.embedder, answer, expected_points), 3)
                      if expected_points else 1.0),
            "LitHit": round(lit_hit_claims(answer, real_claims(gold_claims)), 3),
            **numeric_f1(answer, real_claims(gold_claims)),
        }
        po = self.proc_order(answer)
        if not math.isnan(po):
            m["ProcOrder"] = po
        if kind == "rag":
            m["Faithfulness"] = self.faithfulness(question, answer, contexts)
            m["ContextPrecision"] = self.context_precision(question, gold_answer, contexts)
            m["ContextRecall"] = self.context_recall(gold_claims, contexts)
        else:
            m["CorpusGrounded"] = self.corpus_grounded(question, answer)
        m["Composite"] = composite_v2(m, kind)
        return m


def composite_v2(m: dict, kind: str) -> float:
    """Fixed-weight composite; nan in any weighted metric → nan (row flagged),
    never a silent per-row re-weighting."""
    weights = COMPOSITE_WEIGHTS[kind]
    total_w = sum(weights.values())
    num = 0.0
    for k, w in weights.items():
        v = m.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return math.nan
        num += w * v
    return round(num / total_w, 3)


# ════════════════════════════════════════════════════════════════════════
# Paired bootstrap CI
# ════════════════════════════════════════════════════════════════════════
def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 2000,
                     seed: int = 0) -> dict:
    """95% CI of mean(b) - mean(a) over paired per-question scores.
    Pairs where either side is nan are dropped."""
    pairs = [(x, y) for x, y in zip(a, b)
             if not (math.isnan(x) or math.isnan(y))]
    if len(pairs) < 2:
        return {"delta": None, "ci_lo": None, "ci_hi": None, "n_pairs": len(pairs), "significant": None}
    d = np.array([y - x for x, y in pairs])
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": round(float(d.mean()), 4), "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4), "n_pairs": len(d),
            "significant": bool(lo > 0 or hi < 0)}


def summary_stamp() -> dict:
    return {"metric_suite": METRIC_SUITE_VERSION, "judge_model": JUDGE_MODEL}


# ════════════════════════════════════════════════════════════════════════
# Pilot CLI — old suite vs new suite on existing answers, no inference
# ════════════════════════════════════════════════════════════════════════
def _pilot(answers_file: Path, n: int) -> None:
    import os
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    from core import AgentPaths, load_env, EMBEDDER_MODEL
    from pipeline.eval.eval_finetuned import (
        semsim, cover, num_hit, lit_hit, judge_faith, judge_correct, composite,
    )

    paths = AgentPaths.from_env()
    load_env(paths.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")
    client = OpenAI()
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    scorer = RagasScorer(client, embedder, paths)

    claims_by_id = load_gold_claims(paths.gold_file)
    rows = [json.loads(l) for l in answers_file.open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if str(r.get("id", r.get("q_id"))) in claims_by_id][:n]
    if not rows:
        raise SystemExit("No rows with gold_claims available yet — wait for enrichment.")
    print(f"Pilot: {len(rows)} rows from {answers_file.name} (old vs new suite)\n")

    out_rows = []
    for i, r in enumerate(rows, 1):
        rid = str(r.get("id", r.get("q_id")))
        q, g, a = r["question"], r["gold_answer"], r["answer"]
        ep = r.get("expected_points") or []
        old = {
            "SemSim": round(semsim(embedder, g, a), 3),
            "Faith": judge_faith(client, q, g, a),
            "Correct": judge_correct(client, q, g, ep, a),
            "Cover": round(cover(embedder, a, ep), 3) if ep else None,
            "NumHit": round(num_hit(g, a), 3),
            "LitHit": round(lit_hit(g, a), 3),
        }
        old["Composite"] = round(composite(old), 3)
        new = scorer.score_row(q, a, g, ep, claims_by_id[rid], contexts=None)
        out_rows.append({"id": rid, "old": old, "new": new})
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}", flush=True)

    out_f = answers_file.with_name(answers_file.stem + "_pilot_old_vs_new.json")
    out_f.write_text(json.dumps({"stamp": summary_stamp(), "rows": out_rows}, indent=1),
                     encoding="utf-8")

    def mean(key_path):
        vals = [r[key_path[0]].get(key_path[1]) for r in out_rows]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return round(float(np.mean(vals)), 3) if vals else None

    print(f"\n{'metric':<20}{'old':>8}    {'metric':<20}{'new':>8}")
    old_keys = ["SemSim", "Faith", "Correct", "Cover", "NumHit", "LitHit", "Composite"]
    new_keys = ["AnswerRelevancy", "CorpusGrounded", "AnswerCorrectness", "Cover",
                "NumericF1", "LitHit", "Composite"]
    for ok, nk in zip(old_keys, new_keys):
        print(f"{ok:<20}{mean(('old', ok))!s:>8}    {nk:<20}{mean(('new', nk))!s:>8}")
    print(f"\nWrote {out_f}  (judge calls this run: {scorer.n_calls})")


# ════════════════════════════════════════════════════════════════════════
# Re-score CLI — apply the new suite to an existing answers JSONL (Step 5;
# no model inference, only judging). Track 2 rows (with a 'scenario' field)
# also get the behavioral graders + the Track 2 composite.
# ════════════════════════════════════════════════════════════════════════
def _rescore(answers_file: Path, gold_file: Path | None) -> None:
    import os
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer
    from core import AgentPaths, load_env, EMBEDDER_MODEL

    paths = AgentPaths.from_env()
    load_env(paths.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")
    client = OpenAI()
    embedder = SentenceTransformer(EMBEDDER_MODEL)
    scorer = RagasScorer(client, embedder, paths)

    rows = [json.loads(l) for l in answers_file.open(encoding="utf-8") if l.strip()]
    is_track2 = bool(rows and "scenario" in rows[0])
    gf = gold_file or (paths.eval_file(f"{paths.domain.lower()}_colreg_scenarios.json")
                       if is_track2 else paths.gold_file)
    claims_by_id = load_gold_claims(gf)
    if not claims_by_id:
        raise SystemExit(f"No gold_claims next to {gf} -- run enrich_gold_claims first.")

    if is_track2:
        from pipeline.eval.eval_colreg_scenarios import (
            channel_procedure_score, call_format_score, judge_colreg_correct,
            composite_colreg_v2,
        )

    out_file = answers_file.with_name(answers_file.stem + "_ragas.jsonl")
    keys: set[str] = set()
    n_no_claims = 0
    with out_file.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            claims = claims_by_id.get(str(r.get("id", r.get("q_id"))))
            if claims is None:
                n_no_claims += 1
                m: dict = {"no_gold_claims": True}
            else:
                m = scorer.score_row(r["question"], r["answer"], r["gold_answer"],
                                     r.get("expected_points"), claims, contexts=None)
                if is_track2:
                    chan = channel_procedure_score(r["answer"])
                    cfmt = call_format_score(r.get("own_vessel", ""), r["answer"])
                    creg = judge_colreg_correct(client, r["scenario"], r["question"],
                                                r.get("colreg_rules", []),
                                                r.get("expected_points", []), r["answer"])
                    m.pop("Composite", None)
                    m.update({"ChannelProc": chan, "CallFormatOK": cfmt,
                              "ColregCorrect": None if math.isnan(creg) else round(creg, 3)})
                    m["Composite"] = round(composite_colreg_v2({**m, "ColregCorrect": creg}), 3)
                keys.update(k for k, v in m.items() if isinstance(v, (int, float))
                            and not isinstance(v, bool))
            f.write(json.dumps({**r, "metrics": m}, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(rows):
                print(f"  rescored {i}/{len(rows)}", flush=True)
    if n_no_claims:
        print(f"  [warn] {n_no_claims} rows without gold_claims skipped")

    sums: dict[str, list[float]] = {k: [] for k in sorted(keys)}
    for line in out_file.open(encoding="utf-8"):
        m = json.loads(line)["metrics"]
        for k in sums:
            v = m.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                sums[k].append(v)
    means = {k: round(float(np.mean(v)), 3) if v else None for k, v in sums.items()}
    summary = {"source": answers_file.name, "n": len(rows),
               "track": "conversational_compliance" if is_track2 else "knowledge",
               **summary_stamp(), "means": means,
               "counts": {k: len(v) for k, v in sums.items()}}
    summary_file = answers_file.with_name(answers_file.stem + "_ragas_summary.json")
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{answers_file.name} — new-suite means:")
    for k, v in means.items():
        print(f"  {k:<20} {v}")
    print(f"Wrote {out_file}\n      {summary_file}  (judge calls: {scorer.n_calls})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAGAS suite v2: pilot / re-score")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=Path,
                      help="existing eval answers JSONL to re-score old-vs-new (comparison)")
    mode.add_argument("--rescore", type=Path,
                      help="existing eval answers JSONL to fully re-score with the new suite")
    ap.add_argument("--n", type=int, default=50, help="(pilot only) rows to score")
    ap.add_argument("--gold-file", type=Path, default=None,
                    help="(rescore) gold file whose _claims to use (default: auto by track)")
    args = ap.parse_args()
    if args.pilot:
        _pilot(args.pilot, args.n)
    else:
        _rescore(args.rescore, args.gold_file)
