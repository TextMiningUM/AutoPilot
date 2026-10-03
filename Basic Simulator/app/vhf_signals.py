"""Flags / light / sound / Morse reference data + pure rendering helpers for the VHF
Communications page's "Signals" tab (design_vhf_communications.md Sec 9.1/12.9). Safe to
run locally -- no GPU/model calls, pure data + string functions.

Four sub-channels:
  - Flag MEANINGS are loaded straight from Data/VHF/VHF_JSON/flag_signals.json (the same
    sourced, hand-transcribed-from-Wikipedia content already in the RAG index) -- real,
    sourced text, never duplicated here.
  - Flag APPEARANCE: `FLAG_BLAZONS` stores the real heraldic blazon for every ICS letter
    flag, transcribed VERBATIM from the wikitext of Wikipedia's "International maritime
    signal flags" article (CC BY-SA 4.0, https://en.wikipedia.org/wiki/International_
    maritime_signal_flags, "Letter flags (with ICS meaning)" table's own Blazon column --
    fetched directly, not reconstructed from memory). `FLAG_PATTERNS` is THIS project's own
    interpretation of that prose blazon into a small set of renderable primitives (per
    pale/fess, paly, chequy, saltire, cross, inescutcheon, ...) -- `flag_svg()` renders
    those primitives as an SVG. Geometry (stripe widths, cross thickness, saltire band
    width) is a simplified visual approximation, not pixel-exact to the official IMO ICS
    chart, but the COLORS and PATTERN TYPE are sourced, not guessed.
  - The NATO phonetic alphabet (letter -> code word) is fixed, universally standardized,
    and independently confirmed against this project's own NATO_Phonetic_Alphabet.json.
  - International Morse code (letter/digit -> dot-dash) is a fixed, universal standard.
  - Light/sound manoeuvring + restricted-visibility signal PATTERNS are transcribed
    directly from this project's own light_sound_signals.json (COLREG Rules 32-35, itself
    copied from OOW's colreg_consolidated_2018.json) -- real, sourced, accurate.
"""
from __future__ import annotations
import json
from pathlib import Path

from core.paths import AgentPaths

# ── NATO phonetic alphabet (fixed international standard) ──────────────────
PHONETIC_ALPHABET: dict[str, str] = {
    "A": "Alfa", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo",
    "F": "Foxtrot", "G": "Golf", "H": "Hotel", "I": "India", "J": "Juliett",
    "K": "Kilo", "L": "Lima", "M": "Mike", "N": "November", "O": "Oscar",
    "P": "Papa", "Q": "Quebec", "R": "Romeo", "S": "Sierra", "T": "Tango",
    "U": "Uniform", "V": "Victor", "W": "Whiskey", "X": "X-ray", "Y": "Yankee",
    "Z": "Zulu",
}

# Heraldic tincture -> real color. Standard heraldic palette (white/blue/red/gold/black);
# the specific hex shades are this project's own pick for a clean on-screen render, not an
# official Pantone spec.
TINCTURE_COLORS: dict[str, str] = {
    "argent": "#FFFFFF", "azure": "#00247D", "gules": "#D21034",
    "or": "#FFC72C", "sable": "#1A1A1A",
}

# Real heraldic blazons, transcribed verbatim from the Blazon column of Wikipedia's
# "International maritime signal flags" article (CC BY-SA 4.0) -- see module docstring.
FLAG_BLAZONS: dict[str, str] = {
    "A": "Swallowtailed, per pale argent and azure",
    "B": "Swallowtailed, gules",
    "C": "Azure, a fess gules fimbriated argent",
    "D": "Or, a Spanish fess azure",
    "E": "Per fess azure and gules",
    "F": "Argent, a lozenge throughout gules",
    "G": "Paly of six or and azure",
    "H": "Per pale argent and gules",
    "I": "Or, a pellet",
    "J": "Azure, a fess argent",
    "K": "Per pale or and azure",
    "L": "Quarterly or and sable",
    "M": "Azure, a saltire argent",
    "N": "Chequy of sixteen azure and argent",
    "O": "Per bend gules and or",
    "P": "Azure, an inescutcheon argent",
    "Q": "Or",
    "R": "Gules, a cross or",
    "S": "Argent, an inescutcheon azure",
    "T": "Tierced in pale gules, argent and azure",
    "U": "Quarterly gules and argent",
    "V": "Argent, a saltire gules",
    "W": "Azure, an inescutcheon gules fimbriated argent",
    "X": "Argent, a cross azure",
    "Y": "Bendy sinister of ten or and gules",
    "Z": "Per saltire or, sable, gules and azure",
}

# This project's own interpretation of each blazon above into a renderable primitive
# (see flag_svg()). `swallowtail: True` cuts a V-notch into the fly end (A/B only --
# the only two letters whose blazon says "Swallowtailed").
FLAG_PATTERNS: dict[str, dict] = {
    "A": {"type": "per_pale", "colors": ["argent", "azure"], "swallowtail": True},
    "B": {"type": "solid", "colors": ["gules"], "swallowtail": True},
    "C": {"type": "fess_fimbriated", "field": "azure", "band": "gules", "border": "argent"},
    "D": {"type": "fess_wide", "field": "or", "band": "azure"},
    "E": {"type": "per_fess", "colors": ["azure", "gules"]},
    "F": {"type": "lozenge", "field": "argent", "lozenge": "gules"},
    "G": {"type": "paly", "n": 6, "colors": ["or", "azure"]},
    "H": {"type": "per_pale", "colors": ["argent", "gules"]},
    "I": {"type": "roundel", "field": "or", "roundel": "sable"},
    "J": {"type": "fess", "field": "azure", "band": "argent"},
    "K": {"type": "per_pale", "colors": ["or", "azure"]},
    "L": {"type": "quarterly", "colors": ["or", "sable"]},
    "M": {"type": "saltire", "field": "azure", "saltire": "argent"},
    "N": {"type": "chequy", "n": 4, "colors": ["azure", "argent"]},
    "O": {"type": "per_bend", "colors": ["gules", "or"]},
    "P": {"type": "inescutcheon", "field": "azure", "inner": "argent"},
    "Q": {"type": "solid", "colors": ["or"]},
    "R": {"type": "cross", "field": "gules", "cross": "or"},
    "S": {"type": "inescutcheon", "field": "argent", "inner": "azure"},
    "T": {"type": "tierced_pale", "colors": ["gules", "argent", "azure"]},
    "U": {"type": "quarterly", "colors": ["gules", "argent"]},
    "V": {"type": "saltire", "field": "argent", "saltire": "gules"},
    "W": {"type": "inescutcheon_fimbriated", "field": "azure", "inner": "gules", "border": "argent"},
    "X": {"type": "cross", "field": "argent", "cross": "azure"},
    "Y": {"type": "bendy_sinister", "n": 10, "colors": ["or", "gules"]},
    "Z": {"type": "per_saltire", "colors": ["or", "sable", "gules", "azure"]},
}


def _c(tincture: str) -> str:
    return TINCTURE_COLORS[tincture]


def flag_svg(letter: str, width: int = 160, height: int = 100) -> str:
    """Render FLAG_PATTERNS[letter] as an inline SVG string (sourced colors/pattern type,
    simplified geometry -- see module docstring)."""
    p = FLAG_PATTERNS[letter]
    w, h, shapes = width, height, []
    t = p["type"]

    if t == "solid":
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(p["colors"][0])}"/>')
    elif t == "per_pale":
        c1, c2 = p["colors"]
        shapes.append(f'<rect width="{w/2}" height="{h}" fill="{_c(c1)}"/>')
        shapes.append(f'<rect x="{w/2}" width="{w/2}" height="{h}" fill="{_c(c2)}"/>')
    elif t == "per_fess":
        c1, c2 = p["colors"]
        shapes.append(f'<rect width="{w}" height="{h/2}" fill="{_c(c1)}"/>')
        shapes.append(f'<rect y="{h/2}" width="{w}" height="{h/2}" fill="{_c(c2)}"/>')
    elif t == "tierced_pale":
        c1, c2, c3 = p["colors"]
        for i, c in enumerate((c1, c2, c3)):
            shapes.append(f'<rect x="{i*w/3}" width="{w/3}" height="{h}" fill="{_c(c)}"/>')
    elif t == "paly":
        n = p["n"]
        c1, c2 = p["colors"]
        for i in range(n):
            shapes.append(f'<rect x="{i*w/n}" width="{w/n}" height="{h}" '
                         f'fill="{_c(c1 if i % 2 == 0 else c2)}"/>')
    elif t == "quarterly":
        c1, c2 = p["colors"]
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(c1)}"/>')
        shapes.append(f'<rect x="{w/2}" width="{w/2}" height="{h/2}" fill="{_c(c2)}"/>')
        shapes.append(f'<rect y="{h/2}" width="{w/2}" height="{h/2}" fill="{_c(c2)}"/>')
    elif t == "chequy":
        n = p["n"]
        c1, c2 = p["colors"]
        cw, ch = w / n, h / n
        for row in range(n):
            for col in range(n):
                color = c1 if (row + col) % 2 == 0 else c2
                shapes.append(f'<rect x="{col*cw}" y="{row*ch}" width="{cw}" height="{ch}" '
                             f'fill="{_c(color)}"/>')
    elif t == "per_bend":
        c1, c2 = p["colors"]
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(c1)}"/>')
        shapes.append(f'<polygon points="0,0 0,{h} {w},{h}" fill="{_c(c2)}"/>')
    elif t == "bendy_sinister":
        n = p["n"]
        c1, c2 = p["colors"]
        stripe = max(w, h) / (n / 2)
        shapes.append(
            f'<defs><pattern id="bs_{letter}" width="{stripe*2}" height="{stripe*2}" '
            f'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<rect width="{stripe}" height="{stripe*2}" fill="{_c(c1)}"/>'
            f'<rect x="{stripe}" width="{stripe}" height="{stripe*2}" fill="{_c(c2)}"/>'
            f'</pattern></defs>'
        )
        shapes.append(f'<rect width="{w}" height="{h}" fill="url(#bs_{letter})"/>')
    elif t == "saltire":
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(p["field"])}"/>')
        bw = min(w, h) * 0.22
        sc = _c(p["saltire"])
        shapes.append(f'<line x1="0" y1="0" x2="{w}" y2="{h}" stroke="{sc}" stroke-width="{bw}"/>')
        shapes.append(f'<line x1="{w}" y1="0" x2="0" y2="{h}" stroke="{sc}" stroke-width="{bw}"/>')
    elif t == "per_saltire":
        top, right, bottom, left = p["colors"]
        cx, cy = w / 2, h / 2
        shapes.append(f'<polygon points="0,0 {w},0 {cx},{cy}" fill="{_c(top)}"/>')
        shapes.append(f'<polygon points="{w},0 {w},{h} {cx},{cy}" fill="{_c(right)}"/>')
        shapes.append(f'<polygon points="{w},{h} 0,{h} {cx},{cy}" fill="{_c(bottom)}"/>')
        shapes.append(f'<polygon points="0,{h} 0,0 {cx},{cy}" fill="{_c(left)}"/>')
    elif t == "cross":
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(p["field"])}"/>')
        cc = _c(p["cross"])
        shapes.append(f'<rect y="{h*0.375}" width="{w}" height="{h*0.25}" fill="{cc}"/>')
        shapes.append(f'<rect x="{w*0.41}" width="{w*0.18}" height="{h}" fill="{cc}"/>')
    elif t in ("fess", "fess_wide", "fess_fimbriated"):
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(p["field"])}"/>')
        band_h = h * (0.5 if t == "fess_wide" else 0.3)
        if t == "fess_fimbriated":
            border_h = h * 0.4
            shapes.append(f'<rect y="{(h-border_h)/2}" width="{w}" height="{border_h}" '
                         f'fill="{_c(p["border"])}"/>')
            band_h = h * 0.26
        shapes.append(f'<rect y="{(h-band_h)/2}" width="{w}" height="{band_h}" '
                     f'fill="{_c(p["band"])}"/>')
    elif t in ("inescutcheon", "inescutcheon_fimbriated"):
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(p["field"])}"/>')
        if t == "inescutcheon_fimbriated":
            bw, bh = w * 0.42, h * 0.56
            shapes.append(f'<rect x="{(w-bw)/2}" y="{(h-bh)/2}" width="{bw}" height="{bh}" '
                         f'fill="{_c(p["border"])}"/>')
            iw, ih = w * 0.32, h * 0.42
        else:
            iw, ih = w * 0.4, h * 0.5
        shapes.append(f'<rect x="{(w-iw)/2}" y="{(h-ih)/2}" width="{iw}" height="{ih}" '
                     f'fill="{_c(p["inner"])}"/>')
    elif t == "lozenge":
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(p["field"])}"/>')
        cx, cy = w / 2, h / 2
        shapes.append(f'<polygon points="{cx},{h*0.1} {w*0.85},{cy} {cx},{h*0.9} {w*0.15},{cy}" '
                     f'fill="{_c(p["lozenge"])}"/>')
    elif t == "roundel":
        shapes.append(f'<rect width="{w}" height="{h}" fill="{_c(p["field"])}"/>')
        shapes.append(f'<circle cx="{w/2}" cy="{h/2}" r="{min(w,h)*0.22}" fill="{_c(p["roundel"])}"/>')

    body = "".join(shapes)
    if p.get("swallowtail"):
        notch = w * 0.2
        clip = (f'<clipPath id="swt_{letter}"><polygon points="0,0 {w},0 {w},{h*0.4} '
               f'{w-notch},{h/2} {w},{h*0.6} {w},{h} 0,{h}"/></clipPath>')
        return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
               f'xmlns="http://www.w3.org/2000/svg"><defs>{clip}</defs>'
               f'<g clip-path="url(#swt_{letter})">{body}</g></svg>')
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">{body}</svg>'


def flag_signal_example(letter: str) -> dict[str, str]:
    """Deterministic send/receive illustration for one flag -- no LLM needed, flag
    meanings are fixed/static text, not conversational like VHF radio dialogue."""
    meaning = load_flag_meanings()[letter]
    name = PHONETIC_ALPHABET[letter]
    return {
        "send": f"You hoist flag {letter} ({name}) alone on your signal halyard. Any "
               f"vessel reading it correctly understands: {meaning}",
        "receive": f"You observe flag {letter} ({name}) hoisted alone on another vessel. "
                  f"Per the single-flag ICS code, this means: {meaning}",
    }


# ── International Morse code (fixed international standard) ────────────────
MORSE_CODE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.",
}
_MORSE_TO_LETTER: dict[str, str] = {code: letter for letter, code in MORSE_CODE.items()}


def text_to_morse(text: str) -> str:
    """Encode `text` to Morse -- letters/digits space-separated, ' / ' between words.
    Unsupported characters (punctuation) are dropped silently."""
    words = text.strip().upper().split()
    encoded_words = [
        " ".join(MORSE_CODE[ch] for ch in word if ch in MORSE_CODE)
        for word in words
    ]
    return " / ".join(w for w in encoded_words if w)


def morse_to_text(morse: str) -> str:
    """Decode Morse (letters space-separated, ' / ' between words) back to text."""
    words = morse.strip().split("/")
    decoded_words = [
        "".join(_MORSE_TO_LETTER.get(code, "") for code in word.split())
        for word in words
    ]
    return " ".join(w for w in decoded_words if w)


# ── Flag meanings (loaded from the real sourced RAG content, never duplicated) ─────────
def load_flag_meanings(paths: AgentPaths | None = None) -> dict[str, str]:
    """letter -> ICS single-flag meaning, parsed straight from flag_signals.json's own
    section_id convention (`flag_a`..`flag_z`) -- the same text already in the RAG index."""
    paths = paths or AgentPaths.vhf()
    doc = json.loads((paths.json_dir / "flag_signals.json").read_text(encoding="utf-8"))
    meanings: dict[str, str] = {}
    for chapter in doc["chapters"]:
        for section in chapter["sections"]:
            sid = section["section_id"]
            if sid.startswith("flag_") and len(sid) == len("flag_x"):
                meanings[sid[-1].upper()] = section["text"]
    return meanings


# ── Light/sound manoeuvring + restricted-visibility signals (COLREG Rules 32-35) ───────
# Transcribed directly from Data/VHF/VHF_JSON/light_sound_signals.json's own rule text --
# "short"/"prolonged" match Rule 32's own definitions (short <=1s, prolonged 4-6s).
MANOEUVRING_SIGNALS: list[dict] = [
    {"name": "Altering course to starboard", "pattern": ["short"], "rule": "Rule 34(a)"},
    {"name": "Altering course to port", "pattern": ["short", "short"], "rule": "Rule 34(a)"},
    {"name": "Operating astern propulsion", "pattern": ["short", "short", "short"], "rule": "Rule 34(a)"},
    {"name": "Danger / doubt signal (at least 5 short rapid blasts)",
     "pattern": ["short"] * 5, "rule": "Rule 34(d)"},
]
RESTRICTED_VISIBILITY_SIGNALS: list[dict] = [
    {"name": "Power-driven vessel, making way", "pattern": ["prolonged"],
     "rule": "Rule 35(a)", "interval": "every 2 minutes"},
    {"name": "Power-driven vessel, underway but stopped", "pattern": ["prolonged", "prolonged"],
     "rule": "Rule 35(b)", "interval": "every 2 minutes"},
    {"name": "NUC / RAM / constrained-by-draught / sailing / fishing / towing",
     "pattern": ["prolonged", "short", "short"], "rule": "Rule 35(c)-(f)",
     "interval": "every 2 minutes"},
]

_SYMBOL = {"short": "\u2022", "prolonged": "\u2014"}  # • and —


def pattern_to_symbols(pattern: list[str]) -> str:
    """["short","short"] -> "• •" -- shared rendering for both light flashes (Rule 34(b):
    each flash ~1s, same timing as the whistle signal it supplements) and sound blasts."""
    return " ".join(_SYMBOL[p] for p in pattern)
