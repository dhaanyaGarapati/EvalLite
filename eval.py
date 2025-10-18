
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import textstat # type: ignore
import spacy # type: ignore
import wikipediaapi # type: ignore

try:
    nlp = spacy.load("en_core_web_sm")
except OSError as e:
    from spacy.cli import download # type: ignore
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="EvalLite/1.0 (dhaanya project; https://github.com/dhaanyaGarapati/EvalLite)"
)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def scale_0_100(x: float) -> float:
    return round(100.0 * clamp01(x), 2)

def fluency_rule_based(text: str):
    if not text.strip():
        return 0.0, {"empty": True}
    fre = textstat.flesch_reading_ease(text)
    fkgl = textstat.flesch_kincaid_grade(text)
    avg_sent = textstat.avg_sentence_length(text)

    fre_norm = clamp01((fre - 0) / 100)
    fkgl_norm = 1.0 - clamp01(fkgl / 18.0)
    avg_sent_norm = 1.0 - clamp01((avg_sent - 12) / 24)

    score = 0.5*fre_norm + 0.3*fkgl_norm + 0.2*avg_sent_norm
    return scale_0_100(score), {
        "fre": fre, "fkgl": fkgl, "avg_sentence_len": avg_sent,
        "fre_norm": round(fre_norm,3),
        "fkgl_norm": round(fkgl_norm,3),
        "avg_sentence_norm": round(avg_sent_norm,3),
    }

def extract_entities(text: str) -> List[Tuple[str, str]]:
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART", "EVENT"}:
            ents.append((ent.text.strip(), ent.label_))
    return ents

def wiki_check(entity_text: str) -> bool:
    if not entity_text or len(entity_text) < 2:
        return False
    page = wiki.page(entity_text)
    return bool(page and page.exists())

def factuality_score(text: str):
    ents = extract_entities(text)
    if not ents:
        return 50.0, {"checked": 0, "matched": 0, "entities": []}
    checked = 0
    matched = 0
    rows = []
    seen = set()
    for e, label in ents:
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        ok = wiki_check(e)
        checked += 1
        matched += 1 if ok else 0
        rows.append({"entity": e, "label": label, "exists_in_wikipedia": bool(ok)})
    frac = matched / max(1, checked)
    return scale_0_100(frac), {"checked": checked, "matched": matched, "entities": rows}
