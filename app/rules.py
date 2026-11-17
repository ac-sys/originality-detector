"""Plain-English AI Rule Engine (Stage 1).
Allows lecturers and administrators to write rules in natural English that are compiled into:
  1. Lexical / phrase pattern matchers (quoted terms or 'such as / like / containing' lists)
  2. Structural & stylometric checks (first-person voice, citations, code/math formulas, sentence length uniformity)
  3. Semantic concept matching via the local sentence embedder (when a rule describes a style/topic without literal quotes)
Rules run on top of the statistical v3 ensemble (fakespot + desklib) and adjust both sentence-level and document-level
AI probabilities with full explainability in the report."""
import json, re, time, uuid
import numpy as np
from .config import DATA, LIGHT_MODE

RULES_PATH = DATA / "ai_rules.json"

DEFAULT_RULES = [
    {
        "id": "rule-ai-cliches",
        "instruction": "Flag sentences containing typical AI boilerplate phrases such as 'delve into', 'rich tapestry', 'testament to', 'moreover, it is important to note', 'underscores the importance', 'it is worth noting that', 'crucial role in shaping'",
        "effect": "increase_ai",
        "weight": 0.25,
        "enabled": True,
        "created": 1763380000.0,
    },
    {
        "id": "rule-ai-uniform-burstiness",
        "instruction": "Flag passages with uniform sentence length and low burstiness typical of AI prose",
        "effect": "increase_ai",
        "weight": 0.10,
        "enabled": True,
        "created": 1763380000.5,
    },
    {
        "id": "rule-fieldwork-human",
        "instruction": "Reduce AI score when sentences contain specific fieldwork or empirical phrases such as 'in our fieldwork', 'during my interview', 'we measured', 'participants reported that', 'in our laboratory'",
        "effect": "decrease_ai",
        "weight": 0.20,
        "enabled": True,
        "created": 1763380001.0,
    },
    {
        "id": "rule-exempt-code-stats",
        "instruction": "Do not flag technical formulas, statistical notation, or code snippets such as 'p < 0.05', 'p < 0.01', 'SELECT * FROM', 'def ', 'import numpy', 'import torch', 'https://', 'doi:'",
        "effect": "exempt",
        "weight": 1.0,
        "enabled": True,
        "created": 1763380002.0,
    },
]

_QUOTED = re.compile(r"['\"“”‘’]([^'\"“”‘’]{2,120})['\"“”‘’]")
_CITE_RX = re.compile(r"\([A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?,?\s*(?:19|20)\d\d\)|\[\d{1,3}\]")
_FIRST_PERSON_RX = re.compile(r"\b(i|my|myself)\b", re.I)
_CODE_MATH_RX = re.compile(r"(\bdef\s+\w+\s*\(|\bSELECT\s+.+\bFROM\b|\bimport\s+[a-zA-Z_]\w*|p\s*[<=>]\s*0?\.\d+|\b(?:F|t|r|z|chi2)\s*\(\d+|\b\d+\.\d+\s*±\s*\d+\.\d+|\\(?:sum|frac|int|alpha|beta)\b)")


def _phrase_in_text(phrase: str, text_low: str) -> bool:
    """Match phrase on whole-word boundaries so 'import' never matches inside 'important'."""
    pattern = r"(?<![a-z0-9_])" + re.escape(phrase) + r"(?![a-z0-9_])"
    return bool(re.search(pattern, text_low))


def compile_english_rule(instruction: str, effect: str = None, weight: float = None):
    """Parse a plain-English rule instruction into executable lexical, structural, and semantic checks."""
    low = instruction.strip().lower()
    if not effect:
        if any(p in low for p in ("do not flag", "don't flag", "exempt", "ignore", "exclude", "never flag")):
            effect = "exempt"
        elif any(p in low for p in ("reduce", "lower", "decrease", "human", "less likely", "penalize less", "discount")):
            effect = "decrease_ai"
        else:
            effect = "increase_ai"

    if weight is None:
        m_pct = re.search(r"(\d{1,2})\s*%", low)
        if m_pct:
            weight = round(min(0.9, max(0.05, int(m_pct.group(1)) / 100.0)), 2)
        else:
            weight = 1.0 if effect == "exempt" else (0.25 if effect == "increase_ai" else 0.20)

    phrases = [m.group(1).strip().lower() for m in _QUOTED.finditer(instruction) if m.group(1).strip()]
    if not phrases:
        m_list = re.search(r"(?:such as|like|including|containing|phrases|words|terms)\s*:?\s*(.+)$", instruction, re.I)
        if m_list:
            tail = m_list.group(1).strip().rstrip(".")
            for part in re.split(r",|\bor\b|\band\b", tail):
                clean = part.strip(" '\"“”‘’.;:()")
                if 2 <= len(clean) <= 80:
                    phrases.append(clean.lower())

    structural = []
    if any(k in low for k in ("no citation", "without citation", "missing citation", "uncited", "no reference")):
        structural.append("no_citations")
    if any(k in low for k in ("with citation", "contains citation", "has citation", "properly cited")):
        structural.append("has_citations")
    if not phrases and any(k in low for k in ("first-person", "first person", "personal pronoun", "personal reflection")):
        structural.append("first_person")
    if any(k in low for k in ("code", "formula", "equation", "statistical notation", "math")):
        structural.append("code_or_math")
    if any(k in low for k in ("uniform sentence length", "monotonous", "low burstiness", "same length")):
        structural.append("uniform_length")

    min_words = None
    max_words = None
    m_over = re.search(r"(?:over|more than|longer than|at least|>=?)\s*(\d+)\s*words", low)
    if m_over:
        min_words = int(m_over.group(1))
    m_under = re.search(r"(?:under|fewer than|shorter than|less than|<=?)\s*(\d+)\s*words", low)
    if m_under:
        max_words = int(m_under.group(1))

    semantic_query = None
    if not phrases and not structural and min_words is None and max_words is None:
        cleaned = re.sub(
            r"^(?:please\s+)?(?:always\s+)?(?:flag|mark|increase|reduce|lower|decrease|exempt|ignore|do not flag)\s+(?:sentences|passages|paragraphs|text)?\s*(?:that|which|when|if|with|containing|about)?\s*",
            "",
            instruction.strip(),
            flags=re.I,
        ).strip(" .")
        if len(cleaned.split()) >= 3:
            semantic_query = cleaned

    return {
        "effect": effect,
        "weight": float(weight),
        "phrases": phrases,
        "structural": structural,
        "min_words": min_words,
        "max_words": max_words,
        "semantic_query": semantic_query,
    }


class RuleEngine:
    def __init__(self, path=RULES_PATH):
        self.path = path
        self._EnsureLoaded()

    def _EnsureLoaded(self):
        if not self.path.exists():
            self.save_rules(DEFAULT_RULES)

    def list_rules(self):
        self._EnsureLoaded()
        try:
            rules = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            rules = list(DEFAULT_RULES)
        for r in rules:
            if "compiled" not in r:
                r["compiled"] = compile_english_rule(r.get("instruction", ""), r.get("effect"), r.get("weight"))
        return rules

    def save_rules(self, rules):
        for r in rules:
            c = compile_english_rule(r.get("instruction", ""), r.get("effect"), r.get("weight"))
            r["effect"] = c["effect"]
            r["weight"] = c["weight"]
            r["compiled"] = c
        self.path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
        return rules

    def add_rule(self, instruction: str, effect: str = None, weight: float = None, enabled: bool = True):
        if not instruction or not instruction.strip():
            raise ValueError("Rule instruction cannot be empty")
        rules = self.list_rules()
        c = compile_english_rule(instruction.strip(), effect, weight)
        item = {
            "id": "rule-" + uuid.uuid4().hex[:8],
            "instruction": instruction.strip(),
            "effect": c["effect"],
            "weight": c["weight"],
            "enabled": bool(enabled),
            "created": time.time(),
            "compiled": c,
        }
        rules.append(item)
        self.save_rules(rules)
        return item

    def update_rule(self, rule_id: str, payload: dict):
        rules = self.list_rules()
        for r in rules:
            if r["id"] == rule_id:
                if "instruction" in payload and payload["instruction"].strip():
                    r["instruction"] = payload["instruction"].strip()
                if "effect" in payload and payload["effect"]:
                    r["effect"] = payload["effect"]
                if "weight" in payload and payload["weight"] is not None:
                    r["weight"] = float(payload["weight"])
                if "enabled" in payload:
                    r["enabled"] = bool(payload["enabled"])
                r["compiled"] = compile_english_rule(r["instruction"], r["effect"], r["weight"])
                self.save_rules(rules)
                return r
        return None

    def delete_rule(self, rule_id: str):
        rules = self.list_rules()
        new_rules = [r for r in rules if r["id"] != rule_id]
        if len(new_rules) == len(rules):
            return False
        self.save_rules(new_rules)
        return True

    def evaluate(self, text: str, sentences: list, ai: dict, embeddings=None):
        """Apply enabled plain-English rules to sentence probabilities, flags, and overall AI score."""
        rules = [r for r in self.list_rules() if r.get("enabled", True)]
        n = len(sentences)
        if not n or not rules:
            ai["rules_applied"] = []
            return [[] for _ in range(n)]

        sent_texts = [s[2] for s in sentences]
        sent_lows = [t.lower() for t in sent_texts]
        word_counts = [len(t.split()) for t in sent_texts]

        # Precompute local sentence length uniformity (low burstiness) in 5-sentence window
        uniform_flags = [False] * n
        for i in range(n):
            win = word_counts[max(0, i - 2):min(n, i + 3)]
            mean_w = float(np.mean(win)) if win else 0.0
            if len(win) >= 4 and mean_w >= 12 and (float(np.std(win)) < 2.2 or float(np.std(win)) / mean_w < 0.15):
                uniform_flags[i] = True

        # Optional semantic embeddings for semantic rules
        sem_vecs = {}
        if not LIGHT_MODE and embeddings is not None and len(embeddings) == n:
            from .similarity import Embedder
            sem_queries = [r["compiled"]["semantic_query"] for r in rules if r["compiled"].get("semantic_query")]
            if sem_queries:
                q_embs = Embedder.encode(sem_queries)
                for q, qe in zip(sem_queries, q_embs):
                    sem_vecs[q] = embeddings @ qe

        per_sent_hits = [[] for _ in range(n)]
        rule_summary = {}
        probs = list(ai.get("sentence_probs", [0.0] * n))
        flags = list(ai.get("sentence_flagged", [p >= 0.5 for p in probs]))

        for i in range(n):
            low = sent_lows[i]
            txt = sent_texts[i]
            wc = word_counts[i]
            for r in rules:
                c = r["compiled"]
                matched_reasons = []

                if c.get("min_words") and wc < c["min_words"]:
                    continue
                if c.get("max_words") and wc > c["max_words"]:
                    continue

                # 1. Lexical phrase check (whole-word boundary aware)
                if c.get("phrases"):
                    hits = [p for p in c["phrases"] if _phrase_in_text(p, low)]
                    if hits:
                        matched_reasons.append(f"matched phrase '{hits[0]}'")

                # 2. Structural check
                for st in c.get("structural", []):
                    if st == "no_citations" and wc >= 18 and not _CITE_RX.search(txt):
                        matched_reasons.append("uncited claim")
                    elif st == "has_citations" and _CITE_RX.search(txt):
                        matched_reasons.append("contains citation")
                    elif st == "first_person" and _FIRST_PERSON_RX.search(txt):
                        matched_reasons.append("first-person voice")
                    elif st == "code_or_math" and _CODE_MATH_RX.search(txt):
                        matched_reasons.append("code/statistical notation")
                    elif st == "uniform_length" and uniform_flags[i]:
                        matched_reasons.append("uniform sentence length (low burstiness)")

                # 3. Pure word-length rule (if no phrases or structural checks were specified)
                if not c.get("phrases") and not c.get("structural") and not c.get("semantic_query"):
                    if c.get("min_words") or c.get("max_words"):
                        matched_reasons.append(f"sentence length ({wc} words)")

                # 4. Semantic concept match
                sq = c.get("semantic_query")
                if sq and sq in sem_vecs and float(sem_vecs[sq][i]) >= 0.84:
                    matched_reasons.append(f"semantic match ({sem_vecs[sq][i]:.2f})")

                if not matched_reasons:
                    continue

                eff = c["effect"]
                w = float(c["weight"])
                old_p = probs[i]
                if eff == "exempt":
                    probs[i] = round(min(old_p, 0.05), 3)
                    flags[i] = False
                elif eff == "increase_ai":
                    probs[i] = round(min(0.99, old_p + w), 3)
                    if probs[i] >= 0.55:
                        flags[i] = True
                elif eff == "decrease_ai":
                    probs[i] = round(max(0.01, old_p - w), 3)
                    if probs[i] < 0.45:
                        flags[i] = False

                hit_info = {
                    "id": r["id"],
                    "instruction": r["instruction"],
                    "effect": eff,
                    "weight": w,
                    "reason": ", ".join(matched_reasons),
                    "delta": round(probs[i] - old_p, 3),
                }
                per_sent_hits[i].append(hit_info)
                s_entry = rule_summary.setdefault(
                    r["id"],
                    {
                        "id": r["id"],
                        "instruction": r["instruction"],
                        "effect": eff,
                        "weight": w,
                        "sentences_matched": 0,
                    },
                )
                s_entry["sentences_matched"] += 1

        # Recompute document-level share_flagged and apply bounded document probability adjustment
        tot_words = max(1, sum(word_counts))
        old_share = ai.get("share_flagged", 0.0)
        new_share = round(sum(word_counts[i] for i, f in enumerate(flags) if f) / tot_words, 3)
        ai["sentence_probs"] = probs
        ai["sentence_flagged"] = flags
        ai["share_flagged"] = new_share

        if ai.get("probability") is not None and rule_summary:
            # Word-weighted net delta across the document
            weighted_delta = sum(
                word_counts[i] * sum(h["delta"] for h in per_sent_hits[i]) for i in range(n)
            ) / tot_words
            # Also factor change in flagged share
            share_delta = (new_share - old_share) * 0.35
            net_adj = max(-0.35, min(0.35, weighted_delta + share_delta))
            ai["base_probability"] = ai["probability"]
            ai["probability"] = round(min(0.99, max(0.01, ai["probability"] + net_adj)), 3)
            if ai["probability"] >= 0.85 and new_share >= 0.25 and ai.get("verdict") == "no_strong_signal":
                ai["verdict"] = "possible_ai"
                ai["verdict_text"] = "Possible AI-writing signal (elevated by custom English AI rules)"
            elif ai["probability"] < 0.25 and new_share < 0.05 and ai.get("verdict") != "no_strong_signal":
                ai["verdict"] = "no_strong_signal"
                ai["verdict_text"] = "No strong AI-writing signal (adjusted by custom English AI rules)"

        ai["rules_applied"] = list(rule_summary.values())
        return per_sent_hits

