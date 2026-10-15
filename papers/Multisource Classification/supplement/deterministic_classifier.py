from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass

CLASSES = ("healthy", "benign_lesion", "opmd", "cancer")
SEVERITY = {name: rank for rank, name in enumerate(CLASSES)}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    category: str
    pattern: re.Pattern[str]


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _rule(rule_id: str, category: str, pattern: str) -> Rule:
    return Rule(rule_id, category, re.compile(pattern))


# Explicit precedence is clinically conservative: cancer > OPMD > benign >
# healthy.  A negated-atypia phrase is handled before the OPMD rules.
RULES: tuple[Rule, ...] = (
    _rule(
        "C01_frank_malignancy",
        "cancer",
        r"\b(carcinoma|adenocarcinoma|linfoma|melanoma|sarcoma)\b|"
        r"\bcec\b|carcinoma\s+espino\s*celular|carcinoma\s+epidermoide",
    ),
    _rule(
        "O01_dysplasia_or_atypia",
        "opmd",
        r"\b(displasia|desplasia)\b|\batipias?\b|\batipica\b",
    ),
    _rule(
        "O02_leukoplakia_erythroplakia",
        "opmd",
        r"\b(leuco|leuko)plasia\b|\beritroplasia\b|\blvp\b",
    ),
    _rule(
        "O03_actinic_cheilitis",
        "opmd",
        r"\b(queilite|quelite|quielite)\s+actinica\b|"
        r"\bles(ao|oes)\s+actinica\b|\bqa\b",
    ),
    _rule(
        "O04_lichen_lichenoid",
        "opmd",
        r"\b(liquen|lichen)\s+plano\b|\bliquenoide\b|\bmucosite\s+liquenoide\b|"
        r"\blpo\b|\bllo\b",
    ),
    _rule(
        "O05_explicit_potentially_malignant",
        "opmd",
        r"\bpotencialmente\s+maligna\b|\bdoenca\s+potencialmente\s+maligna\b",
    ),
    _rule(
        "O06_atypical_proliferative_lesion",
        "opmd",
        r"\blesao\s+escamosa\s+proliferativa\b|"
        r"\blesao\s+glandular\s+proliferativa\s+atipica\b",
    ),
    _rule(
        "O07_nicotinic_stomatitis",
        "opmd",
        r"\b(estomatite|dermatite)\s+nicotinica\b",
    ),
    _rule(
        "H01_explicit_absence_of_lesion",
        "healthy",
        r"\bsem\s+(sinais\s+de\s+)?les(ao|oes)\b|\bmucosa\s+normal\b|"
        r"\bexame\s+normal\b|\bremissao\s+completa\b",
    ),
    _rule("H02_normal_variant_torus", "healthy", r"\btorus\b"),
    _rule(
        "B01_reactive_hyperplasia",
        "benign_lesion",
        r"\bhiperplasia\b|\bhiperplasica\b|\bhfi\b|\bhf\b|\bfibroma\b|"
        r"\bfibrolipomatosa\b",
    ),
    _rule(
        "B02_friction_or_trauma",
        "benign_lesion",
        r"\b(hiper)?(queratose|ceratose)\s+friccional\b|\btrauma\b|"
        r"\btraumatica\b|\bmordida\b|\bmordiscada\b|\blesao\s+reativa\b",
    ),
    _rule(
        "B03_infectious",
        "benign_lesion",
        r"\bcandid|\bherpes\b|\bhersim\b|\bmicose\b|\bleishmaniose\b|"
        r"\bmononucleose\b|\bactinomyces\b",
    ),
    _rule(
        "B04_cyst_or_salivary",
        "benign_lesion",
        r"\bcisto\b|\bcistica\b|\bmucocele\b|\bsialocisto\b|\bceratocisto\b",
    ),
    _rule(
        "B05_vascular",
        "benign_lesion",
        r"\bhemangioma\b|\bvascular\b|\bvarizes?\b",
    ),
    _rule(
        "B06_benign_tumor_or_pigmented",
        "benign_lesion",
        r"\blipoma\b|\bpapiloma\b|\bmelanoacantoma\b|\bmelanotica\b|"
        r"\bmelanocitica\b|\bnevo\b|\btatuagem\b",
    ),
    _rule(
        "B07_inflammatory_or_immune",
        "benign_lesion",
        r"\bgranuloma\b|\bgranulomatosa\b|\binflamacao\b|\bestomatite\s+protetica\b|"
        r"\bpenfigo\b|\bpenfigoide\b|\bpsoriase\b|\bglossite\b|"
        r"\bqueilite\s+(angular|glandular)\b",
    ),
    _rule(
        "B08_other_specific_benign",
        "benign_lesion",
        r"\bexostose\b|\bosteoradionecrose\b|\bfistula\b|"
        r"\blesao\s+fibro\s+ossea\b|\beritema\s+migratorio\s+benigno\b",
    ),
)


def classify_phrase(text: str) -> tuple[str, str]:
    value = normalize(text)
    # "Sem atipias" is evidence against the otherwise broad atypia rule.
    value_for_opmd = re.sub(r"\bsem\s+atipias?\b", "", value)
    matches: list[Rule] = []
    for rule in RULES:
        candidate = value_for_opmd if rule.category == "opmd" else value
        if rule.pattern.search(candidate):
            matches.append(rule)
    if not matches:
        return "unknown", "UNMAPPED"
    winner = max(matches, key=lambda item: SEVERITY[item.category])
    return winner.category, winner.rule_id

