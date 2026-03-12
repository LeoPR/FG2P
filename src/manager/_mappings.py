"""
Fonte única de verdade para mapeamento experimento → nome canônico.

Usado por _sync.sync_performance() e _reports.guide() para sincronizar
performance.json e gerar relatórios. Alterar aqui propaga para tudo.
"""
import re
from typing import Optional
from ._constants import PERFORMANCE_PATH

# ---------------------------------------------------------------------------
# Mapeamento: nome de pasta do experimento → nome exibido em performance.json
# ---------------------------------------------------------------------------
EXPERIMENT_TO_NAME: dict[str, str] = {
    "exp0_baseline_70split":                         "FG2P Exp0 (Baseline 70/10/20)",
    "exp0_legacy_simple":                            "FG2P Exp0 LEGACY (Simple 70/10/20 — no stratify/early-stop)",
    "exp1_baseline_60split":                         "FG2P Exp1 (Baseline 60/10/30)",
    "exp2_extended_512hidden":                       "FG2P Exp2 (Extended)",
    "exp3_panphon_trainable":                        "FG2P Exp3 (PanPhon)",
    "exp4_panphon_fixed_24d":                        "FG2P Exp4 (PanPhon Fixed)",
    "exp5_intermediate_60split":                     "FG2P Exp5 (Intermediate)",
    "exp6_distance_aware_loss":                      "FG2P Exp6 (Distance-Aware Loss)",
    "exp7_lambda_lower_bound_0.05":                  "FG2P Exp7 (Lambda Lower Bound (\u03bb=0.05))",
    "exp7_lambda_mid_candidate_0.20":                "FG2P Exp7 (Lambda Mid Candidate (\u03bb=0.20))",
    "exp7_lambda_upper_bound_0.50":                  "FG2P Exp7 (Lambda Upper Bound (\u03bb=0.50))",
    "exp8_panphon_distance_aware":                   "FG2P Exp8 (PanPhon + Distance-Aware)",
    "exp9_intermediate_distance_aware":              "FG2P Exp9 (Intermediate + Distance-Aware)",
    "exp10_extended_distance_aware":                 "FG2P Exp10 (Extended + Distance-Aware)",
    "exp11_baseline_decomposed":                     "FG2P Exp11 (Baseline Decomposed Encoding)",
    "exp101_baseline_60split_separators":            "FG2P Exp101 (Baseline + Syllabic Separator)",
    "exp102_intermediate_60split_separators":        "FG2P Exp102 (Intermediate + Syllabic Separator)",
    "exp103_intermediate_sep_distance_aware":        "FG2P Exp103 (Intermediate + Sep + Distance-Aware)",
    "exp104_intermediate_sep_da_custom_dist":        "FG2P Exp104 (Intermediate + Sep + DA + Custom Dist \u2014 bug)",
    "exp104b_intermediate_sep_da_custom_dist_fixed": "FG2P Exp104b (DA + Sep + Custom Dist Fixed) \u2014 SOTA PER",
    "exp105_reduced_data_50split":                   "FG2P Exp105 (50% train data, with hyphen) \u2014 ablation",
    "exp106_no_hyphen_50split":                      "FG2P Exp106 (50% train data, no hyphen) \u2014 ablation",
    "exp107_maxdata_95train":                        "FG2P Exp107 (95% train, LatPhon comparison \u2014 N=960)",
}


def map_experiment_to_name(exp_name: str) -> Optional[str]:
    """Converte nome interno do experimento para nome canônico de performance.json."""
    if exp_name in EXPERIMENT_TO_NAME:
        return EXPERIMENT_TO_NAME[exp_name]
    # Fallback por prefixo (para experimentos com timestamp no nome)
    for key, display in EXPERIMENT_TO_NAME.items():
        if exp_name.startswith(key):
            return display
    return None


# ---------------------------------------------------------------------------
# Parsing de métricas a partir de evaluation_*.txt
# ---------------------------------------------------------------------------
_EVAL_METRICS_RE = {
    "per":      re.compile(r"PER \(Phoneme Error Rate\):\s*([0-9.]+)%"),
    "wer":      re.compile(r"WER \(Word Error Rate\):\s*([0-9.]+)%"),
    "accuracy": re.compile(r"Accuracy \(Word-level\):\s*([0-9.]+)%"),
}
_EXPERIMENT_RE = re.compile(r"^Experiment:\s*(.+)$", re.MULTILINE)
_TIMESTAMP_RE  = re.compile(r"^Timestamp:\s*(.+)$",  re.MULTILINE)


def parse_eval_metrics(path) -> Optional[dict]:
    """Lê um evaluation_*.txt e retorna dict com per/wer/accuracy (floats %)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    metrics = {}
    for key, rx in _EVAL_METRICS_RE.items():
        m = rx.search(text)
        if m:
            metrics[key] = float(m.group(1))
    return metrics or None


def parse_eval_experiment(path) -> Optional[str]:
    """Extrai nome do experimento de um evaluation_*.txt."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _EXPERIMENT_RE.search(text)
    return m.group(1).strip() if m else None


def parse_eval_timestamp(path) -> Optional[str]:
    """Extrai timestamp de um evaluation_*.txt."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _TIMESTAMP_RE.search(text)
    return m.group(1).strip() if m else None


def load_performance_index() -> dict:
    """Carrega performance.json e retorna dict {name: entry}."""
    import json
    if not PERFORMANCE_PATH.exists():
        return {}
    try:
        data = json.loads(PERFORMANCE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {m.get("name"): m for m in data.get("fg2p_models", []) if m.get("name")}
