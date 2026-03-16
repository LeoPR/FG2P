#!/usr/bin/env python3
"""
Generate comparative visualizations: FG2P vs LatPhon + A-D class distribution.

Output:
    - class_distribution_all_experiments.png: A/B/C/D distribution for all experiments
    - baseline_comparison.png: FG2P vs LatPhon, WFST, ByT5-Small (PER with CI)
    - class_distribution_top5.png: A/B/C/D distribution for top 5 models
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple
import json

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from extract_visualization_data import ExperimentDataExtractor

# ============================================================================
# CONFIG
# ============================================================================

FIGURE_DPI = 300
STYLE = "seaborn-v0_8-darkgrid"
COLORS_CLASSES = {
    "A": "#2ecc71",  # verde (correto/muito próximo)
    "B": "#3498db",  # azul (próximo)
    "C": "#f39c12",  # laranja (distante)
    "D": "#e74c3c",  # vermelho (muito distante)
}

# Dados dos baselines para comparação (ver docs/article/ARTICLE.md §6 e README.md)
BASELINES = [
    {
        "label": "FG2P\n(28.8k words)",
        "per": 0.49,
        "ci_low": 0.47,
        "ci_high": 0.51,
        "color": "#2ecc71",   # verde
        "note": "0.49% [0.47–0.51%]",
    },
    {
        "label": "LatPhon 2025\n(~500 words)",
        "per": 0.86,
        "ci_low": 0.56,
        "ci_high": 1.16,
        "color": "#f39c12",   # laranja
        "note": "0.86% [0.56–1.16%]",
    },
    {
        "label": "WFST / Phonetisaurus\n(~500 words)",
        "per": 2.70,
        "ci_low": 2.20,
        "ci_high": 3.20,
        "color": "#e67e22",   # laranja escuro
        "note": "2.70% ±0.50",
    },
    {
        "label": "ByT5-Small\n(~500 words)",
        "per": 9.1,
        "ci_low": None,
        "ci_high": None,
        "color": "#e74c3c",   # vermelho
        "note": "9.1% (no CI)",
    },
]

# Compat aliases usados no restante do script
FG2P_PER      = BASELINES[0]["per"]
FG2P_CI_LOW   = BASELINES[0]["ci_low"]
FG2P_CI_HIGH  = BASELINES[0]["ci_high"]
FG2P_TEST_SIZE = 28782
LATPHON_PER    = BASELINES[1]["per"]
LATPHON_CI_LOW = BASELINES[1]["ci_low"]
LATPHON_CI_HIGH = BASELINES[1]["ci_high"]
LATPHON_TEST_SIZE = 500


DEFAULT_VIS_POLICY = {
    "exclude_patterns": ["legacy", "107"],
    "strategy_groups": [
        {"group": "Core baseline", "prefix": "exp1_"},
        {"group": "No-sep + DA", "prefix": "exp9_"},
        {"group": "Sep + CE", "prefix": "exp102_"},
        {"group": "Sep + DA", "prefix": "exp103_"},
        {"group": "Sep + DA + dist", "prefix": "exp104b_"},
        {"group": "Full corrected", "prefix": "exp104d_"},
        {"group": "Ablation/errata", "prefix": "exp104c_"},
    ],
}


def _load_visualization_policy() -> dict:
    """Carrega policy de visualização (com fallback seguro para defaults)."""
    policy_path = Path("conf/visualization_policy.json")
    if not policy_path.exists():
        return dict(DEFAULT_VIS_POLICY)

    try:
        loaded = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_VIS_POLICY)

    policy = dict(DEFAULT_VIS_POLICY)
    if isinstance(loaded, dict):
        if isinstance(loaded.get("exclude_patterns"), list):
            policy["exclude_patterns"] = loaded["exclude_patterns"]
        if isinstance(loaded.get("strategy_groups"), list):
            valid_groups = [
                g for g in loaded["strategy_groups"]
                if isinstance(g, dict) and isinstance(g.get("group"), str) and isinstance(g.get("prefix"), str)
            ]
            if valid_groups:
                policy["strategy_groups"] = valid_groups
    return policy


def load_data():
    """Carrega dados de todos os experimentos."""
    extractor = ExperimentDataExtractor(".")
    all_metrics = extractor.load_all_metrics()
    all_metadata = extractor.load_all_metadata()
    return extractor, all_metrics, all_metadata


def safe_print(msg):
    """Print com tratamento de encoding para Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        # Fallback para ASCII
        print(msg.encode('utf-8', errors='replace').decode('utf-8'))


def plot_baseline_comparison(all_metrics):
    """
    Chart 2: FG2P vs all baselines — LatPhon, WFST, ByT5-Small.
    PER with error bars (95% CI where available).
    """
    safe_print("[2/3] Plotting baseline comparison...")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=FIGURE_DPI)

    policy = _load_visualization_policy()
    valid = _valid_metrics(all_metrics, policy)
    best_per_exp, _ = _best_by_metrics(valid)
    fg2p_per = valid[best_per_exp].per if best_per_exp else BASELINES[0]["per"]
    fg2p_words = valid[best_per_exp].total_words if best_per_exp else FG2P_TEST_SIZE
    fg2p_label = (
        f"FG2P\n({_short_label(best_per_exp)})\n({_format_words_k(fg2p_words)} words)"
        if best_per_exp
        else BASELINES[0]["label"]
    )

    fg2p_ci_low = None
    fg2p_ci_high = None
    if best_per_exp:
        best_run_id = valid[best_per_exp].run_id
        error_file = _error_file_for_exp_run(best_per_exp, best_run_id)
        if error_file:
            fg2p_ci_low, fg2p_ci_high = _estimate_per_ci_from_error_file(error_file, fg2p_per)

    if fg2p_ci_low is not None and fg2p_ci_high is not None:
        fg2p_note = f"{fg2p_per:.2f}% [{fg2p_ci_low:.2f}–{fg2p_ci_high:.2f}%]"
    elif best_per_exp:
        fg2p_note = f"{fg2p_per:.2f}% (CI n/d)"
    else:
        fg2p_note = BASELINES[0]["note"]

    baselines = [dict(BASELINES[0]), *BASELINES[1:]]
    baselines[0]["per"] = fg2p_per
    baselines[0]["label"] = fg2p_label
    baselines[0]["note"] = fg2p_note
    baselines[0]["ci_low"] = fg2p_ci_low
    baselines[0]["ci_high"] = fg2p_ci_high

    x_pos = np.arange(len(baselines))

    for i, b in enumerate(baselines):
        ax.bar(i, b["per"], color=b["color"], alpha=0.75, width=0.55,
               edgecolor="black", linewidth=1.5, zorder=2)

        if b["ci_low"] is not None:
            yerr = [[b["per"] - b["ci_low"]], [b["ci_high"] - b["per"]]]
            ax.errorbar(i, b["per"], yerr=yerr, fmt="none",
                        color="black", capsize=9, capthick=2, linewidth=2, zorder=3)

        offset = b["per"] * 0.06 + 0.15
        ax.text(i, b["per"] + offset, b["note"],
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    if fg2p_per < BASELINES[1]["ci_low"]:
        ax.annotate(
            "FG2P PER remains below\nLatPhon lower CI bound (0.56%)",
            xy=(0.5, fg2p_per), xycoords=("data", "data"),
            xytext=(1.5, 2.2), textcoords="data",
            fontsize=9, style="italic", ha="center",
            arrowprops=dict(arrowstyle="-", color="gray", lw=1),
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

    ax.set_ylabel("PER (%)", fontsize=12, fontweight="bold")
    ax.set_title("Baseline Comparison — Phoneme Error Rate (PER)\n"
                 "FG2P vs LatPhon 2025, WFST/Phonetisaurus, ByT5-Small",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b["label"] for b in baselines], fontsize=11)
    ymax = max((b["ci_high"] if b["ci_high"] is not None else b["per"]) for b in baselines)
    ax.set_ylim(0, max(11.5, ymax * 1.15))
    ax.grid(axis="y", alpha=0.3, zorder=1)

    plt.tight_layout()
    plt.savefig("results/baseline_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Saved: results/baseline_comparison.png")
    plt.close()


def _short_label(exp_name):
    """Extrai label curto robusto: exp104b_... -> Exp104b, exp200xxx_... -> Exp200xxx."""
    if not exp_name:
        return "FG2P"
    m = re.match(r"exp([A-Za-z0-9]+)(?:_|$)", exp_name)
    return f"Exp{m.group(1)}" if m else exp_name.replace("exp", "Exp", 1)


def _format_words_k(words: int) -> str:
    """Formata contagem de palavras em escala k com 1 casa decimal."""
    if words >= 1000:
        return f"{words / 1000.0:.1f}k"
    return str(words)


def _latest_error_file_for_exp(exp_name: str) -> Optional[Path]:
    """Retorna o error_analysis mais recente para um experimento."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    pattern = f"*/error_analysis_{exp_name}__*.txt"
    candidates = sorted(results_dir.glob(pattern))
    return candidates[-1] if candidates else None


def _error_file_for_exp_run(exp_name: str, run_id: str) -> Optional[Path]:
    """Retorna error_analysis do run exato; fallback para o mais recente do experimento."""
    results_dir = Path("results")
    if not results_dir.exists():
        return None

    pattern_exact = f"*/error_analysis_{exp_name}__{run_id}.txt"
    exact = list(results_dir.glob(pattern_exact))
    if exact:
        return exact[0]

    return _latest_error_file_for_exp(exp_name)


def _estimate_per_ci_from_error_file(error_file: Path, per_pct: float) -> Tuple[Optional[float], Optional[float]]:
    """Estima IC 95% de Wilson para PER a partir de total de fonemas e PER reportado.

    O arquivo de análise contém as contagens por classe A/B/C/D (total de fonemas).
    Como o número bruto de erros de PER não está explícito no arquivo, usamos
    round(PER * total_fonemas) como aproximação para construir o IC de Wilson.
    """
    try:
        content = error_file.read_text(encoding="utf-8", errors="ignore")
        counts = []
        for cls in ("A", "B", "C", "D"):
            m = re.search(rf"Classe\s+{cls}:\s+(\d+)", content)
            if not m:
                return None, None
            counts.append(int(m.group(1)))

        total_phonemes = sum(counts)
        if total_phonemes <= 0:
            return None, None

        err_count = int(round((per_pct / 100.0) * total_phonemes))
        return _wilson_interval_pct(err_count, total_phonemes)
    except Exception:
        return None, None


def _wilson_interval_pct(error_count: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% para proporção, retornando em porcentagem."""
    if n <= 0:
        return 0.0, 0.0
    p = error_count / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    radius = (z / denom) * np.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return max(0.0, (center - radius) * 100.0), min(100.0, (center + radius) * 100.0)


def _valid_metrics(all_metrics, policy):
    patterns = tuple(policy.get("exclude_patterns", []))
    return {k: v for k, v in all_metrics.items()
            if not any(p in k for p in patterns)}


def _best_by_metrics(valid_metrics):
    if not valid_metrics:
        return None, None
    best_per = min(valid_metrics.items(), key=lambda x: x[1].per)[0]
    best_wer = min(valid_metrics.items(), key=lambda x: x[1].wer)[0]
    return best_per, best_wer


def _highlight_info(exp_name, best_per_exp, best_wer_exp):
    if exp_name == best_per_exp and exp_name == best_wer_exp:
        return "★ Best PER/WER", "#d1c4e9"
    if exp_name == best_per_exp:
        return "★ Best PER", "#c8e6c9"
    if exp_name == best_wer_exp:
        return "★ Best WER", "#bbdefb"
    return None, None


def plot_class_distribution_top5(all_metrics):
    """
    Chart: strategy-grouped lineup + explicit ablation row.
    Exp104d é a referência principal; Exp104b fica como marco histórico;
    Exp104c aparece apenas como ablação/errata.
    Left: A/B/C/D table. Right: B/C/D bars with adjusted scale.
    """
    safe_print("[3/3] Plotting class distribution (strategy-grouped)...")

    policy = _load_visualization_policy()
    valid = _valid_metrics(all_metrics, policy)
    best_per_exp, best_wer_exp = _best_by_metrics(valid)

    # Seleção determinística por estratégia (vitrine) + uma linha de ablação.
    grouped_specs = [
        (g["group"], g["prefix"]) for g in policy.get("strategy_groups", [])
    ]

    selected = []
    for group_name, prefix in grouped_specs:
        candidates = [e for e in valid if e.startswith(prefix)]
        if not candidates:
            continue
        exp_name = min(candidates, key=lambda e: valid[e].per)
        selected.append((group_name, exp_name))

    experiments = [e for _, e in selected]
    groups = [g for g, _ in selected]

    labels_short = [_short_label(e) for e in experiments]

    highlights = []
    for e in experiments:
        if e == best_per_exp:
            highlights.append(("★ Best PER", "#c8e6c9"))
        elif e.startswith("exp9_"):
            highlights.append(("★ WER anchor", "#bbdefb"))
        elif e.startswith("exp104c_"):
            highlights.append(("Ablation", "#ffe0b2"))
        else:
            highlights.append((None, None))

    # Labels com badge em segunda linha para modelos destacados
    labels_display = [
        f"{lbl}\n{badge}" if badge else lbl
        for lbl, (badge, _) in zip(labels_short, highlights)
    ]

    class_a = [all_metrics[e].class_a_pct for e in experiments]
    class_b = [all_metrics[e].class_b_pct for e in experiments]
    class_c = [all_metrics[e].class_c_pct for e in experiments]
    class_d = [all_metrics[e].class_d_pct for e in experiments]

    fig, (ax_table, ax_chart) = plt.subplots(1, 2, figsize=(15, 6), dpi=FIGURE_DPI,
                                              gridspec_kw={"width_ratios": [1, 1.6]})

    # ===== Esquerda: Tabela A/B/C/D =====
    ax_table.axis("off")
    col_labels = ["Group", "Model", "A (%)", "B (%)", "C (%)", "D (%)"]
    table_data = [
        [grp, lbl, f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{d:.2f}"]
        for grp, lbl, a, b, c, d in zip(groups, labels_short, class_a, class_b, class_c, class_d)
    ]
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.0, 2.2)

    header_color = "#2c3e50"
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
            continue
        exp_idx = row - 1
        _, hl_color = highlights[exp_idx] if exp_idx < len(highlights) else (None, None)
        if col == 0:
            cell.set_facecolor("#f5f5f5")
        elif col == 1:
            # Destaca coluna do modelo com cor de highlight quando aplicável
            cell.set_facecolor(hl_color if hl_color else "#f5f5f5")
            if hl_color:
                cell.set_text_props(fontweight="bold")
        elif col == 2:
            cell.set_facecolor(COLORS_CLASSES["A"])
        elif col == 3:
            cell.set_facecolor(COLORS_CLASSES["B"])
        elif col == 4:
            cell.set_facecolor(COLORS_CLASSES["C"])
        elif col == 5:
            cell.set_facecolor(COLORS_CLASSES["D"])

    ax_table.set_title("A/B/C/D Distribution by Strategy Group", fontsize=12, fontweight="bold", pad=12)

    # ===== Direita: Barras agrupadas B/C/D =====
    x = np.arange(len(labels_display))
    width = 0.22

    # Banda de fundo para modelos destacados
    for i, (badge, hl_color) in enumerate(highlights):
        if hl_color:
            ax_chart.axvspan(i - 0.45, i + 0.45, color=hl_color, alpha=0.35, zorder=0)

    ax_chart.bar(x - width, class_b, width, label="B — close (~1 feature)",
                 color=COLORS_CLASSES["B"], zorder=2)
    ax_chart.bar(x,          class_c, width, label="C — distant (2–3 features)",
                 color=COLORS_CLASSES["C"], zorder=2)
    ax_chart.bar(x + width,  class_d, width, label="D — catastrophic (4+ features)",
                 color=COLORS_CLASSES["D"], zorder=2)

    max_val = max(max(class_b), max(class_c), max(class_d))
    label_offset = max_val * 0.04
    for i, (b, c, d) in enumerate(zip(class_b, class_c, class_d)):
        ax_chart.text(i - width, b + label_offset, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
        ax_chart.text(i,          c + label_offset, f"{c:.2f}", ha="center", va="bottom", fontsize=9)
        ax_chart.text(i + width,  d + label_offset, f"{d:.2f}", ha="center", va="bottom", fontsize=9)

    ax_chart.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax_chart.set_ylabel("B+C+D Errors (% of words)", fontsize=11, fontweight="bold")
    ax_chart.set_title("Errors by Class — Strategy Grouped", fontsize=12, fontweight="bold")
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(labels_display, fontsize=10)
    ax_chart.legend(fontsize=9, loc="upper right")
    ax_chart.grid(axis="y", alpha=0.3, zorder=1)
    ax_chart.set_ylim([0, max_val * 1.45])

    fig.suptitle("Phonological Class Distribution — Core Strategies + Ablation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/class_distribution_top5.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Saved: results/class_distribution_top5.png")
    plt.close()


def main():
    safe_print("\n" + "="*70)
    safe_print("GENERATING COMPARATIVE VISUALIZATIONS (FG2P vs LatPhon)")
    safe_print("="*70 + "\n")

    try:
        extractor, all_metrics, _all_metadata = load_data()
        safe_print(f"[OK] Loaded {len(all_metrics)} experiments with metrics\n")

        plot_baseline_comparison(all_metrics)
        plot_class_distribution_top5(all_metrics)

        safe_print("\n" + "="*70)
        safe_print("[OK] ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
        safe_print("="*70)
        safe_print("\nFiles generated in results/:")
        safe_print("  1. baseline_comparison.png")
        safe_print("  2. class_distribution_top5.png")
        safe_print("")

    except Exception as e:
        safe_print(f"\n[ERRO]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
