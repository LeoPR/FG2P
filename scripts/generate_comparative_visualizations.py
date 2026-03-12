#!/usr/bin/env python3
"""
Gerar visualizações comparativas: FG2P vs LatPhon + distribuição de classes A-D.

Saída:
  - class_distribution_all_experiments.png: Distribuição A/B/C/D para todos os experimentos
  - baseline_comparison.png: FG2P vs LatPhon, WFST, ByT5-Small (PER com CI)
  - top_5_models_metrics.png: Top 5 modelos com métricas detalhadasver
  - class_distribution_top5.png: Distribuição A/B/C/D para top 5 modelos
"""

import re
import sys
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
        "per": 9.10,
        "ci_low": None,
        "ci_high": None,
        "color": "#e74c3c",   # vermelho
        "note": "9.10% (sem CI)",
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


def plot_baseline_comparison():
    """
    Gráfico 2: FG2P vs todos os baselines — LatPhon, WFST, ByT5-Small.
    PER com barras de erro (95% CI onde disponível).
    """
    safe_print("[2/3] Plotando comparacao com baselines...")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=FIGURE_DPI)

    x_pos = np.arange(len(BASELINES))

    for i, b in enumerate(BASELINES):
        ax.bar(i, b["per"], color=b["color"], alpha=0.75, width=0.55,
               edgecolor="black", linewidth=1.5, zorder=2)

        if b["ci_low"] is not None:
            yerr = [[b["per"] - b["ci_low"]], [b["ci_high"] - b["per"]]]
            ax.errorbar(i, b["per"], yerr=yerr, fmt="none",
                        color="black", capsize=9, capthick=2, linewidth=2, zorder=3)

        offset = b["per"] * 0.06 + 0.15
        ax.text(i, b["per"] + offset, b["note"],
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Destaque: IC não sobrepostos entre FG2P e LatPhon
    ax.annotate(
        "ICs não se sobrepõem\n(FG2P upper 0.51% < LatPhon lower 0.56%)",
        xy=(0.5, 0.86), xycoords=("data", "data"),
        xytext=(1.5, 2.2), textcoords="data",
        fontsize=9, style="italic", ha="center",
        arrowprops=dict(arrowstyle="-", color="gray", lw=1),
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
    )

    ax.set_ylabel("PER (%)", fontsize=12, fontweight="bold")
    ax.set_title("Comparação de Baselines — Phoneme Error Rate (PER)\n"
                 "FG2P vs LatPhon 2025, WFST/Phonetisaurus, ByT5-Small",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([b["label"] for b in BASELINES], fontsize=11)
    ax.set_ylim([0, 11.5])
    ax.grid(axis="y", alpha=0.3, zorder=1)

    plt.tight_layout()
    plt.savefig("results/baseline_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/baseline_comparison.png")
    plt.close()


def plot_top_5_models(all_metrics, all_metadata):
    """
    Gráfico 3: Top 5 modelos (por PER) com métricas detalhadasver.
    Tabela com: PER, WER, Accuracy, Params, Classe A/B/C/D.
    """
    safe_print("[3/4] Plotando top 5 modelos...")

    # Obter top 5 por PER
    sorted_by_per = sorted(all_metrics.items(), key=lambda x: x[1].per)
    top_5 = dict(sorted_by_per[:5])

    # Preparar dados para tabela
    rows = []
    for exp_name, metrics in top_5.items():
        exp_short = exp_name.replace("exp", "E")
        metadata = all_metadata.get(exp_name, None)
        params = f"{metadata.total_params_m:.1f}M" if metadata else "N/A"

        row = [
            exp_short,
            f"{metrics.per:.2f}%",
            f"{metrics.wer:.2f}%",
            f"{metrics.accuracy:.2f}%",
            params,
            f"{metrics.class_a_pct:.1f}%",
            f"{metrics.class_b_pct:.1f}%",
            f"{metrics.class_c_pct:.1f}%",
            f"{metrics.class_d_pct:.1f}%",
        ]
        rows.append(row)

    # Figura
    fig, ax = plt.subplots(figsize=(14, 5), dpi=FIGURE_DPI)
    ax.axis("off")

    columns = ["Exp", "PER", "WER", "Acc", "Params", "Cl.A", "Cl.B", "Cl.C", "Cl.D"]

    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", loc="center",
                     colWidths=[0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10, 0.10])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Estilo header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Estilo dados
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#ecf0f1")
            else:
                table[(i, j)].set_facecolor("#ffffff")

    plt.title("Top 5 Modelos por PER (com Distribuição de Classes)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("results/top_5_models_metrics.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/top_5_models_metrics.png")
    plt.close()


def _short_label(exp_name):
    """Extrai label curto: exp104b_... -> Exp104b, exp9_... -> Exp9"""
    m = re.match(r"exp(\d+\w?)_", exp_name)
    return f"Exp{m.group(1)}" if m else exp_name.replace("exp", "Exp")


# Experimentos com vantagem artificial: split viciado (legacy/exp0) ou 95% de treino (exp107)
_BIASED_PATTERNS = ("legacy", "107")

# Sempre incluídos: baseline CE (exp1) + melhor WER (exp9)
_FORCED_PREFIXES = ("exp1_", "exp9_")

# Modelos com destaque: prefixo -> (badge no label, cor de fundo na tabela)
_HIGHLIGHTS = {
    "exp104b_": ("★ Best PER", "#c8e6c9"),   # verde claro
    "exp9_":    ("★ Best WER", "#bbdefb"),   # azul claro
}


def _highlight_info(exp_name):
    for prefix, info in _HIGHLIGHTS.items():
        if exp_name.startswith(prefix):
            return info
    return None, None


def plot_class_distribution_top5(all_metrics):
    """
    Gráfico: Top 5 modelos validados por PER + Exp1 e Exp9 forçados.
    Exp104b (★ Best PER) e Exp9 (★ Best WER) são destacados na tabela e no gráfico.
    Esquerda: tabela A/B/C/D. Direita: barras B/C/D com escala ajustada.
    """
    safe_print("[3/3] Plotando distribuicao de classes para top 5 validados...")

    valid = {k: v for k, v in all_metrics.items()
             if not any(p in k for p in _BIASED_PATTERNS)}

    top_5_keys = {e for e, _ in sorted(valid.items(), key=lambda x: x[1].per)[:5]}
    forced = {k for k in valid if any(k.startswith(p) for p in _FORCED_PREFIXES)}
    selected_keys = top_5_keys | forced

    experiments = [e for e, _ in sorted(valid.items(), key=lambda x: x[1].per)
                   if e in selected_keys]

    labels_short = [_short_label(e) for e in experiments]
    highlights = [_highlight_info(e) for e in experiments]  # list of (badge, color) or (None, None)

    # Labels com badge em segunda linha para modelos destacados
    labels_display = [
        f"{lbl}\n{badge}" if badge else lbl
        for lbl, (badge, _) in zip(labels_short, highlights)
    ]

    class_a = [all_metrics[e].class_a_pct for e in experiments]
    class_b = [all_metrics[e].class_b_pct for e in experiments]
    class_c = [all_metrics[e].class_c_pct for e in experiments]
    class_d = [all_metrics[e].class_d_pct for e in experiments]

    fig, (ax_table, ax_chart) = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI,
                                              gridspec_kw={"width_ratios": [1, 1.6]})

    # ===== Esquerda: Tabela A/B/C/D =====
    ax_table.axis("off")
    col_labels = ["Modelo", "A (%)", "B (%)", "C (%)", "D (%)"]
    table_data = [
        [lbl, f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{d:.2f}"]
        for lbl, a, b, c, d in zip(labels_short, class_a, class_b, class_c, class_d)
    ]
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
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
            # Destaca linha do modelo com cor de highlight ou neutro
            cell.set_facecolor(hl_color if hl_color else "#f5f5f5")
            if hl_color:
                cell.set_text_props(fontweight="bold")
        elif col == 1:
            cell.set_facecolor(COLORS_CLASSES["A"])
        elif col == 2:
            cell.set_facecolor(COLORS_CLASSES["B"])
        elif col == 3:
            cell.set_facecolor(COLORS_CLASSES["C"])
        elif col == 4:
            cell.set_facecolor(COLORS_CLASSES["D"])

    ax_table.set_title("Distribuição completa A/B/C/D", fontsize=12, fontweight="bold", pad=12)

    # ===== Direita: Barras agrupadas B/C/D =====
    x = np.arange(len(labels_display))
    width = 0.22

    # Banda de fundo para modelos destacados
    for i, (badge, hl_color) in enumerate(highlights):
        if hl_color:
            ax_chart.axvspan(i - 0.45, i + 0.45, color=hl_color, alpha=0.35, zorder=0)

    ax_chart.bar(x - width, class_b, width, label="B — próximo (~1 feat)",
                 color=COLORS_CLASSES["B"], zorder=2)
    ax_chart.bar(x,          class_c, width, label="C — distante (2–3 feat)",
                 color=COLORS_CLASSES["C"], zorder=2)
    ax_chart.bar(x + width,  class_d, width, label="D — catastrófico (4+ feat)",
                 color=COLORS_CLASSES["D"], zorder=2)

    max_val = max(max(class_b), max(class_c), max(class_d))
    label_offset = max_val * 0.04
    for i, (b, c, d) in enumerate(zip(class_b, class_c, class_d)):
        ax_chart.text(i - width, b + label_offset, f"{b:.2f}", ha="center", va="bottom", fontsize=9)
        ax_chart.text(i,          c + label_offset, f"{c:.2f}", ha="center", va="bottom", fontsize=9)
        ax_chart.text(i + width,  d + label_offset, f"{d:.2f}", ha="center", va="bottom", fontsize=9)

    ax_chart.set_xlabel("Modelo", fontsize=11, fontweight="bold")
    ax_chart.set_ylabel("Erros B+C+D (% de palavras)", fontsize=11, fontweight="bold")
    ax_chart.set_title("Erros por Classe — Modelos Validados por PER", fontsize=12, fontweight="bold")
    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels(labels_display, fontsize=10)
    ax_chart.legend(fontsize=9, loc="upper right")
    ax_chart.grid(axis="y", alpha=0.3, zorder=1)
    ax_chart.set_ylim([0, max_val * 1.45])

    fig.suptitle("Distribuição de Classes Fonológicas — Modelos Validados por PER",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/class_distribution_top5.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/class_distribution_top5.png")
    plt.close()


def main():
    safe_print("\n" + "="*70)
    safe_print("GERANDO VISUALIZACOES COMPARATIVAS (FG2P vs LatPhon)")
    safe_print("="*70 + "\n")

    try:
        extractor, all_metrics, all_metadata = load_data()
        safe_print(f"[OK] Carregados {len(all_metrics)} experimentos com metricas\n")

        plot_baseline_comparison()
        plot_top_5_models(all_metrics, all_metadata)
        plot_class_distribution_top5(all_metrics)

        safe_print("\n" + "="*70)
        safe_print("[OK] TODAS AS VISUALIZACOES GERADAS COM SUCESSO")
        safe_print("="*70)
        safe_print("\nArquivos gerados em results/:")
        safe_print("  1. baseline_comparison.png")
        safe_print("  2. top_5_models_metrics.png")
        safe_print("  3. class_distribution_top5.png")
        safe_print("")

    except Exception as e:
        safe_print(f"\n[ERRO]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
