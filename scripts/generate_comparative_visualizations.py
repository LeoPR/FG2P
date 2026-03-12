#!/usr/bin/env python3
"""
Gerar visualizações comparativas: FG2P vs LatPhon + distribuição de classes A-D.

Saída:
  - class_distribution_all_experiments.png: Distribuição A/B/C/D para todos os experimentos
  - latphon_comparison.png: Comparação FG2P vs LatPhon (PER, WER, CI)
  - top_5_models_metrics.png: Top 5 modelos com métricas detalhadasver
  - class_distribution_top5.png: Distribuição A/B/C/D para top 5 modelos
"""

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

# LatPhon dados (de STATUS.md)
LATPHON_PER = 0.86
LATPHON_CI_LOW = 0.57
LATPHON_CI_HIGH = 1.22
LATPHON_TEST_SIZE = 500

# FG2P dados (de STATUS.md)
FG2P_PER = 0.49
FG2P_CI_LOW = 0.46
FG2P_CI_HIGH = 0.52
FG2P_TEST_SIZE = 28782


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


def plot_class_distribution_all_experiments(all_metrics):
    """
    Gráfico 1: Distribuição A/B/C/D para todos os experimentos.
    Usa dois subplots: escala completa (0-100%) e detalhe (0-5% para B/C/D).
    """
    safe_print("[1/4] Plotando distribuicao de classes para todos os experimentos...")

    experiments = sorted(all_metrics.keys())
    n_exp = len(experiments)

    # Extrair dados
    labels_exp = []
    class_a = []
    class_b = []
    class_c = []
    class_d = []

    for exp_name in experiments:
        metrics = all_metrics[exp_name]
        labels_exp.append(exp_name.replace("exp", "E"))
        class_a.append(metrics.class_a_pct)
        class_b.append(metrics.class_b_pct)
        class_c.append(metrics.class_c_pct)
        class_d.append(metrics.class_d_pct)

    # Figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), dpi=FIGURE_DPI)

    x = np.arange(len(labels_exp))
    width = 0.8

    # ===== Subplot 1: Escala completa (0-100%) =====
    p1 = ax1.bar(x, class_a, width, label="Classe A (Correto/Imperceptível)", color=COLORS_CLASSES["A"])
    p2 = ax1.bar(x, class_b, width, bottom=class_a, label="Classe B (Próximo)", color=COLORS_CLASSES["B"])
    p3 = ax1.bar(x, class_c, width, bottom=np.array(class_a) + np.array(class_b), label="Classe C (Distante)", color=COLORS_CLASSES["C"])
    p4 = ax1.bar(x, class_d, width, bottom=np.array(class_a) + np.array(class_b) + np.array(class_c), label="Classe D (Muito Distante)", color=COLORS_CLASSES["D"])

    ax1.set_xlabel("Experimento", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Distribuição (% de palavras)", fontsize=11, fontweight="bold")
    ax1.set_title("Escala Completa (0-100%)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_exp, rotation=45, ha="right", fontsize=8)
    ax1.set_ylim([0, 100])
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ===== Subplot 2: Zoom B/C/D (0-5%) =====
    p1b = ax2.bar(x, class_b, width, label="Classe B (Próximo)", color=COLORS_CLASSES["B"])
    p2b = ax2.bar(x, class_c, width, bottom=class_b, label="Classe C (Distante)", color=COLORS_CLASSES["C"])
    p3b = ax2.bar(x, class_d, width, bottom=np.array(class_b) + np.array(class_c), label="Classe D (Muito Distante)", color=COLORS_CLASSES["D"])

    ax2.set_xlabel("Experimento", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Distribuição B+C+D (% de palavras)", fontsize=11, fontweight="bold")
    ax2.set_title("Detalhe: Erros Apenas (0-5%)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_exp, rotation=45, ha="right", fontsize=8)
    ax2.set_ylim([0, 5])
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Distribuição de Classes Fonológicas por Experimento (Todos os 24)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("results/class_distribution_all_experiments.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/class_distribution_all_experiments.png (2 subplots)")
    plt.close()


def plot_latphon_comparison():
    """
    Gráfico 2: Comparação FG2P vs LatPhon.
    PER com barras de CI.
    """
    safe_print("[2/4] Plotando comparacao com LatPhon...")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=FIGURE_DPI)

    models = ["FG2P\n(28.8k words)", "LatPhon\n(~500 words)"]
    pers = [FG2P_PER, LATPHON_PER]
    ci_errors = [
        [FG2P_PER - FG2P_CI_LOW, FG2P_CI_HIGH - FG2P_PER],
        [LATPHON_PER - LATPHON_CI_LOW, LATPHON_CI_HIGH - LATPHON_PER],
    ]

    colors = ["#2ecc71", "#e74c3c"]
    x_pos = np.arange(len(models))

    # Barras com CI
    bars = ax.bar(x_pos, pers, color=colors, alpha=0.7, width=0.5, edgecolor="black", linewidth=2)

    # Error bars (95% CI)
    ci_errors_array = np.array(ci_errors).T
    ax.errorbar(x_pos, pers, yerr=ci_errors_array, fmt="none", color="black", capsize=10, capthick=2, linewidth=2)

    # Anotações
    for i, (per, model) in enumerate(zip(pers, models)):
        if i == 0:
            ci_text = f"[{FG2P_CI_LOW:.2f}%, {FG2P_CI_HIGH:.2f}%]"
        else:
            ci_text = f"[{LATPHON_CI_LOW:.2f}%, {LATPHON_CI_HIGH:.2f}%]"

        ax.text(i, per + 0.15, f"{per:.2f}%\nIC={ci_text}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("PER (%)", fontsize=12, fontweight="bold")
    ax.set_title("FG2P vs LatPhon: Phoneme Error Rate\n(Wilson 95% Confidence Interval)", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim([0, 1.5])
    ax.grid(axis="y", alpha=0.3)

    # Anotação: "Non-overlapping CIs"
    ax.text(0.5, 1.35, "ICs não se sobrepõem\n→ Diferença estatisticamente significativa",
            ha="center", fontsize=10, style="italic", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig("results/latphon_comparison.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/latphon_comparison.png")
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


def plot_class_distribution_top5(all_metrics):
    """
    Gráfico 4: Distribuição A/B/C/D para top 5 modelos.
    Usa 2 subplots: escala completa e zoom em B/C/D.
    """
    safe_print("[4/4] Plotando distribuicao de classes para top 5...")

    # Top 5 por PER
    sorted_by_per = sorted(all_metrics.items(), key=lambda x: x[1].per)
    top_5 = dict(sorted_by_per[:5])

    experiments = list(top_5.keys())
    labels_exp = [e.replace("exp", "E") for e in experiments]

    class_a = [all_metrics[e].class_a_pct for e in experiments]
    class_b = [all_metrics[e].class_b_pct for e in experiments]
    class_c = [all_metrics[e].class_c_pct for e in experiments]
    class_d = [all_metrics[e].class_d_pct for e in experiments]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=FIGURE_DPI)

    x = np.arange(len(labels_exp))
    width = 0.2

    # ===== Subplot 1: Escala completa (0-100%) =====
    bars_a1 = ax1.bar(x - 1.5*width, class_a, width, label="Classe A", color=COLORS_CLASSES["A"])
    bars_b1 = ax1.bar(x - 0.5*width, class_b, width, label="Classe B", color=COLORS_CLASSES["B"])
    bars_c1 = ax1.bar(x + 0.5*width, class_c, width, label="Classe C", color=COLORS_CLASSES["C"])
    bars_d1 = ax1.bar(x + 1.5*width, class_d, width, label="Classe D", color=COLORS_CLASSES["D"])

    ax1.set_xlabel("Modelo", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Distribuição (% de palavras)", fontsize=11, fontweight="bold")
    ax1.set_title("Escala Completa (0-100%)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_exp, fontsize=10)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim([0, 100])

    # Valores em A
    for i, v in enumerate(class_a):
        ax1.text(i - 1.5*width, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)

    # ===== Subplot 2: Zoom em B/C/D (0-5%) =====
    bars_b2 = ax2.bar(x - width, class_b, width, label="Classe B", color=COLORS_CLASSES["B"])
    bars_c2 = ax2.bar(x, class_c, width, label="Classe C", color=COLORS_CLASSES["C"])
    bars_d2 = ax2.bar(x + width, class_d, width, label="Classe D", color=COLORS_CLASSES["D"])

    ax2.set_xlabel("Modelo", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Distribuição B+C+D (% de palavras)", fontsize=11, fontweight="bold")
    ax2.set_title("Detalhe: Erros Apenas (0-5%)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_exp, fontsize=10)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim([0, 5])

    # Valores em B/C/D
    for i, (b, c, d) in enumerate(zip(class_b, class_c, class_d)):
        ax2.text(i - width, b + 0.1, f"{b:.2f}", ha="center", fontsize=7)
        ax2.text(i, c + 0.1, f"{c:.2f}", ha="center", fontsize=7)
        ax2.text(i + width, d + 0.1, f"{d:.2f}", ha="center", fontsize=7)

    fig.suptitle("Distribuição de Classes Fonológicas — Top 5 Modelos por PER", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig("results/class_distribution_top5.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/class_distribution_top5.png (2 subplots)")
    plt.close()


def main():
    safe_print("\n" + "="*70)
    safe_print("GERANDO VISUALIZACOES COMPARATIVAS (FG2P vs LatPhon)")
    safe_print("="*70 + "\n")

    try:
        extractor, all_metrics, all_metadata = load_data()
        safe_print(f"[OK] Carregados {len(all_metrics)} experimentos com metricas\n")

        plot_class_distribution_all_experiments(all_metrics)
        plot_latphon_comparison()
        plot_top_5_models(all_metrics, all_metadata)
        plot_class_distribution_top5(all_metrics)

        safe_print("\n" + "="*70)
        safe_print("[OK] TODAS AS VISUALIZACOES GERADAS COM SUCESSO")
        safe_print("="*70)
        safe_print("\nArquivos gerados em results/:")
        safe_print("  1. class_distribution_all_experiments.png")
        safe_print("  2. latphon_comparison.png")
        safe_print("  3. top_5_models_metrics.png")
        safe_print("  4. class_distribution_top5.png")
        safe_print("")

    except Exception as e:
        safe_print(f"\n[ERRO]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
