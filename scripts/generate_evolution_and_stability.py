#!/usr/bin/env python3
"""
Generate evolution, DA Loss gain, and stability visualizations.

Shows:
1. Timeline evolution: Exp0_baseline -> Exp1 -> Exp5 -> Exp9 -> Exp104b
2. Exp0_legacy puzzle: why 0.38% if baseline is 0.59%?
3. DA Loss gain: isolated comparison (same capacity, only loss changes)

Output:
    - evolution_per_wer.png: PER/WER evolution timeline
    - exp0_legacy_mystery.png: exp0_baseline vs exp0_legacy comparison
    - da_loss_gain.png: isolated DA Loss effect (Exp1 vs Exp7_0.20 vs Exp9)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
from extract_visualization_data import ExperimentDataExtractor

FIGURE_DPI = 300
COLORS = {
    "baseline": "#e74c3c",      # vermelho (baseline, pior)
    "intermediate": "#f39c12",   # laranja (intermediária)
    "da_loss": "#3498db",        # azul (com DA Loss)
    "sota": "#2ecc71",           # verde (SOTA)
    "legacy": "#9b59b6",         # roxo (enigmático)
}


def safe_print(msg):
    """Print com tratamento de encoding."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('utf-8', errors='replace').decode('utf-8'))


def load_data():
    """Load all metrics."""
    extractor = ExperimentDataExtractor(".")
    return extractor.load_all_metrics()


def _select_sota_per_exp(all_metrics):
    """Seleciona referência PER para o estágio final da timeline."""
    preferred = [
        "exp104d_structural_tokens_correct",
        "exp104b_intermediate_sep_da_custom_dist_fixed",
    ]
    for name in preferred:
        if name in all_metrics:
            return name

    valid = {k: v for k, v in all_metrics.items()
             if "legacy" not in k and "107" not in k}
    if not valid:
        return None
    return min(valid.items(), key=lambda x: x[1].per)[0]


def plot_evolution_timeline(all_metrics):
    """
    Gráfico 1: Timeline de evolução PER/WER
    Mostra caminho: Exp0_baseline → Exp1 → Exp5 → Exp9 → Exp104b
    """
    safe_print("[1/3] Plotando timeline evolutiva...")

    final_exp = _select_sota_per_exp(all_metrics)
    if final_exp is None:
        final_exp = "exp9_intermediate_distance_aware"
    final_label = "E104d SOTA\n(+sep+dist)" if final_exp == "exp104d_structural_tokens_correct" else "E104b SOTA\n(+sep+dist)"

    # Caminho de evolução principal (mostrando progresso)
    evolution_path = [
        ("exp0_baseline_70split", "E0 Baseline\n(70% split)", COLORS["baseline"]),
        ("exp1_baseline_60split", "E1 Baseline\n(60% split)", COLORS["baseline"]),
        ("exp5_intermediate_60split", "E5 Intermed.\n(+capacity)", COLORS["intermediate"]),
        ("exp9_intermediate_distance_aware", "E9 DA Loss\n(lambda=0.20)", COLORS["da_loss"]),
        (final_exp, final_label, COLORS["sota"]),
    ]

    x_pos = np.arange(len(evolution_path))
    labels = [e[1] for e in evolution_path]
    pers = []
    wers = []
    colors_seq = [e[2] for e in evolution_path]

    for exp_name, _, _ in evolution_path:
        if exp_name in all_metrics:
            m = all_metrics[exp_name]
            pers.append(m.per)
            wers.append(m.wer)
        else:
            pers.append(None)
            wers.append(None)

    # Figura com 2 Y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=FIGURE_DPI)

    # PER (left axis)
    ax1.set_xlabel("Evolution Stage", fontsize=12, fontweight="bold")
    ax1.set_ylabel("PER (%)", fontsize=12, fontweight="bold", color=COLORS["baseline"])
    ax1.tick_params(axis="y", labelcolor=COLORS["baseline"])

    line_per = ax1.plot(x_pos, pers, marker="o", linewidth=2.5, markersize=8,
                        color=COLORS["baseline"], label="PER", zorder=3)
    ax1.set_ylim([0.3, 0.7])

    # WER (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("WER (%)", fontsize=12, fontweight="bold", color=COLORS["sota"])
    ax2.tick_params(axis="y", labelcolor=COLORS["sota"])

    line_wer = ax2.plot(x_pos, wers, marker="s", linewidth=2.5, markersize=8,
                        color=COLORS["sota"], label="WER", zorder=3)
    ax2.set_ylim([4.5, 5.8])

    # Anotações de progresso
    for i, (per, wer) in enumerate(zip(pers, wers)):
        if i == 0:
            ax1.text(i, per + 0.03, f"{per:.2f}%", ha="center", fontsize=9, fontweight="bold")
        elif i == len(pers) - 1:
            ax1.text(i, per - 0.05, f"{per:.2f}%", ha="center", fontsize=9, fontweight="bold", color=COLORS["sota"])
        else:
            ax1.text(i, per + 0.03, f"{per:.2f}%", ha="center", fontsize=8)

    # Cores nas barras de fundo
    for i, color in enumerate(colors_seq):
        ax1.axvspan(i - 0.4, i + 0.4, alpha=0.1, color=color, zorder=0)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.grid(axis="y", alpha=0.3, zorder=1)

    # Title
    title_text = (
        "Evolution: Baseline -> Intermediate -> DA Loss -> SOTA\n"
        f"Showing PER reduction ({pers[0]:.2f}% -> {pers[-1]:.2f}%) "
        f"and WER behavior ({wers[0]:.2f}% -> {wers[-1]:.2f}%)"
    )
    ax1.set_title(title_text, fontsize=13, fontweight="bold", pad=15)

    # Legend
    lines = line_per + line_wer
    labels_legend = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels_legend, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig("results/evolution_per_wer.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/evolution_per_wer.png")
    plt.close()


def plot_exp0_legacy_mystery(all_metrics):
    """
    Gráfico 2: O enigma do Exp0_legacy
    Compara exp0_baseline_70split (0.59%) vs exp0_legacy_simple (0.38%)
    Documenta a investigacao sobre split estratificado
    """
    safe_print("[2/3] Plotando investigacao exp0_legacy...")

    exp0_baseline = all_metrics.get("exp0_baseline_70split")
    exp0_legacy = all_metrics.get("exp0_legacy_simple")

    if not exp0_baseline or not exp0_legacy:
        safe_print("   [SKIP] exp0_baseline ou exp0_legacy nao encontrados")
        return

    fig, ax = plt.subplots(figsize=(11, 6), dpi=FIGURE_DPI)

    models = ["Exp0 Baseline\n(70% split,\nstratified=True)",
              "Exp0 Legacy\n(70% split,\nstratified=False?)"]
    pers = [exp0_baseline.per, exp0_legacy.per]
    wers = [exp0_baseline.wer, exp0_legacy.wer]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, pers, width, label="PER", color=COLORS["baseline"], alpha=0.8)
    ax.bar(x + width/2, wers, width, label="WER", color=COLORS["sota"], alpha=0.8)

    # Anotacoes
    for i, (per, wer) in enumerate(zip(pers, wers)):
        ax.text(i - width/2, per + 0.05, f"{per:.2f}%", ha="center", fontsize=10, fontweight="bold")
        ax.text(i + width/2, wer + 0.15, f"{wer:.2f}%", ha="center", fontsize=10, fontweight="bold")

    # Delta
    delta_per = exp0_baseline.per - exp0_legacy.per
    ax.text(0.5, 5.8, f"Delta PER: {delta_per:+.2f}pp\n(Baseline {exp0_baseline.per:.2f}% → Legacy {exp0_legacy.per:.2f}%)",
            ha="center", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.4))

    ax.set_ylabel("Error (%)", fontsize=12, fontweight="bold")
    ax.set_title("Investigation: exp0_baseline vs exp0_legacy\nWhat explains the 0.21pp PER delta?",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 6])
    ax.grid(axis="y", alpha=0.3)

    # Anotacao explicativa
    explanation = (
        "QUESTION: did legacy run with +3.5x gradient updates?\n"
        "- batch_size=36 (legacy) vs batch_size=96 (baseline)\n"
        "- epochs=120 without early stop vs with early stop (patience=10)\n"
        "-> Training regime != split bias"
    )
    ax.text(0.02, 0.05, explanation, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))

    plt.tight_layout()
    plt.savefig("results/exp0_legacy_mystery.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/exp0_legacy_mystery.png")
    plt.close()


def plot_da_loss_gain(all_metrics):
    """
    Gráfico 3: Isolamento do efeito DA Loss
    Compara: Exp1 (baseline) vs Exp7_0.20 (DA loss) vs Exp9 (DA loss + capacidade maior)
    Mesma split (60%), mostrando ganho APENAS da loss
    """
    safe_print("[3/3] Plotando isolamento do ganho DA Loss...")

    exp1 = all_metrics.get("exp1_baseline_60split")
    exp7_020 = all_metrics.get("exp7_lambda_mid_candidate_0.20")
    exp9 = all_metrics.get("exp9_intermediate_distance_aware")

    if not (exp1 and exp7_020 and exp9):
        safe_print("   [SKIP] Algum experimento nao encontrado")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=FIGURE_DPI)

    # Subplot 1: Isolamento puro (Exp1 vs Exp7_0.20, mesma arquitetura)
    models1 = ["Exp1\nBaseline (CE)", "Exp7_0.20\nDA Loss (lambda=0.20)"]
    pers1 = [exp1.per, exp7_020.per]
    wers1 = [exp1.wer, exp7_020.wer]
    x1 = np.arange(len(models1))

    ax1.bar(x1 - 0.2, pers1, 0.4, label="PER", color=COLORS["baseline"], alpha=0.8)
    ax1.bar(x1 + 0.2, wers1, 0.4, label="WER", color=COLORS["da_loss"], alpha=0.8)

    for i, (per, wer) in enumerate(zip(pers1, wers1)):
        ax1.text(i - 0.2, per + 0.02, f"{per:.2f}%", ha="center", fontsize=9, fontweight="bold")
        ax1.text(i + 0.2, wer + 0.08, f"{wer:.2f}%", ha="center", fontsize=9, fontweight="bold")

    gain_per1 = exp1.per - exp7_020.per
    ax1.text(0.5, 5.5, f"DA Loss Gain (isolado):\nPER: {gain_per1:+.2f}pp\nWER: {exp1.wer - exp7_020.wer:+.2f}pp",
             ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.4))

    ax1.set_ylabel("Error (%)", fontsize=11, fontweight="bold")
    ax1.set_title("1. Pure Isolation: Same capacity\n(256h, 4.3M params, 60% split)",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(models1, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_ylim([0, 6])
    ax1.grid(axis="y", alpha=0.3)

    # Subplot 2: Com aumento de capacidade (Exp1 vs Exp9, diferentes capacidades)
    models2 = ["Exp1\nBaseline\n(256h, CE)", "Exp9\nDA Loss + Capacity\n(384h, lambda=0.20)"]
    pers2 = [exp1.per, exp9.per]
    wers2 = [exp1.wer, exp9.wer]
    x2 = np.arange(len(models2))

    ax2.bar(x2 - 0.2, pers2, 0.4, label="PER", color=COLORS["baseline"], alpha=0.8)
    ax2.bar(x2 + 0.2, wers2, 0.4, label="WER", color=COLORS["sota"], alpha=0.8)

    for i, (per, wer) in enumerate(zip(pers2, wers2)):
        ax2.text(i - 0.2, per + 0.02, f"{per:.2f}%", ha="center", fontsize=9, fontweight="bold")
        ax2.text(i + 0.2, wer + 0.08, f"{wer:.2f}%", ha="center", fontsize=9, fontweight="bold")

    gain_per2 = exp1.per - exp9.per
    ax2.text(0.5, 5.5, f"Ganho combinado:\nPER: {gain_per2:+.2f}pp (−{gain_per2/exp1.per*100:.1f}%)\nWER: {exp1.wer - exp9.wer:+.2f}pp",
             ha="center", fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.4))

    ax2.set_ylabel("Error (%)", fontsize=11, fontweight="bold")
    ax2.set_title("2. With increased capacity\n(384h, 9.7M params, 60% split)",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(models2, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_ylim([0, 6])
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle("Gain Isolation: DA Loss vs Baseline (Cross-Entropy)",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig("results/da_loss_gain.png", dpi=FIGURE_DPI, bbox_inches="tight")
    safe_print("   [OK] Salvo: results/da_loss_gain.png")
    plt.close()


def main():
    safe_print("\n" + "="*70)
    safe_print("GERANDO VISUALIZACOES DE EVOLUCAO")
    safe_print("="*70 + "\n")

    try:
        all_metrics = load_data()
        safe_print(f"[OK] Carregados {len(all_metrics)} experimentos\n")

        plot_evolution_timeline(all_metrics)
        plot_exp0_legacy_mystery(all_metrics)
        plot_da_loss_gain(all_metrics)

        safe_print("\n" + "="*70)
        safe_print("[OK] VISUALIZACOES DE EVOLUCAO GERADAS")
        safe_print("="*70)
        safe_print("\nArquivos gerados:")
        safe_print("  1. evolution_per_wer.png           (timeline de progresso)")
        safe_print("  2. exp0_legacy_mystery.png         (investigacao split)")
        safe_print("  3. da_loss_gain.png                (efeito da loss isolado)")
        safe_print("")

    except Exception as e:
        safe_print(f"\n[ERRO]: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
