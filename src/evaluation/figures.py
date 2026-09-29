"""Figuras de publicação para os Capítulos 4 e 5 do TCC.

O destino é um documento impresso, não uma página web: as figuras são estáticas,
em modo claro, e toda identidade de série é redundante — cor **mais** marcador,
estilo de linha ou hachura — para que continuem legíveis em impressão
monocromática e para leitores com deficiência de visão de cores.

A paleta é a instância de referência do método de dataviz, validada com
`validate_palette.js` nas combinações efetivamente usadas aqui:

- 3 slots (`#2a78d6`, `#eb6834`, `#1baf7a`) em `--pairs all`: pior par CVD ΔE 9.2,
  visão normal 24.0 — usada nas curvas ROC, onde as linhas se sobrepõem.
- 4 slots (os três acima + `#eda100`) no pairlist adjacente: pior par CVD ΔE 9.1,
  visão normal 22.9 — usada nas varreduras OFAT.
- 2 slots (`#2a78d6`, `#eb6834`): pior par CVD ΔE 24.7 — usada nas comparações
  binárias (treino vs. validação, benigno vs. ataque, λ=0 vs. DEAS).

`#1baf7a` e `#eda100` ficam abaixo de 3:1 contra a superfície clara, então vale a
regra de alívio: toda série recebe rótulo direto visível e existe a tabela
equivalente no relatório Markdown.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Tokens da paleta (instância de referência, modo claro)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Ordem categórica fixa. Nunca ciclar nem gerar um nono tom.
SERIES: Tuple[str, ...] = (
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
)

# Rampa sequencial de uma única matiz (azul), clara -> escura.
SEQUENTIAL_BLUE: Tuple[str, ...] = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
)

# Codificação secundária: marcador e estilo de linha por slot, para que a
# identidade nunca dependa só da cor (impressão em tons de cinza, CVD).
MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES: Tuple[Any, ...] = ("-", (0, (5, 2)), (0, (1, 1.5)), (0, (7, 2, 1, 2)))

# Marcador de família estatisticamente rara nas figuras. Precisa ser ASCII: o sinal
# de aviso U+26A0, usado nos relatórios Markdown, não existe em Segoe UI nem em
# DejaVu Sans e o matplotlib o desenharia como um retângulo vazio.
RARE_MARKER = " *"

FIGURE_DPI = 300


def _pyplot():
    """Importa matplotlib com backend não interativo e tipografia de publicação."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",  # hairline sólido; tracejado lê como projeção
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.autolayout": False,
        }
    )
    return plt


def _finish(fig, axes, path: str) -> str:
    """Aplica a cromagem recessiva comum e grava a figura."""
    for ax in np.atleast_1d(axes).ravel():
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.title.set_color(INK_PRIMARY)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt = _pyplot()
    plt.close(fig)
    return os.path.abspath(path)


def _sequential_step(fraction: float) -> str:
    """Amostra a rampa azul de uma matiz em [0, 1]."""
    fraction = float(np.clip(fraction, 0.0, 1.0))
    index = int(round(fraction * (len(SEQUENTIAL_BLUE) - 1)))
    return SEQUENTIAL_BLUE[index]


# ---------------------------------------------------------------------------
# Capítulo 4 — Módulo 2: convergência do FC-DAE
# ---------------------------------------------------------------------------
def plot_ae_convergence(history: Sequence[Dict[str, float]], dataset: str, path: str) -> str:
    """Curva de perda de treino e validação por época, com a melhor época marcada.

    Duas séries de mesma unidade (MSE) num único eixo — nunca eixo duplo.
    """
    plt = _pyplot()
    epochs = [int(h["epoch"]) for h in history]
    train = [float(h["train_loss"]) for h in history]
    val = [float(h["val_loss"]) for h in history]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(epochs, train, color=SERIES[0], lw=2, label="Treino (benigno)")
    ax.plot(epochs, val, color=SERIES[1], lw=2, ls=LINESTYLES[1], label="Validação (benigno)")

    best_idx = int(np.argmin(val))
    ax.plot(
        epochs[best_idx], val[best_idx],
        marker="o", ms=8, color=SERIES[1],
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5,  # anel de 2px na superfície
    )
    # Rótulo direto seletivo: só o extremo que importa.
    ax.annotate(
        f"melhor época {epochs[best_idx]}\nMSE {val[best_idx]:.6f}",
        xy=(epochs[best_idx], val[best_idx]),
        xytext=(8, 14), textcoords="offset points",
        fontsize=8, color=INK_SECONDARY,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Época")
    ax.set_ylabel("Erro de reconstrução (MSE, escala log)")
    ax.set_title(f"Convergência do FC-DAE — {dataset}")
    ax.legend(loc="upper right")
    return _finish(fig, ax, path)


# ---------------------------------------------------------------------------
# Capítulo 4 — Módulo 3: reconstrução diferencial
# ---------------------------------------------------------------------------
def plot_mse_distribution(
    mse: np.ndarray,
    y_true: np.ndarray,
    dataset: str,
    path: str,
    tau_ae: Optional[float] = None,
    tau_label: str = "tau_AE",
) -> str:
    """Distribuição do erro de reconstrução, benigno vs. ataque, com o limiar."""
    plt = _pyplot()
    mse = np.asarray(mse, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)

    benign, attack = mse[y_true == 0], mse[y_true == 1]
    positive = mse[mse > 0]
    floor = np.percentile(positive, 0.1) if positive.size else 1e-9
    bins = np.logspace(np.log10(max(floor, 1e-12)), np.log10(max(mse.max(), floor * 10)), 70)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    # Histogramas preenchidos com transparência; densidade normaliza os tamanhos
    # muito diferentes das duas populações.
    ax.hist(benign, bins=bins, density=True, color=SERIES[0], alpha=0.55, label="Benigno")
    ax.hist(
        attack, bins=bins, density=True, color=SERIES[1], alpha=0.55,
        label="Ataque", hatch="//", edgecolor=SERIES[1], linewidth=0.0,
    )

    if tau_ae is not None:
        ax.axvline(tau_ae, color=INK_PRIMARY, lw=1.4, ls=LINESTYLES[1])
        ax.annotate(
            f"{tau_label} = {tau_ae:.5f}",
            xy=(tau_ae, ax.get_ylim()[1] * 0.92),
            xytext=(6, 0), textcoords="offset points",
            fontsize=8, color=INK_PRIMARY, rotation=90, va="top",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Erro de reconstrução por fluxo (MSE, escala log)")
    ax.set_ylabel("Densidade")
    ax.set_title(f"Reconstrução diferencial — {dataset}")
    ax.legend(loc="upper left")
    return _finish(fig, ax, path)


# ---------------------------------------------------------------------------
# Capítulo 5 — curvas ROC da cadeia de ablação
# ---------------------------------------------------------------------------
# Três séries coloridas (limite validado em --pairs all para linhas sobrepostas)
# contando a cadeia de ablação, mais o teto supervisionado como referência
# cinza — referência não é identidade, logo não consome um slot categórico.
ROC_ABLATION_CHAIN: Tuple[str, ...] = (
    "B1_iForest_Raw",
    "B3b_PreIF_DIF_Pure",
    "B6a_Hybrid_Linear",
)
ROC_REFERENCE = "B4_RandomForest_Supervised"

ROC_SHORT_LABELS: Dict[str, str] = {
    "B1_iForest_Raw": "B1 · iForest bruto",
    "B2_Autoencoder_Alone": "B2 · FC-DAE isolado",
    "B3_PreIF_iForest": "B3 · Pre-IF + iForest",
    "B3b_PreIF_DIF_Pure": "B3b · DIF puro (λ=0)",
    "B4_RandomForest_Supervised": "B4 · RF supervisionado",
    "B6a_Hybrid_Linear": "B6a · Fusão DIF+DEAS",
    "B6b_Hybrid_OR": "B6b · Regra OR",
}


def plot_roc_curves(
    y_true: np.ndarray,
    scores_by_model: Dict[str, np.ndarray],
    dataset: str,
    path: str,
    models: Sequence[str] = ROC_ABLATION_CHAIN,
    reference: Optional[str] = ROC_REFERENCE,
) -> str:
    """Curvas ROC da cadeia de ablação, com o teto supervisionado como referência."""
    from sklearn.metrics import auc, roc_curve

    plt = _pyplot()
    y_true = np.asarray(y_true, dtype=int)

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    # Classificador aleatório: contexto, não série.
    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=0.9, ls=LINESTYLES[2], zorder=1)

    slot = 0
    for name in models:
        if name not in scores_by_model:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores_by_model[name])
        ax.plot(
            fpr, tpr,
            color=SERIES[slot], lw=2, ls=LINESTYLES[slot % len(LINESTYLES)],
            label=f"{ROC_SHORT_LABELS.get(name, name)} (AUC {auc(fpr, tpr):.4f})",
            zorder=3 + slot,
        )
        slot += 1

    if reference and reference in scores_by_model:
        fpr, tpr, _ = roc_curve(y_true, scores_by_model[reference])
        ax.plot(
            fpr, tpr,
            color=INK_MUTED, lw=1.4, ls=LINESTYLES[1],
            label=f"{ROC_SHORT_LABELS.get(reference, reference)} (AUC {auc(fpr, tpr):.4f})",
            zorder=2,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.003)
    ax.set_xlabel("Taxa de falsos positivos (FPR)")
    ax.set_ylabel("Taxa de verdadeiros positivos (Recall)")
    ax.set_title(f"Curvas ROC no teste cego — {dataset}")
    ax.legend(loc="lower right")
    return _finish(fig, ax, path)


# ---------------------------------------------------------------------------
# Capítulo 5 — ablação do DEAS
# ---------------------------------------------------------------------------
def plot_deas_ablation(results: Sequence[Dict[str, Any]], path: str, metric: str = "f1_score") -> str:
    """Barras agrupadas λ=0 vs. DEAS por dataset, com o delta rotulado.

    Duas séries de magnitude no mesmo eixo; o delta é anotado em texto porque é
    a grandeza que o leitor busca, e não deve depender de medir barras no olho.
    """
    from src.evaluation.reporting import describe_fusion_degeneracy

    plt = _pyplot()
    # Um dataset cuja fusão degenerou é rotulado na própria figura: sem isso, a barra
    # "DEAS" mostraria um ganho que não vem do DEAS.
    diagnoses = [describe_fusion_degeneracy(r["calibration"], r["baselines"]) for r in results]
    labels = [
        f"{r['dataset']}\n(fusão degenerada, α={d['alpha']:g})" if d else r["dataset"]
        for r, d in zip(results, diagnoses)
    ]
    before = [float(r["baselines"]["B3b_PreIF_DIF_Pure"][metric]) for r in results]
    after = [float(r["baselines"]["B6a_Hybrid_Linear"][metric]) for r in results]

    x = np.arange(len(labels))
    width = 0.34
    gap = 0.012  # afastamento de 2px entre barras adjacentes, via superfície

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    bars_before = ax.bar(
        x - width / 2 - gap, before, width, color=SERIES[0],
        label="B3b · DIF puro (λ=0)",
    )
    bars_after = ax.bar(
        x + width / 2 + gap, after, width, color=SERIES[1],
        label="B6a · DIF+DEAS com fusão", hatch="//", edgecolor=SURFACE, linewidth=0.0,
    )

    for group in (bars_before, bars_after):
        for bar in group:
            ax.annotate(
                f"{bar.get_height():.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontsize=7.5, color=INK_SECONDARY,
            )

    metric_name = {"f1_score": "F1-Score", "auc_roc": "AUC-ROC"}.get(metric, metric)
    for i, (b, a) in enumerate(zip(before, after)):
        delta = a - b
        suffix = " (não é DEAS)" if diagnoses[i] else ""
        ax.annotate(
            f"Δ {delta:+.4f}{suffix}",
            xy=(x[i], max(b, a)),
            xytext=(0, 16), textcoords="offset points",
            ha="center", fontsize=8.5, fontweight="bold",
            # Sinal do delta é redundante com o próprio texto (+/-), não só cor.
            color=INK_PRIMARY,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    # Folga acima da barra mais alta para o rótulo do delta; os ticks param em 1.0
    # porque a métrica é limitada a [0, 1] e um tick acima disso seria enganoso.
    ax.set_ylim(0, 1.22)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_ylabel(metric_name)
    ax.set_title(f"Ablação do DEAS — ganho em {metric_name} no teste cego", pad=34)
    # Legenda acima da área de plotagem: dentro dela colidiria com as barras altas.
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), ncol=2)
    ax.grid(axis="x", visible=False)
    return _finish(fig, ax, path)


# ---------------------------------------------------------------------------
# Capítulo 5 — taxa de detecção estratificada
# ---------------------------------------------------------------------------
def plot_stratified_recall_heatmap(
    stratified_recall: Dict[str, Dict[str, Any]],
    dataset: str,
    path: str,
    models: Optional[Sequence[str]] = None,
    rare_classes: Optional[Dict[str, int]] = None,
) -> str:
    """Mapa de calor Recall × (família de ataque, modelo).

    Magnitude contínua -> rampa sequencial de uma matiz (azul), clara->escura.
    Nunca arco-íris. Cada célula traz o valor impresso, de modo que a informação
    não depende da cor.
    """
    plt = _pyplot()
    models = list(models or stratified_recall.keys())
    models = [m for m in models if m in stratified_recall]

    counts: Dict[str, int] = {}
    for per_model in stratified_recall.values():
        for family, stats in per_model.items():
            counts[str(family)] = int(stats.get("count", 0))
    families = sorted(counts, key=lambda f: -counts[f])

    matrix = np.array(
        [
            [float(stratified_recall.get(m, {}).get(f, {}).get("recall", np.nan)) for m in models]
            for f in families
        ]
    )

    # Dimensionado para caber em \linewidth de uma página A4 sem reduzir o texto a
    # ponto de ilegibilidade: uma figura de 12" encolhe para ~0,5x e derruba os
    # rótulos abaixo de 5 pt.
    fig, ax = plt.subplots(figsize=(0.88 * len(models) + 2.4, 0.40 * len(families) + 1.5))

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", list(SEQUENTIAL_BLUE))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    for i in range(len(families)):
        for j in range(len(models)):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            # Tinta clara sobre os passos escuros da rampa, escura sobre os claros.
            ax.text(
                j, i, f"{value * 100:.1f}",
                ha="center", va="center", fontsize=7.5,
                color="#ffffff" if value > 0.55 else INK_PRIMARY,
            )

    rare = rare_classes or {}
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(
        [f"{f} ({counts[f]:,})" + (RARE_MARKER if f in rare else "") for f in families], fontsize=8
    )
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(
        [ROC_SHORT_LABELS.get(m, m).split(" · ")[0] for m in models], fontsize=8
    )
    ax.set_title(f"Taxa de detecção por família de ataque (%) — {dataset}")
    ax.grid(visible=False)

    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    bar.set_label("Recall", fontsize=8, color=INK_SECONDARY)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, color=INK_MUTED)

    if rare:
        fig.text(
            0.01, -0.02,
            f"{RARE_MARKER.strip()} família com menos de 15 instâncias no teste: "
            f"alta variância, leitura qualitativa.",
            fontsize=7.5, color=INK_SECONDARY,
        )
    return _finish(fig, ax, path)


# ---------------------------------------------------------------------------
# Capítulo 5 — comparação entre datasets
# ---------------------------------------------------------------------------
def plot_cross_dataset_comparison(
    results: Sequence[Dict[str, Any]], path: str, metric: str = "f1_score"
) -> str:
    """Barras agrupadas: cada baseline em cada dataset (até 3 datasets = 3 slots).

    Três slots é o limite validado em `--pairs all`, adequado porque as barras de
    datasets diferentes aparecem lado a lado em todos os agrupamentos.
    """
    plt = _pyplot()
    datasets = [r["dataset"] for r in results]
    if len(datasets) > len(SERIES):
        raise ValueError("Mais datasets do que slots categóricos disponíveis.")

    model_order = [
        "B1_iForest_Raw", "B2_Autoencoder_Alone", "B3_PreIF_iForest",
        "B3b_PreIF_DIF_Pure", "B6a_Hybrid_Linear", "B6b_Hybrid_OR",
        "B4_RandomForest_Supervised",
    ]
    models = [m for m in model_order if all(m in r["baselines"] for r in results)]

    x = np.arange(len(models))
    total = len(results)
    width = 0.80 / max(1, total)

    fig, ax = plt.subplots(figsize=(0.80 * len(models) + 2.0, 3.8))
    for slot, result in enumerate(results):
        values = [float(result["baselines"][m][metric]) for m in models]
        offset = (slot - (total - 1) / 2) * (width + 0.01)
        ax.bar(
            x + offset, values, width,
            color=SERIES[slot], label=result["dataset"],
            hatch=["", "//", "\\\\", "xx"][slot % 4],
            edgecolor=SURFACE, linewidth=0.0,
        )

    metric_name = {"f1_score": "F1-Score", "auc_roc": "AUC-ROC", "recall": "Recall"}.get(metric, metric)
    ax.set_xticks(x)
    ax.set_xticklabels([ROC_SHORT_LABELS.get(m, m).split(" · ")[0] for m in models], fontsize=8)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} por baseline e dataset — teste cego")
    ax.legend(loc="lower left", ncol=min(3, total))
    ax.grid(axis="x", visible=False)
    return _finish(fig, ax, path)


# ---------------------------------------------------------------------------
# Capítulo 4 — sensibilidade OFAT
# ---------------------------------------------------------------------------
OFAT_FACTOR_LABELS: Dict[str, str] = {
    "m": "Dimensão latente (m)",
    "L": "Profundidade das redes Φᵢ (L)",
    "t": "Número de árvores (t)",
    "p": "Percentil do limiar não-supervisionado (p)",
    "lambda_deas": "Fator de ponderação DEAS (λ)",
}

# Quatro métricas no mesmo eixo [0, 1]; pairlist adjacente validado.
OFAT_METRICS: Tuple[Tuple[str, str], ...] = (
    ("f1_score", "F1-Score"),
    ("auc_roc", "AUC-ROC"),
    ("recall", "Recall"),
    ("fpr", "FPR (menor é melhor)"),
)


def plot_ofat_sweep(
    records: Sequence[Dict[str, Any]], factor: str, dataset: str, path: str
) -> str:
    """Curva de sensibilidade univariada de um fator OFAT."""
    plt = _pyplot()
    values = [r["param_value"] for r in records]
    positions = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for slot, (key, label) in enumerate(OFAT_METRICS):
        series = [float(r.get(key, np.nan)) for r in records]
        ax.plot(
            positions, series,
            color=SERIES[slot], lw=2, ls=LINESTYLES[slot % len(LINESTYLES)],
            marker=MARKERS[slot], ms=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.4,  # anel de superfície
            label=label,
        )
        # Rótulo direto no último ponto: <= 4 séries, então todas são rotuladas.
        if np.isfinite(series[-1]):
            ax.annotate(
                f"{series[-1]:.3f}",
                xy=(positions[-1], series[-1]),
                xytext=(5, 0), textcoords="offset points",
                fontsize=7.5, color=INK_SECONDARY, va="center",
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in values])
    ax.set_ylim(-0.02, 1.06)
    ax.set_xlabel(OFAT_FACTOR_LABELS.get(factor, factor))
    ax.set_ylabel("Métrica no teste cego")
    # O fator já está no rótulo do eixo x; repeti-lo no título só rouba espaço.
    ax.set_title(f"Sensibilidade OFAT — {dataset}")
    ax.legend(loc="center right")
    return _finish(fig, ax, path)
