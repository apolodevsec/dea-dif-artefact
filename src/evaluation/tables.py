"""Emissão das tabelas do TCC em LaTeX e Markdown.

Cada tabela é gerada nos dois formatos a partir do mesmo `experimental_results.json`,
porque o formato de destino depende de como o TCC é escrito: `.tex` para `\\input{}`
em LaTeX, `.md` para colar em Word/Google Docs. Os números são idênticos nos dois.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Rótulo curto e legível de cada baseline nas tabelas do texto.
MODEL_LABELS: Dict[str, str] = {
    "B1_iForest_Raw": "B1 — iForest (features brutas)",
    "B2_Autoencoder_Alone": "B2 — FC-DAE isolado (MSE)",
    "B3_PreIF_iForest": "B3 — Pre-IF + iForest clássico",
    "B3b_PreIF_DIF_Pure": "B3b — Pre-IF + DIF puro ($\\lambda = 0$)",
    "B4_RandomForest_Supervised": "B4 — Random Forest supervisionado",
    "B6a_Hybrid_Linear": "B6a — Fusão linear ponderada",
    "B6b_Hybrid_OR": "B6b — Regra disjuntiva (OR)",
}

MODEL_ORDER: Sequence[str] = (
    "B1_iForest_Raw",
    "B2_Autoencoder_Alone",
    "B3_PreIF_iForest",
    "B3b_PreIF_DIF_Pure",
    "B4_RandomForest_Supervised",
    "B6a_Hybrid_Linear",
    "B6b_Hybrid_OR",
)

_LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}


def _escape_latex(text: Any) -> str:
    """Escapa um rótulo de texto, preservando trechos matemáticos entre `$`."""
    raw = str(text)
    if "$" in raw:  # já contém matemática deliberada; escapa só o que é seguro
        return raw.replace("%", r"\%").replace("&", r"\&")
    for char, replacement in _LATEX_ESCAPES.items():
        raw = raw.replace(char, replacement)
    return raw


def _pct(value: Any, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(number):
        return "--"
    return f"{number * 100:.{digits}f}"


def _num(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(number):
        return "--"
    return f"{number:.{digits}f}"


def _write(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return os.path.abspath(path)


def _latex_table(
    caption: str,
    label: str,
    column_spec: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: Optional[str] = None,
) -> str:
    lines = [
        r"% Gerado por src/make_tcc_assets.py -- nao editar a mao.",
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\small",
        f"\\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(header) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(rf"\begin{{flushleft}}\footnotesize {note}\end{{flushleft}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _markdown_table(
    title: str, header: Sequence[str], rows: Sequence[Sequence[str]], note: Optional[str] = None
) -> str:
    lines = [f"**{title}**", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| :--- " + "| ---: " * (len(header) - 1) + "|")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    if note:
        lines.extend(["", f"_{note}_"])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Tabela 4 — métricas comparativas no teste cego
# ---------------------------------------------------------------------------
def table4_rows(baselines: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for name in MODEL_ORDER:
        metrics = baselines.get(name)
        if not metrics:
            continue
        rows.append(
            [
                MODEL_LABELS.get(name, name),
                _pct(metrics.get("accuracy")),
                _pct(metrics.get("precision")),
                _pct(metrics.get("recall")),
                _pct(metrics.get("f1_score")),
                _num(metrics.get("auc_roc")),
                _pct(metrics.get("fpr")),
                _num(metrics.get("latency_ms_per_sample")),
            ]
        )
    return rows


def emit_table4(result: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    dataset = result["dataset"]
    slug = dataset.replace(" ", "_")
    header = [
        "Modelo", "Acurácia (\\%)", "Precisão (\\%)", "Recall (\\%)",
        "F1 (\\%)", "AUC-ROC", "FPR (\\%)", "Latência (ms)",
    ]
    rows = [[_escape_latex(r[0])] + r[1:] for r in table4_rows(result["baselines"])]

    tex = _latex_table(
        caption=f"Métricas comparativas de detecção no conjunto de teste cego — {dataset}.",
        label=f"tab:tabela4-{slug.lower()}",
        column_spec="lrrrrrrr",
        header=header,
        rows=rows,
        note=(
            "Limiares calibrados exclusivamente na partição de validação. "
            "B3b e B6a compartilham a mesma floresta, de modo que a latência reportada "
            "para ambos é a do passo de inferência compartilhado."
        ),
    )
    md_header = [h.replace("\\%", "%") for h in header]
    md_rows = [[r[0].replace("$\\lambda = 0$", "λ = 0").replace("\\_", "_")] + r[1:] for r in rows]
    md = _markdown_table(f"Tabela 4 — {dataset}", md_header, md_rows)

    return {
        "tex": _write(os.path.join(out_dir, f"tabela4_{slug}.tex"), tex),
        "md": _write(os.path.join(out_dir, f"tabela4_{slug}.md"), md),
    }


# ---------------------------------------------------------------------------
# Recall estratificado por família de ataque
# ---------------------------------------------------------------------------
def emit_stratified_recall(result: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    dataset = result["dataset"]
    slug = dataset.replace(" ", "_")
    stratified = result["stratified_recall"]
    rare = result.get("rare_classes", {})

    models = [m for m in MODEL_ORDER if m in stratified]
    counts: Dict[str, int] = {}
    for per_model in stratified.values():
        for family, stats in per_model.items():
            counts[str(family)] = int(stats.get("count", 0))
    families = sorted(counts, key=lambda f: -counts[f])

    short = [m.split("_")[0] for m in models]
    header = ["Família de ataque", "Instâncias"] + short
    rows: List[List[str]] = []
    for family in families:
        marker = r"$^{\dagger}$" if family in rare else ""
        row = [_escape_latex(family) + marker, f"{counts[family]:,}".replace(",", ".")]
        row += [_pct(stratified.get(m, {}).get(family, {}).get("recall"), 1) for m in models]
        rows.append(row)

    note = (
        "Taxa de detecção (Recall) por família, em \\%. "
        + (
            r"$^{\dagger}$ família com menos de 15 instâncias no teste: alta variância amostral, "
            "leitura qualitativa."
            if rare
            else ""
        )
    )
    tex = _latex_table(
        caption=f"Taxa de detecção estratificada por família de ataque — {dataset}.",
        label=f"tab:recall-estratificado-{slug.lower()}",
        column_spec="lr" + "r" * len(models),
        header=header,
        rows=rows,
        note=note,
    )
    md_rows = [[r[0].replace(r"$^{\dagger}$", " ⚠").replace("\\_", "_")] + r[1:] for r in rows]
    md = _markdown_table(
        f"Recall estratificado — {dataset}", header, md_rows, note.replace(r"$^{\dagger}$", "⚠")
    )

    return {
        "tex": _write(os.path.join(out_dir, f"recall_estratificado_{slug}.tex"), tex),
        "md": _write(os.path.join(out_dir, f"recall_estratificado_{slug}.md"), md),
    }


# ---------------------------------------------------------------------------
# Calibração e ablação
# ---------------------------------------------------------------------------
def emit_calibration(result: Dict[str, Any], out_dir: str) -> Dict[str, str]:
    dataset = result["dataset"]
    slug = dataset.replace(" ", "_")
    calibration, dif = result["calibration"], result["dif"]

    rows = [
        [
            f"$\\tau_{{AE}}$ (percentil $p={_num(calibration.get('percentile'), 0)}$)",
            _num(calibration.get("tau_ae_percentile"), 6),
            "Não supervisionado",
        ],
        [
            f"$\\tau_{{AE}}$ ($\\mu + {_num(calibration.get('k_sigma'), 1)}\\sigma$)",
            _num(calibration.get("tau_ae_mu_sigma"), 6),
            "Não supervisionado",
        ],
        [
            f"$\\tau_{{DIF,ref}}$ (percentil $p={_num(dif.get('val_percentile'), 0)}$)",
            _num(dif.get("tau_dif_ref_deas"), 6),
            "Não supervisionado",
        ],
    ]
    for key, value in result["calibration"].get("supervised", {}).items():
        rows.append([_escape_latex(key), _num(value, 6), "Supervisionado (Youden)"])

    note = (
        "Regra adotada para $\\tau_{AE}$: "
        f"\\texttt{{{calibration.get('tau_ae_rule', 'percentile')}}}. "
        "Os limiares não supervisionados usam apenas tráfego benigno de validação; "
        "os supervisionados usam a validação rotulada e nunca o teste."
    )
    tex = _latex_table(
        caption=f"Parâmetros calibrados sem vazamento — {dataset}.",
        label=f"tab:calibracao-{slug.lower()}",
        column_spec="lrl",
        header=["Parâmetro", "Valor", "Regime"],
        rows=rows,
        note=note,
    )
    md_rows = [
        [
            r[0].replace("$\\tau_{AE}$", "τ_AE").replace("$\\tau_{DIF,ref}$", "τ_DIF_ref")
            .replace("$\\mu + ", "μ + ").replace("\\sigma$", "σ").replace("$p=", "p=")
            .replace("$", "").replace("\\_", "_"),
            r[1], r[2],
        ]
        for r in rows
    ]
    md = _markdown_table(
        f"Calibração — {dataset}", ["Parâmetro", "Valor", "Regime"], md_rows,
        note.replace("$\\tau_{AE}$", "τ_AE").replace("\\texttt{", "`").replace("}", "`"),
    )
    return {
        "tex": _write(os.path.join(out_dir, f"calibracao_{slug}.tex"), tex),
        "md": _write(os.path.join(out_dir, f"calibracao_{slug}.md"), md),
    }


def emit_deas_ablation(results: Sequence[Dict[str, Any]], out_dir: str) -> Dict[str, str]:
    rows: List[List[str]] = []
    for result in results:
        b3b = result["baselines"]["B3b_PreIF_DIF_Pure"]
        b6a = result["baselines"]["B6a_Hybrid_Linear"]
        rows.append(
            [
                _escape_latex(result["dataset"]),
                _pct(b3b.get("f1_score")),
                _pct(b6a.get("f1_score")),
                f"{result['delta_deas_f1']:+.4f}",
                _num(b3b.get("auc_roc")),
                _num(b6a.get("auc_roc")),
                f"{result['delta_deas_auc']:+.4f}",
            ]
        )

    header = [
        "Dataset", "F1 B3b (\\%)", "F1 B6a (\\%)", "$\\Delta$ F1",
        "AUC B3b", "AUC B6a", "$\\Delta$ AUC",
    ]
    note = (
        "B3b e B6a compartilham a mesma floresta de isolamento (a indução das árvores "
        "não depende de $\\lambda$), de modo que o delta é atribuível exclusivamente ao DEAS."
    )
    tex = _latex_table(
        caption="Ablação do DEAS: ganho isolado do desvio contínuo por dataset.",
        label="tab:ablacao-deas",
        column_spec="lrrrrrr",
        header=header,
        rows=rows,
        note=note,
    )
    md_header = [h.replace("\\%", "%").replace("$\\Delta$", "Δ") for h in header]
    md = _markdown_table(
        "Ablação do DEAS", md_header, rows, note.replace("$\\lambda$", "λ")
    )
    return {
        "tex": _write(os.path.join(out_dir, "ablacao_deas.tex"), tex),
        "md": _write(os.path.join(out_dir, "ablacao_deas.md"), md),
    }


def emit_partition_summary(results: Sequence[Dict[str, Any]], out_dir: str) -> Dict[str, str]:
    """Tabela do protocolo de particionamento efetivo, por dataset."""
    rows: List[List[str]] = []
    for result in results:
        partitions = result["partitions"]
        sizes = partitions["sizes"]
        share = partitions.get("global_partition_share", {})
        normal = share.get("normal", {})
        attack = share.get("attack", {})
        rows.append(
            [
                _escape_latex(result["dataset"]),
                str(partitions["n_features"]),
                str(result["autoencoder"]["bottleneck_dim"]),
                f"{sizes['ae_train_normal']:,}".replace(",", "."),
                f"{sizes['val_total']:,}".replace(",", "."),
                f"{sizes['test_total']:,}".replace(",", "."),
                f"{normal.get('train_unsupervised', 0) * 100:.1f}/"
                f"{normal.get('validation', 0) * 100:.1f}/"
                f"{normal.get('test', 0) * 100:.1f}",
                f"{attack.get('validation', 0) * 100:.1f}/"
                f"{attack.get('test', 0) * 100:.1f}",
            ]
        )

    header = [
        "Dataset", "$n$", "$m$", "Treino benigno", "Validação", "Teste",
        "Benigno tr/val/te (\\%)", "Ataque val/te (\\%)",
    ]
    note = (
        "A fronteira treino/teste é fixada pelos artefatos processados, de modo que as "
        "frações efetivas diferem do esquema nominal 70/15/15; o teste permanece cego."
    )
    tex = _latex_table(
        caption="Protocolo de particionamento efetivo por dataset.",
        label="tab:particionamento",
        column_spec="lrrrrrrr",
        header=header,
        rows=rows,
        note=note,
    )
    md_header = [h.replace("\\%", "%").replace("$n$", "n").replace("$m$", "m") for h in header]
    md = _markdown_table("Particionamento efetivo", md_header, rows, note)
    return {
        "tex": _write(os.path.join(out_dir, "particionamento.tex"), tex),
        "md": _write(os.path.join(out_dir, "particionamento.md"), md),
    }
