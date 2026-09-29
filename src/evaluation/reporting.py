"""Construção dos relatórios Markdown do protocolo experimental (Módulo 5)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_METRIC_HEADER = (
    "| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | "
    "AUC-ROC | FPR (%) | Latência (ms/fluxo) |"
)
_METRIC_SEP = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"

BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "B1_iForest_Raw": "Isolation Forest clássico sobre as features brutas normalizadas.",
    "B2_Autoencoder_Alone": "FC-DAE isolado, decidindo pelo erro de reconstrução MSE.",
    "B3_PreIF_iForest": "Pre-IF: Isolation Forest clássico sobre o espaço latente z.",
    "B3b_PreIF_DIF_Pure": "Ablação: DIF com lambda = 0 (projeções Phi_i sem DEAS).",
    "B4_RandomForest_Supervised": "Teto comparativo supervisionado (Random Forest rotulado).",
    "B6a_Hybrid_Linear": "Arquitetura proposta: DIF+DEAS fundido linearmente ao MSE (alfa calibrado).",
    "B6b_Hybrid_OR": "Arquitetura proposta com regra de decisão disjuntiva (OR).",
}


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(number):
        return "—"
    return f"{number:.{digits}f}"


def describe_fusion_degeneracy(
    calibration: Dict[str, Any],
    baselines: Dict[str, Dict[str, Any]],
    tolerance: float = 1e-3,
) -> Optional[Dict[str, Any]]:
    """Detecta o colapso da fusão linear ponderada num de seus dois termos.

    A fusão é `alpha * s_DIF + (1 - alpha) * e_MSE_norm`, com `alpha` otimizado por
    grid search na validação. Quando o ótimo cai nos extremos do intervalo, B6a deixa
    de ser híbrido: em `alpha = 0` vira o autoencoder isolado (B2), em `alpha = 1` vira
    o DIF puro. Isso **invalida a leitura de `Delta_DEAS`**, que passa a comparar dois
    detectores distintos em vez de isolar a contribuição do desvio contínuo.

    :returns: `None` quando a fusão é genuína; caso contrário, o diagnóstico.
    """
    alpha = calibration.get("supervised", {}).get("B6a_Hybrid_Linear_alpha")
    if alpha is None:
        return None

    alpha = float(alpha)
    if alpha <= tolerance:
        equivalent, term = "B2_Autoencoder_Alone", "o erro de reconstrução do autoencoder"
    elif alpha >= 1.0 - tolerance:
        equivalent, term = "B3b_PreIF_DIF_Pure", "o escore do DIF"
    else:
        return None

    b6a = baselines.get("B6a_Hybrid_Linear", {})
    twin = baselines.get(equivalent, {})
    identical = (
        "f1_score" in b6a
        and "f1_score" in twin
        and abs(float(b6a["f1_score"]) - float(twin["f1_score"])) < 1e-12
    )

    return {
        "alpha": alpha,
        "equivalent_to": equivalent,
        "dominant_term": term,
        "metrics_identical": identical,
        "message": (
            f"A calibração escolheu `alpha = {alpha:g}`, de modo que a fusão linear "
            f"colapsa inteiramente em {term}: **B6a é equivalente a {equivalent}** neste "
            f"dataset"
            + (" (métricas idênticas)" if identical else "")
            + ". Em consequência, `Delta_DEAS` aqui **não** mede a contribuição do DEAS — "
            "compara o detector sobrevivente contra B3b. Não reporte esse delta como ganho "
            "do DEAS no texto; reporte o colapso e, se quiser o efeito isolado do lambda, "
            "compare `score_dif_standard` contra `score_dif_deas` da mesma floresta."
        ),
    }


def _metrics_table(summary_df: pd.DataFrame) -> List[str]:
    lines = [_METRIC_HEADER, _METRIC_SEP]
    for model_name, row in summary_df.iterrows():
        lines.append(
            f"| **{model_name}** "
            f"| {_fmt_pct(row.get('accuracy'))} "
            f"| {_fmt_pct(row.get('precision'))} "
            f"| {_fmt_pct(row.get('recall'))} "
            f"| {_fmt_pct(row.get('f1_score'))} "
            f"| {_fmt_num(row.get('auc_roc'))} "
            f"| {_fmt_pct(row.get('fpr'))} "
            f"| {_fmt_num(row.get('latency_ms_per_sample'), 4)} |"
        )
    return lines


def build_dataset_report(
    dataset_name: str,
    partition_meta: Dict[str, Any],
    ae_report: Dict[str, Any],
    dif_report: Dict[str, Any],
    calibration: Dict[str, Any],
    summary_df: pd.DataFrame,
    stratified_recall: Dict[str, Dict[str, Any]],
    delta_deas_f1: float,
    delta_deas_auc: float,
    rare_classes: Dict[str, int],
    ofat: Optional[Dict[str, Any]] = None,
    notes: Optional[Sequence[str]] = None,
) -> str:
    """Gera o relatório Markdown completo de um dataset."""
    sizes = partition_meta.get("sizes", {})
    protocol = partition_meta.get("protocol", {})
    lines: List[str] = []

    lines.append(f"# Relatório Experimental — {dataset_name}")
    lines.append("")
    lines.append(
        "Protocolo *leakage-free*: o conjunto de teste fornecido no artefato permanece "
        "cego. Toda calibração (limiares, ponto de operação de Youden e alfa da fusão) "
        "consome exclusivamente partições derivadas de `X_train_baseline`."
    )

    # --- Partições ---
    lines.append("")
    lines.append("## 1. Partições e Protocolo")
    lines.append("")
    lines.append("| Partição | Composição | Fluxos |")
    lines.append("| :--- | :--- | ---: |")
    lines.append(
        f"| Treino FC-DAE + DIF | benigno (`{partition_meta.get('normal_label')}`) | "
        f"{sizes.get('ae_train_normal', 0):,} |"
    )
    lines.append(f"| Validação não-supervisionada | benigno | {sizes.get('ae_val_normal', 0):,} |")
    lines.append(
        f"| Validação rotulada | benigno + ataques estratificados | "
        f"{sizes.get('val_total', 0):,} ({sizes.get('val_attacks', 0):,} ataques) |"
    )
    lines.append(f"| Ataques reservados ao B4 | ataques de treino | {sizes.get('sup_train_attacks', 0):,} |")
    lines.append(
        f"| **Teste cego** | benigno + ataques | "
        f"{sizes.get('test_total', 0):,} ({sizes.get('test_attacks', 0):,} ataques) |"
    )
    lines.append("")
    lines.append(
        f"- Features de entrada (n): **{partition_meta.get('n_features')}** "
        f"| coluna de rótulo: `{partition_meta.get('label_column')}`"
    )
    lines.append(
        f"- Divisão benigna treino/validação: "
        f"{(1 - protocol.get('val_normal_ratio', 0.15)):.0%} / {protocol.get('val_normal_ratio', 0.15):.0%}"
        f" | ataques de treino movidos para validação: {protocol.get('val_attack_ratio', 0.2):.0%}"
    )
    lines.append(f"- Semente determinística: `{protocol.get('random_state')}`")

    share = partition_meta.get("global_partition_share")
    if share:
        lines.append("")
        lines.append(
            "**Particionamento efetivo** (fração do total de cada classe, já que a fronteira "
            "treino/teste vem fixada pelo artefato):"
        )
        lines.append("")
        lines.append("| Classe | Treino não-supervisionado | Validação | Reservado ao B4 | Teste |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        lines.append(
            f"| Benigno ({share['normal']['total']:,}) | {share['normal']['train_unsupervised']:.1%} "
            f"| {share['normal']['validation']:.1%} | — | {share['normal']['test']:.1%} |"
        )
        lines.append(
            f"| Ataque ({share['attack']['total']:,}) | 0% "
            f"| {share['attack']['validation']:.1%} | {share['attack']['reserved_supervised']:.1%} "
            f"| {share['attack']['test']:.1%} |"
        )

    provenance = partition_meta.get("split_provenance")
    if provenance:
        lines.append("")
        lines.append("### 1.1 Procedência do Split Fornecido")
        lines.append("")
        lines.append("| Evidência | Valor |")
        lines.append("| :--- | :---: |")
        lines.append(f"| Fração do teste sobre o total | {provenance['test_fraction']:.2%} |")
        lines.append(
            f"| Maior divergência de proporção de classe (treino vs. teste) | "
            f"{provenance['max_class_proportion_delta_pp']:.4f} p.p. |"
        )
        lines.append(
            f"| Famílias exclusivas do teste | "
            f"{', '.join(provenance['families_only_in_test']) or 'nenhuma'} |"
        )
        lines.append(
            f"| Cenário de ataques inéditos (*zero-day*) avaliado | "
            f"{'sim' if provenance['preserves_novel_attack_scenario'] else '**não**'} |"
        )
        lines.append("")
        admonition = "WARNING" if provenance["looks_like_stratified_resplit"] else "NOTE"
        lines.append(f"> [!{admonition}]")
        lines.append(f"> {provenance['verdict']}")

    temporal = partition_meta.get("temporal_support")
    if temporal is not None and not temporal.get("b5_lstm_executable", False):
        lines.append("")
        lines.append("> [!WARNING]")
        lines.append(
            "> **Baseline B5 (LSTM-AE) indisponível neste artefato.** Nenhuma feature "
            "temporal foi detectada e não existe arquivo paralelo de timestamps/metadados "
            "no diretório do dataset, de modo que o janelamento cronológico (W=10, passo "
            "S=1) não tem base. Reexecutar o Módulo 1 preservando a coluna `Timestamp` é "
            "pré-requisito para B5."
        )
    if protocol.get("max_train_rows") or protocol.get("max_test_rows"):
        lines.append(
            f"> [!NOTE]\n> Execução com subamostragem estratificada ativa "
            f"(`max_train_rows={protocol.get('max_train_rows')}`, "
            f"`max_test_rows={protocol.get('max_test_rows')}`). "
            f"Os valores absolutos não são comparáveis a uma execução integral."
        )

    # --- Módulo 2 ---
    lines.append("")
    lines.append("## 2. Treino e Validação do FC-DAE (Módulo 2)")
    lines.append("")
    lines.append("| Item | Valor |")
    lines.append("| :--- | :---: |")
    lines.append(f"| Topologia | n={ae_report.get('input_dim')} → n/2 → n/4 → m={ae_report.get('bottleneck_dim')} → n/4 → n/2 → n |")
    lines.append(f"| Melhor perda de validação (MSE) | `{_fmt_num(ae_report.get('best_val_loss'), 6)}` |")
    lines.append(f"| Melhor época | {ae_report.get('best_epoch')} de {ae_report.get('epochs_run')} executadas |")
    lines.append(f"| Early stopping acionado | {'sim' if ae_report.get('early_stopped') else 'não'} |")
    lines.append(f"| Duração do treino | {_fmt_num(ae_report.get('train_seconds'), 1)} s |")
    lines.append(f"| Dispositivo | `{ae_report.get('device')}` |")
    lines.append("")
    rule = str(calibration.get("tau_ae_rule", "percentile"))
    lines.append(
        f"Limiar não-supervisionado do autoencoder, calibrado no MSE do tráfego benigno de "
        f"validação. **Regra adotada: `{rule}`** → "
        f"`tau_AE = {_fmt_num(calibration.get('tau_ae_ref'), 6)}`."
    )
    lines.append("")
    lines.append("| Formulação | `tau_AE` | FPR esperado no benigno | Adotada |")
    lines.append("| :--- | :---: | :---: | :---: |")
    lines.append(
        f"| Percentil `p = {_fmt_num(calibration.get('percentile'), 0)}` (Eq. 4 / Seção 4.7.1) "
        f"| `{_fmt_num(calibration.get('tau_ae_percentile'), 6)}` "
        f"| {_fmt_pct(calibration.get('expected_fpr_percentile'))} "
        f"| {'**sim**' if rule == 'percentile' else 'não'} |"
    )
    lines.append(
        f"| `mu + {_fmt_num(calibration.get('k_sigma'), 1)}*sigma` (`ae.md`) "
        f"| `{_fmt_num(calibration.get('tau_ae_mu_sigma'), 6)}` "
        f"| {_fmt_pct(calibration.get('expected_fpr_mu_sigma'))} "
        f"| {'**sim**' if rule == 'mu_sigma' else 'não'} |"
    )
    lines.append("")
    lines.append(
        "> [!IMPORTANT]\n"
        "> As duas formulações **não são equivalentes**. `mu + k*sigma` só coincidiria com "
        f"o percentil 99 se o MSE fosse aproximadamente normal; neste dataset ele cai no "
        f"percentil empírico **{_fmt_num(calibration.get('mu_sigma_empirical_percentile'), 2)}**. "
        "O texto do TCC e o código devem declarar a mesma regra — atualmente o padrão do "
        "pipeline é o percentil `p`, alinhado à Eq. 4 e à varredura OFAT do fator `p`."
    )

    # --- Módulo 4 ---
    lines.append("")
    lines.append("## 3. Treino e Calibração do DIF (Módulo 4)")
    lines.append("")
    lines.append("| Hiperparâmetro | Valor |")
    lines.append("| :--- | :---: |")
    lines.append(f"| Árvores de isolamento (t) | {dif_report.get('n_estimators')} |")
    lines.append(f"| Subamostra por árvore (psi) | {dif_report.get('max_samples')} |")
    lines.append(f"| Redes de projeção (r) | {dif_report.get('n_representations')} |")
    lines.append(f"| Camadas por rede (L) | {dif_report.get('n_layers')} |")
    lines.append(f"| Dimensão de projeção (d) | {dif_report.get('projection_dim')} |")
    lines.append(f"| Fator DEAS (lambda) | {dif_report.get('lambda_deas')} |")
    lines.append(f"| Duração do ajuste | {_fmt_num(dif_report.get('fit_seconds'), 1)} s |")
    lines.append("")
    lines.append(
        f"Limiar de referência não-supervisionado no percentil "
        f"{_fmt_num(dif_report.get('val_percentile'), 1)} da validação benigna: "
        f"`tau_DIF_ref(DEAS) = {_fmt_num(dif_report.get('tau_dif_ref_deas'), 6)}`, "
        f"`tau_DIF_ref(padrão) = {_fmt_num(dif_report.get('tau_dif_ref_standard'), 6)}`."
    )

    # --- Tabela 4 ---
    lines.append("")
    lines.append("## 4. Tabela 4 — Métricas Comparativas no Teste Cego")
    lines.append("")
    lines.extend(_metrics_table(summary_df))
    lines.append("")
    for name in summary_df.index:
        description = BASELINE_DESCRIPTIONS.get(str(name))
        if description:
            lines.append(f"- **{name}** — {description}")

    # --- Ablação ---
    lines.append("")
    lines.append("## 5. Ablação do DEAS")
    lines.append("")
    lines.append("Cadeia comparativa de três degraus:")
    lines.append("1. **Espaço latente vs. features brutas:** B1 → B3.")
    lines.append("2. **Projeções neurais não-lineares Phi_i:** B3 → B3b (DIF com lambda = 0).")
    lines.append("3. **Contribuição isolada do DEAS:** B3b → B6a (lambda > 0 + fusão linear).")
    lines.append("")
    lines.append(f"- Ganho de F1-Score (Delta_DEAS_F1): `{delta_deas_f1:+.4f}`")
    lines.append(f"- Ganho de AUC-ROC (Delta_DEAS_AUC): `{delta_deas_auc:+.4f}`")
    lines.append("")

    degeneracy = describe_fusion_degeneracy(calibration, summary_df.to_dict(orient="index"))
    if degeneracy:
        lines.append("> [!CAUTION]")
        lines.append(f"> **Fusão degenerada.** {degeneracy['message']}")
        lines.append("")
    lines.append(
        "> [!NOTE]\n"
        "> B3b e B6 compartilham a **mesma floresta** (a indução das árvores não "
        "depende de lambda). A topologia fica fixa e apenas o termo de desvio contínuo "
        "varia, o que torna o delta atribuível exclusivamente ao DEAS. Como h_deas e "
        "h_standard são acumulados na mesma travessia, a latência reportada para os "
        "dois é a do passo de inferência compartilhado."
    )

    # --- Recall estratificado ---
    if stratified_recall:
        lines.append("")
        lines.append("## 6. Taxa de Detecção Estratificada por Família de Ataque")
        lines.append("")
        model_names = [str(m) for m in summary_df.index if str(m) in stratified_recall]
        lines.append("| Família de Ataque | Total no Teste | " + " | ".join(model_names) + " |")
        lines.append("| :--- | ---: | " + " | ".join([":---:"] * len(model_names)) + " |")

        families: Dict[str, int] = {}
        for per_model in stratified_recall.values():
            for family, stats in per_model.items():
                families[str(family)] = int(stats.get("count", 0))

        for family in sorted(families, key=lambda f: -families[f]):
            row = [
                _fmt_pct(stratified_recall.get(m, {}).get(family, {}).get("recall"))
                for m in model_names
            ]
            flag = " ⚠️" if family in rare_classes else ""
            lines.append(f"| **{family}**{flag} | {families[family]:,} | " + " | ".join(row) + " |")

        if rare_classes:
            lines.append("")
            lines.append("> [!WARNING]")
            lines.append(
                "> **Ressalva estatística de classes raras (⚠️):** "
                + ", ".join(f"`{k}` ({v} instâncias)" for k, v in sorted(rare_classes.items()))
                + ". Com menos de "
                + str(protocol.get("rare_threshold", 15))
                + " instâncias no teste, as taxas de detecção têm alta variância e devem "
                "ser lidas qualitativamente."
            )

    # --- Calibração ---
    lines.append("")
    lines.append("## 7. Parâmetros Calibrados (Anti-Leakage)")
    lines.append("")
    lines.append("| Parâmetro | Valor | Regime |")
    lines.append("| :--- | :---: | :--- |")
    lines.append(
        f"| `tau_AE` (regra `{calibration.get('tau_ae_rule', 'percentile')}`) | "
        f"`{_fmt_num(calibration.get('tau_ae_ref'), 6)}` | não-supervisionado (benigno de validação) |"
    )
    lines.append(
        f"| `tau_DIF_ref` (p{_fmt_num(dif_report.get('val_percentile'), 1)}) | "
        f"`{_fmt_num(dif_report.get('tau_dif_ref_deas'), 6)}` | não-supervisionado (benigno de validação) |"
    )
    for key, value in calibration.get("supervised", {}).items():
        lines.append(f"| `{key}` | `{_fmt_num(value, 6)}` | supervisionado (Youden, validação rotulada) |")

    # --- OFAT ---
    if ofat and ofat.get("sweeps"):
        lines.append("")
        lines.append("## 8. Análise de Sensibilidade OFAT")
        lines.append("")
        for param, records in ofat["sweeps"].items():
            lines.append(f"### Parâmetro `{param}`")
            lines.append("")
            lines.append("| Valor | tau | F1-Score | AUC-ROC | Recall | FPR |")
            lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
            for record in records:
                lines.append(
                    f"| {record.get('param_value')} | {_fmt_num(record.get('tau'), 6)} "
                    f"| {_fmt_pct(record.get('f1_score'))} | {_fmt_num(record.get('auc_roc'))} "
                    f"| {_fmt_pct(record.get('recall'))} | {_fmt_pct(record.get('fpr'))} |"
                )
            lines.append("")
        if ofat.get("metadata", {}).get("methodology_notes"):
            lines.append("> [!NOTE]")
            lines.append("> " + str(ofat["metadata"]["methodology_notes"]))

    if notes:
        lines.append("")
        lines.append("## Observações Metodológicas")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def build_consolidated_report(results: Sequence[Dict[str, Any]]) -> str:
    """Gera o relatório comparativo entre os datasets executados."""
    lines: List[str] = []
    lines.append("# Consolidação Multi-Dataset — FC-DAE + Deep Isolation Forest")
    lines.append("")
    lines.append(
        "Execução do mesmo protocolo *leakage-free* sobre cada dataset processado. "
        "Todas as métricas referem-se ao conjunto de teste cego."
    )

    lines.append("")
    lines.append("## Visão Geral das Execuções")
    lines.append("")
    lines.append("| Dataset | n features | m latente | Treino benigno | Teste | MSE val. do FC-DAE |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        sizes = result["partitions"].get("sizes", {})
        lines.append(
            f"| **{result['dataset']}** "
            f"| {result['partitions'].get('n_features')} "
            f"| {result['autoencoder'].get('bottleneck_dim')} "
            f"| {sizes.get('ae_train_normal', 0):,} "
            f"| {sizes.get('test_total', 0):,} "
            f"| {_fmt_num(result['autoencoder'].get('best_val_loss'), 6)} |"
        )

    lines.append("")
    lines.append("## Arquitetura Proposta (B6a — Fusão Linear Ponderada)")
    lines.append("")
    lines.append("| Dataset | Acurácia | Precisão | Recall | F1-Score | AUC-ROC | FPR | alfa |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for result in results:
        row = result["baselines"].get("B6a_Hybrid_Linear", {})
        alpha = result["calibration"].get("supervised", {}).get("B6a_Hybrid_Linear_alpha")
        lines.append(
            f"| **{result['dataset']}** | {_fmt_pct(row.get('accuracy'))} | {_fmt_pct(row.get('precision'))} "
            f"| {_fmt_pct(row.get('recall'))} | {_fmt_pct(row.get('f1_score'))} "
            f"| {_fmt_num(row.get('auc_roc'))} | {_fmt_pct(row.get('fpr'))} | {_fmt_num(alpha, 2)} |"
        )

    lines.append("")
    lines.append("## Ablação do DEAS por Dataset")
    lines.append("")
    lines.append(
        "| Dataset | F1 B3b (lambda=0) | F1 B6a (DEAS) | Delta F1 | AUC B3b | AUC B6a | Delta AUC | alfa |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    degenerate: List[Dict[str, Any]] = []
    for result in results:
        b3b = result["baselines"].get("B3b_PreIF_DIF_Pure", {})
        b6a = result["baselines"].get("B6a_Hybrid_Linear", {})
        diagnosis = describe_fusion_degeneracy(result["calibration"], result["baselines"])
        alpha = result["calibration"].get("supervised", {}).get("B6a_Hybrid_Linear_alpha")
        flag = " ⚠" if diagnosis else ""
        if diagnosis:
            degenerate.append({"dataset": result["dataset"], **diagnosis})
        lines.append(
            f"| **{result['dataset']}**{flag} | {_fmt_pct(b3b.get('f1_score'))} "
            f"| {_fmt_pct(b6a.get('f1_score'))} | `{result['delta_deas_f1']:+.4f}` "
            f"| {_fmt_num(b3b.get('auc_roc'))} | {_fmt_num(b6a.get('auc_roc'))} "
            f"| `{result['delta_deas_auc']:+.4f}` | {_fmt_num(alpha, 2)} |"
        )

    if degenerate:
        lines.append("")
        lines.append("> [!CAUTION]")
        lines.append(
            "> **⚠ Fusão degenerada em "
            + ", ".join(d["dataset"] for d in degenerate)
            + ".** Nesses datasets a calibração levou `alfa` a um extremo do intervalo, "
            "de modo que B6a colapsa num de seus dois termos e o `Delta` da linha **não** "
            "mede a contribuição do DEAS. Detalhes no relatório de cada dataset."
        )

    lines.append("")
    lines.append("## Tabela 4 Completa por Dataset")
    for result in results:
        lines.append("")
        lines.append(f"### {result['dataset']}")
        lines.append("")
        summary_df = pd.DataFrame.from_dict(result["baselines"], orient="index")
        lines.extend(_metrics_table(summary_df))

    lines.append("")
    return "\n".join(lines)
