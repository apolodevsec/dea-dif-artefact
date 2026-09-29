"""Consolida figuras e tabelas prontas para os Capítulos 4 e 5 do TCC.

Lê o que a rotina de treino já gravou em `reports/` — sem retreinar nada — e emite
um pacote único em `reports/tcc_assets/`:

- figuras `.png` a 300 dpi, em modo claro, com identidade de série redundante
  (cor + marcador/hachura) para sobreviver à impressão monocromática;
- cada tabela em `.tex` (para `\\input{}`) **e** `.md` (para colar no Word);
- `MANIFEST.md`, que mapeia cada arquivo à seção do TCC onde entra.

Uso:

```powershell
python -m src.make_tcc_assets --reports_dir reports
```

Algumas figuras dependem de artefatos opcionais da rotina de treino e são
silenciosamente omitidas (com aviso no manifesto) quando eles não existem:

| Figura | Exige |
| :--- | :--- |
| Curvas ROC | `--save_scores` |
| Reconstrução diferencial (MSE) | `--save_latent` ou `--save_scores` |
| Sensibilidade OFAT | `--run_ofat` |
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

if __package__ in (None, ""):  # permite `python src/make_tcc_assets.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import figures, tables  # noqa: E402
from src.evaluation.reporting import build_consolidated_report  # noqa: E402

# Cada asset declara em que capítulo/seção do TCC ele entra.
CHAPTER_4 = "Capítulo 4 — Metodologia e Framework Proposto"
CHAPTER_5 = "Capítulo 5 — Resultados Experimentais"


class AssetIndex:
    """Acumula os assets gerados e as omissões, para o manifesto final."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.entries: List[Dict[str, str]] = []
        self.skipped: List[Dict[str, str]] = []

    def add(self, path: str, chapter: str, section: str, description: str) -> None:
        self.entries.append(
            {
                "path": os.path.relpath(path, self.root).replace(os.sep, "/"),
                "chapter": chapter,
                "section": section,
                "description": description,
            }
        )

    def add_pair(self, paths: Dict[str, str], chapter: str, section: str, description: str) -> None:
        for fmt in ("tex", "md"):
            if fmt in paths:
                self.add(paths[fmt], chapter, section, f"{description} (`.{fmt}`)")

    def skip(self, what: str, reason: str) -> None:
        self.skipped.append({"what": what, "reason": reason})
        print(f"  [omitido] {what} — {reason}")


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# Chaves introduzidas pela versão atual da rotina. Um relatório sem elas vem de uma
# execução anterior e não pode ser comparado lado a lado com os demais: o limiar
# tau_AE, por exemplo, mudou de mu+3sigma para o percentil p=95.
REQUIRED_SCHEMA_KEYS: Sequence[Sequence[str]] = (
    ("calibration", "tau_ae_rule"),
    ("partitions", "split_provenance"),
    ("partitions", "global_partition_share"),
)


def _missing_schema_keys(result: Dict[str, Any]) -> List[str]:
    missing: List[str] = []
    for path in REQUIRED_SCHEMA_KEYS:
        node: Any = result
        for key in path:
            if not isinstance(node, dict) or key not in node:
                missing.append(".".join(path))
                break
            node = node[key]
    return missing


def discover_results(reports_dir: str, allow_stale: bool = False) -> List[Dict[str, Any]]:
    """Coleta os `experimental_results.json` de cada subdiretório de dataset.

    Relatórios de execuções anteriores à versão atual da rotina são recusados por
    padrão: consolidá-los com os atuais produziria uma tabela em que datasets
    diferentes foram calibrados por regras diferentes.
    """
    results: List[Dict[str, Any]] = []
    stale: List[str] = []
    subsampled: List[str] = []

    if not os.path.isdir(reports_dir):
        raise NotADirectoryError(f"Diretório de relatórios inexistente: '{reports_dir}'")

    for entry in sorted(os.listdir(reports_dir)):
        candidate = os.path.join(reports_dir, entry, "experimental_results.json")
        payload = _load_json(candidate)
        if payload is None:
            continue

        missing = _missing_schema_keys(payload)
        if missing and not allow_stale:
            stale.append(f"{entry} (faltam: {', '.join(missing)})")
            continue

        protocol = payload.get("partitions", {}).get("protocol", {})
        if protocol.get("max_train_rows") or protocol.get("max_test_rows"):
            subsampled.append(entry)

        payload["_dir"] = os.path.join(reports_dir, entry)
        results.append(payload)

    if stale:
        raise ValueError(
            "Relatórios de uma versão anterior da rotina encontrados em "
            f"'{reports_dir}': {'; '.join(stale)}. "
            "Reexecute a rotina de treino nesses datasets (ou use --allow_stale para "
            "gerar os assets mesmo assim, ciente de que os datasets não são comparáveis)."
        )

    if subsampled:
        print(
            f"[AVISO] Execução subamostrada em: {', '.join(subsampled)}. "
            "Os valores absolutos não são comparáveis a uma execução integral e não "
            "devem ir para o TCC como resultado final."
        )

    if not results:
        raise FileNotFoundError(
            f"Nenhum 'experimental_results.json' encontrado sob '{reports_dir}'. "
            f"Rode primeiro: python -m src.train_pipeline --data_root <raiz-dos-dados>"
        )
    return results


def build_dataset_assets(result: Dict[str, Any], out_dir: str, index: AssetIndex) -> None:
    """Gera figuras e tabelas de um dataset."""
    dataset = result["dataset"]
    slug = dataset.replace(" ", "_")
    fig_dir = os.path.join(out_dir, "figuras")
    tab_dir = os.path.join(out_dir, "tabelas")
    source_dir = result["_dir"]

    print(f"\n[{dataset}]")

    # --- Capítulo 4: convergência do FC-DAE ---
    history = result.get("autoencoder", {}).get("history")
    if history:
        path = figures.plot_ae_convergence(
            history, dataset, os.path.join(fig_dir, f"convergencia_fc_dae_{slug}.png")
        )
        index.add(path, CHAPTER_4, "4.7 — FC-DAE", f"Curva de convergência treino/validação ({dataset})")
        print(f"  figura: convergencia_fc_dae_{slug}.png")
    else:
        index.skip(f"Convergência do FC-DAE ({dataset})", "histórico de treino ausente no JSON")

    # --- Capítulo 4: reconstrução diferencial (MSE benigno vs. ataque) ---
    mse_frame = None
    scores_path = os.path.join(source_dir, "test_scores.parquet")
    latent_path = os.path.join(source_dir, "latent_space_test.parquet")
    if os.path.exists(scores_path):
        frame = pd.read_parquet(scores_path)
        if "B2_Autoencoder_Alone" in frame.columns:
            mse_frame = (frame["B2_Autoencoder_Alone"].to_numpy(), frame["y_true"].to_numpy())
    elif os.path.exists(latent_path):
        frame = pd.read_parquet(latent_path)
        label_column = result["partitions"]["label_column"]
        if "mse" in frame.columns and label_column in frame.columns:
            normal = result["partitions"]["normal_label"]
            y_true = (frame[label_column].astype(str) != normal).astype(int).to_numpy()
            mse_frame = (frame["mse"].to_numpy(), y_true)

    if mse_frame is not None:
        path = figures.plot_mse_distribution(
            mse_frame[0], mse_frame[1], dataset,
            os.path.join(fig_dir, f"reconstrucao_diferencial_{slug}.png"),
            tau_ae=result["calibration"].get("tau_ae_ref"),
            tau_label=f"tau_AE ({result['calibration'].get('tau_ae_rule', 'percentile')})",
        )
        index.add(
            path, CHAPTER_4, "4.7.1 — Limiar de anomalia",
            f"Distribuição do MSE benigno vs. ataque com tau_AE ({dataset})",
        )
        print(f"  figura: reconstrucao_diferencial_{slug}.png")
    else:
        index.skip(
            f"Reconstrução diferencial ({dataset})",
            "rode a rotina com --save_scores (ou --save_latent)",
        )

    # --- Capítulo 5: curvas ROC da cadeia de ablação ---
    if os.path.exists(scores_path):
        frame = pd.read_parquet(scores_path)
        scores_by_model = {
            column: frame[column].to_numpy()
            for column in frame.columns
            if column in tables.MODEL_ORDER
        }
        path = figures.plot_roc_curves(
            frame["y_true"].to_numpy(), scores_by_model, dataset,
            os.path.join(fig_dir, f"curvas_roc_{slug}.png"),
        )
        index.add(
            path, CHAPTER_5, "5.x — Desempenho comparativo",
            f"Curvas ROC da cadeia de ablação, com teto supervisionado ({dataset})",
        )
        print(f"  figura: curvas_roc_{slug}.png")
    else:
        index.skip(f"Curvas ROC ({dataset})", "rode a rotina com --save_scores")

    # --- Capítulo 5: recall estratificado (mapa de calor) ---
    stratified = result.get("stratified_recall")
    if stratified:
        models = [m for m in tables.MODEL_ORDER if m in stratified]
        path = figures.plot_stratified_recall_heatmap(
            stratified, dataset,
            os.path.join(fig_dir, f"recall_estratificado_{slug}.png"),
            models=models, rare_classes=result.get("rare_classes", {}),
        )
        index.add(
            path, CHAPTER_5, "5.x — Detecção por família",
            f"Mapa de calor da taxa de detecção por família de ataque ({dataset})",
        )
        print(f"  figura: recall_estratificado_{slug}.png")

    # --- Capítulo 4: varreduras OFAT ---
    sweeps = result.get("ofat", {}).get("sweeps", {})
    for factor, records in sweeps.items():
        if not records:
            continue
        path = figures.plot_ofat_sweep(
            records, factor, dataset,
            os.path.join(fig_dir, f"ofat_{factor}_{slug}.png"),
        )
        index.add(
            path, CHAPTER_4, "4.8.6 — Sensibilidade OFAT",
            f"Varredura univariada do fator {factor} ({dataset})",
        )
        print(f"  figura: ofat_{factor}_{slug}.png")

    m_records = result.get("ofat", {}).get("m_retrained")
    if m_records:
        path = figures.plot_ofat_sweep(
            m_records, "m", dataset, os.path.join(fig_dir, f"ofat_m_{slug}.png")
        )
        index.add(
            path, CHAPTER_4, "4.8.6 — Sensibilidade OFAT",
            f"Varredura da dimensão latente m com retreino do FC-DAE ({dataset})",
        )
        print(f"  figura: ofat_m_{slug}.png")
    elif not sweeps:
        index.skip(f"Sensibilidade OFAT ({dataset})", "rode a rotina com --run_ofat")

    # --- Tabelas ---
    index.add_pair(
        tables.emit_table4(result, tab_dir), CHAPTER_5, "Tabela 4",
        f"Métricas comparativas no teste cego ({dataset})",
    )
    index.add_pair(
        tables.emit_stratified_recall(result, tab_dir), CHAPTER_5, "5.x — Detecção por família",
        f"Recall estratificado por família de ataque ({dataset})",
    )
    index.add_pair(
        tables.emit_calibration(result, tab_dir), CHAPTER_4, "4.8.4 — Calibração",
        f"Limiares e alfa calibrados sem vazamento ({dataset})",
    )
    print(f"  tabelas: tabela4 / recall_estratificado / calibracao ({slug})")


def build_cross_dataset_assets(
    results: Sequence[Dict[str, Any]], out_dir: str, index: AssetIndex
) -> None:
    """Gera os assets comparativos entre datasets."""
    fig_dir = os.path.join(out_dir, "figuras")
    tab_dir = os.path.join(out_dir, "tabelas")
    print("\n[consolidado]")

    complete = [
        r for r in results
        if "B3b_PreIF_DIF_Pure" in r.get("baselines", {})
        and "B6a_Hybrid_Linear" in r.get("baselines", {})
    ]

    if complete:
        path = figures.plot_deas_ablation(
            complete, os.path.join(fig_dir, "ablacao_deas_f1.png"), metric="f1_score"
        )
        index.add(
            path, CHAPTER_5, "5.x — Ablação do DEAS",
            "Ganho de F1 do DEAS por dataset (λ=0 vs. DEAS, mesma floresta)",
        )
        print("  figura: ablacao_deas_f1.png")

        index.add_pair(
            tables.emit_deas_ablation(complete, tab_dir), CHAPTER_5, "5.x — Ablação do DEAS",
            "Ganho isolado do DEAS em F1 e AUC por dataset",
        )
        print("  tabela: ablacao_deas")

    if len(results) >= 2:
        for metric, filename in (("f1_score", "comparativo_f1.png"), ("auc_roc", "comparativo_auc.png")):
            path = figures.plot_cross_dataset_comparison(
                results, os.path.join(fig_dir, filename), metric=metric
            )
            index.add(
                path, CHAPTER_5, "5.x — Comparação multi-dataset",
                f"{metric} de cada baseline nas bases avaliadas",
            )
            print(f"  figura: {filename}")
    else:
        index.skip(
            "Comparação multi-dataset",
            "apenas um dataset em reports/; rode a rotina com --data_root para as três bases",
        )

    index.add_pair(
        tables.emit_partition_summary(results, tab_dir), CHAPTER_4, "4.8.1 — Particionamento",
        "Protocolo de particionamento efetivo por dataset",
    )
    print("  tabela: particionamento")


def write_manifest(index: AssetIndex, results: Sequence[Dict[str, Any]]) -> str:
    """Escreve o mapa asset -> seção do TCC."""
    lines = [
        "# Assets para o TCC — Capítulos 4 e 5",
        "",
        "Gerado por `python -m src.make_tcc_assets` a partir dos relatórios em `reports/`.",
        "Não editar à mão: regenerar após cada execução da rotina de treino.",
        "",
        f"Datasets incluídos: {', '.join(r['dataset'] for r in results)}.",
        "",
        "## Como usar",
        "",
        "- **LaTeX:** `\\input{tabelas/tabela4_NSL-KDD.tex}` e "
        "`\\includegraphics[width=\\linewidth]{figuras/curvas_roc_NSL-KDD.png}`.",
        "- **Word / Google Docs:** cole o conteúdo do `.md` correspondente e insira o `.png`.",
        "- As figuras estão a 300 dpi, em modo claro, e identificam cada série por cor "
        "**e** marcador/hachura, de modo a permanecerem legíveis em impressão monocromática.",
        "",
    ]

    for chapter in (CHAPTER_4, CHAPTER_5):
        chapter_entries = [e for e in index.entries if e["chapter"] == chapter]
        if not chapter_entries:
            continue
        lines.extend([f"## {chapter}", "", "| Arquivo | Seção | Conteúdo |", "| :--- | :--- | :--- |"])
        for entry in sorted(chapter_entries, key=lambda e: (e["section"], e["path"])):
            lines.append(f"| `{entry['path']}` | {entry['section']} | {entry['description']} |")
        lines.append("")

    if index.skipped:
        lines.extend(["## Assets não gerados", "", "| Asset | Motivo |", "| :--- | :--- |"])
        for item in index.skipped:
            lines.append(f"| {item['what']} | {item['reason']} |")
        lines.append("")

    from src.evaluation.reporting import describe_fusion_degeneracy

    degenerate = [
        (r["dataset"], d)
        for r in results
        if (d := describe_fusion_degeneracy(r.get("calibration", {}), r.get("baselines", {})))
    ]
    if degenerate:
        lines.extend(["## Atenção antes de escrever os resultados", "", "> **Fusão degenerada.**", ""])
        for dataset, diagnosis in degenerate:
            lines.append(
                f"- **{dataset}**: `alpha = {diagnosis['alpha']:g}` — B6a colapsa em "
                f"`{diagnosis['equivalent_to']}`"
                + (" (métricas idênticas)" if diagnosis["metrics_identical"] else "")
                + ". O `Delta_DEAS` deste dataset **não** é o ganho do DEAS e não deve ser "
                "reportado como tal."
            )
        lines.append("")

    lines.extend(
        [
            "## Ressalvas metodológicas a transportar para o texto",
            "",
            "- Os splits fornecidos são refatiamentos estratificados 70/30, não os splits "
            "canônicos das bases: o cenário *zero-day* não está sendo avaliado.",
            "- `tau_AE` usa o percentil `p=95`; `ae.md` e `CONTEXT.md` ainda descrevem "
            "\"mu+3sigma ou percentil 99\" e precisam ser alinhados.",
            "- O baseline B5 (LSTM-AE) não foi executado: os artefatos não preservam `Timestamp`.",
            "- Famílias com menos de 15 instâncias no teste aparecem marcadas; sua taxa de "
            "detecção tem alta variância e deve ser lida qualitativamente.",
            "",
        ]
    )

    path = os.path.join(index.root, "MANIFEST.md")
    os.makedirs(index.root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return os.path.abspath(path)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Consolida figuras e tabelas prontas para os Capítulos 4 e 5 do TCC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reports_dir", type=str, default="reports", help="Diretório dos relatórios")
    parser.add_argument(
        "--out_dir", type=str, default=None, help="Destino (padrão: <reports_dir>/tcc_assets)"
    )
    parser.add_argument(
        "--datasets", type=str, default=None, help="Lista separada por vírgula para filtrar datasets"
    )
    parser.add_argument(
        "--allow_stale",
        action="store_true",
        help="Inclui relatórios de versões anteriores da rotina (datasets deixam de ser comparáveis)",
    )
    parser.add_argument(
        "--no_refresh_consolidated",
        action="store_true",
        help="Não reconstrói reports/consolidated_results.md a partir dos JSONs encontrados",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or os.path.join(args.reports_dir, "tcc_assets")
    results = discover_results(args.reports_dir, allow_stale=args.allow_stale)

    if args.datasets:
        wanted = [n.strip().lower() for n in args.datasets.split(",")]
        results = [r for r in results if any(w in r["dataset"].lower() for w in wanted)]
        if not results:
            parser.error(f"Nenhum dataset corresponde a '{args.datasets}'.")

    index = AssetIndex(out_dir)
    print(f"Gerando assets do TCC em '{os.path.abspath(out_dir)}'")

    for result in results:
        build_dataset_assets(result, out_dir, index)
    build_cross_dataset_assets(results, out_dir, index)

    # A rotina de treino grava o consolidado apenas com os datasets daquela invocação,
    # de modo que rodar as bases separadamente deixa um consolidado incompleto. Aqui
    # ele é reconstruído a partir de todos os JSONs presentes, sem retreinar nada.
    consolidated = None
    if not args.no_refresh_consolidated and not args.datasets:
        consolidated = os.path.join(args.reports_dir, "consolidated_results.md")
        with open(consolidated, "w", encoding="utf-8") as handle:
            handle.write(build_consolidated_report(results))
        print(
            f"\nConsolidado reconstruído com {len(results)} dataset(s): '{consolidated}'"
        )
    elif args.datasets:
        print("\n[NOTA] Consolidado não reconstruído: --datasets filtra o conjunto.")

    manifest = write_manifest(index, results)
    print(f"\n{len(index.entries)} assets gerados, {len(index.skipped)} omitidos.")
    print(f"Manifesto: '{manifest}'")

    return {
        "assets": index.entries,
        "skipped": index.skipped,
        "manifest": manifest,
        "consolidated": consolidated,
    }


if __name__ == "__main__":
    main()
