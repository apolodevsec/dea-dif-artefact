import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.evaluation.partitioner import DataPartitioner
from src.evaluation.evaluator import ExperimentalEvaluator
from src.evaluation.baselines import BaselineRunner
from src.evaluation.sensitivity import OFATSensitivityAnalyzer


def generate_synthetic_benchmark_data(
    n_samples: int = 1500,
    n_features: int = 77,
    m_latent: int = 9,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Gera um dataset sintético para teste e validação controlada do pipeline de avaliação."""
    rng = np.random.RandomState(random_state)

    n_normal = int(n_samples * 0.8)
    n_dos = int(n_samples * 0.12)
    n_scan = int(n_samples * 0.07)
    n_rare = n_samples - (n_normal + n_dos + n_scan)  # ~1% (~10-15 amostras raras)

    # Tráfego Benigno
    X_norm = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features)).astype(np.float32)
    labels_norm = ["BENIGN"] * n_normal

    # Ataques DoS
    X_dos = rng.normal(loc=3.5, scale=1.5, size=(n_dos, n_features)).astype(np.float32)
    labels_dos = ["DoS"] * n_dos

    # Ataques PortScan
    X_scan = rng.normal(loc=5.0, scale=1.2, size=(n_scan, n_features)).astype(np.float32)
    labels_scan = ["PortScan"] * n_scan

    # Ataque Raro (ex: Heartbleed)
    X_rare = rng.normal(loc=4.2, scale=1.0, size=(n_rare, n_features)).astype(np.float32)
    labels_rare = ["Heartbleed"] * n_rare

    X_all = np.vstack([X_norm, X_dos, X_scan, X_rare])
    labels_all = labels_norm + labels_dos + labels_scan + labels_rare

    # Gera projeções latentes Z e erros MSE sintéticos coerentes
    Z_all = X_all[:, :m_latent].copy()

    # Erro de reconstrução MSE: baixo para benigno, elevado para ataques
    mse_norm = rng.uniform(0.0001, 0.002, size=n_normal).astype(np.float32)
    mse_dos = rng.uniform(0.006, 0.04, size=n_dos).astype(np.float32)
    mse_scan = rng.uniform(0.01, 0.07, size=n_scan).astype(np.float32)
    mse_rare = rng.uniform(0.008, 0.05, size=n_rare).astype(np.float32)
    mse_all = np.concatenate([mse_norm, mse_dos, mse_scan, mse_rare])

    # Cria dataframe com features brutas e coluna de rótulo
    cols = [f"feat_{i}" for i in range(n_features)]
    df = pd.DataFrame(X_all, columns=cols)
    df["Label"] = labels_all

    return df, Z_all, mse_all


def build_markdown_report(
    summary_df: pd.DataFrame,
    stratified_recall: Dict[str, Dict[str, Any]],
    thresholds: Dict[str, float],
    delta_deas_f1: float,
    delta_deas_auc: float,
    rare_classes: list[str],
) -> str:
    """Gera o relatório estruturado em Markdown aderente à Tabela 4 do TCC."""
    lines = []
    lines.append("# Relatório de Validação Experimental e Avaliação de Baselines")
    lines.append("\n**Protocolo de Validação:** 70/15/15 tráfego normal, 20/80 anomalias estratificadas (*Leakage-Free*).")
    lines.append("\n## Tabela 4: Métricas Comparativas de Baselines\n")

    # Cabeçalho da Tabela 4
    lines.append("| Modelo | Acurácia (%) | Precisão (%) | Recall (%) | F1-Score (%) | AUC-ROC | FPR (%) | Latência (ms/fluxo) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for model_name, row in summary_df.iterrows():
        acc = f"{row.get('accuracy', 0.0) * 100:.2f}%"
        prec = f"{row.get('precision', 0.0) * 100:.2f}%"
        rec = f"{row.get('recall', 0.0) * 100:.2f}%"
        f1 = f"{row.get('f1_score', 0.0) * 100:.2f}%"
        auc = f"{row.get('auc_roc', float('nan')):.4f}"
        fpr = f"{row.get('fpr', 0.0) * 100:.2f}%"
        lat = f"{row.get('latency_ms_per_sample', 0.0):.4f}"
        lines.append(f"| **{model_name}** | {acc} | {prec} | {rec} | {f1} | {auc} | {fpr} | {lat} |")

    lines.append("\n## Análise de Ablação do DEAS\n")
    lines.append("A cadeia comparativa de três degraus de ablação isola metodologicamente as contribuições:")
    lines.append("1. **Espaço Latente vs. Features Brutas:** B1 (iForest Bruto) $\\to$ B3 (Pre-IF Latente).")
    lines.append("2. **Projeções Não-Lineares Neurais ($\\Phi_i$):** B3 (iForest Clássico) $\\to$ B3b (DIF Puro, $\\lambda=0$).")
    lines.append("3. **Impacto Específico do DEAS:** B3b (DIF $\\lambda=0$) $\\to$ B6a (DIF com DEAS $\\lambda=0.5$ + Fusão Linear).")
    lines.append(f"\n- **Ganho de F1-Score ($\\Delta_{{\\text{{DEAS}}}}^{{\\text{{F1}}}}$):** `{delta_deas_f1:+.4f}`")
    lines.append(f"- **Ganho de AUC-ROC ($\\Delta_{{\\text{{DEAS}}}}^{{\\text{{AUC}}}}$):** `{delta_deas_auc:+.4f}`")

    lines.append("\n## Recall Estratificado por Família de Ataque\n")
    lines.append("| Família de Ataque | Total no Teste | " + " | ".join([f"{m} (%)" for m in summary_df.index]) + " |")
    lines.append("| :--- | :---: | " + " | ".join([":---:" for _ in summary_df.index]) + " |")

    # Identifica todas as famílias de ataque presentes
    all_attacks = set()
    for m in stratified_recall.values():
        all_attacks.update(m.keys())

    for att in sorted(list(all_attacks)):
        counts = [m[att]["count"] for m in stratified_recall.values() if att in m]
        total_cnt = counts[0] if counts else 0
        row_vals = []
        for model_name in summary_df.index:
            rec_val = stratified_recall.get(model_name, {}).get(att, {}).get("recall", 0.0)
            row_vals.append(f"{rec_val * 100:.2f}%")
        lines.append(f"| **{att}** | {total_cnt} | " + " | ".join(row_vals) + " |")

    if rare_classes:
        lines.append("\n> [!WARNING]")
        lines.append(f"> **Ressalva Estatística de Classes Raras:** As classes {', '.join(rare_classes)} possuem menos de 15 instâncias registradas. Suas taxas de detecção apresentam alta variância e devem ser interpretadas qualitativamente.")

    lines.append("\n## Parâmetros de Calibração sem Vazamento (*Anti-Leakage*)\n")
    lines.append("| Parâmetro / Limiar | Valor Calibrado na Validação |")
    lines.append("| :--- | :---: |")
    for k, v in thresholds.items():
        lines.append(f"| `{k}` | `{v:.5f}` |")

    return "\n".join(lines)


def run_evaluation_pipeline(
    data_path: Optional[str] = None,
    synthetic: bool = False,
    n_samples: int = 1500,
    n_features: int = 77,
    m_latent: int = 9,
    output_dir: str = "reports",
    random_state: int = 42,
    run_ofat: bool = False,
    run_lstm: bool = False,
) -> Dict[str, Any]:
    """Executa o pipeline completo de avaliação experimental."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Carregamento dos dados
    if synthetic or data_path is None or not os.path.exists(data_path):
        df, Z_all, mse_all = generate_synthetic_benchmark_data(
            n_samples=n_samples,
            n_features=n_features,
            m_latent=m_latent,
            random_state=random_state,
        )
    else:
        df = pd.read_parquet(data_path)
        feature_cols = [c for c in df.columns if c not in ["Label", "Attack_Category", "Timestamp"]]
        # ATENÇÃO: este caminho NÃO treina o FC-DAE. O "latente" é apenas um recorte das
        # m primeiras features brutas e o MSE é preenchido com zeros, o que degenera B2
        # (autoencoder isolado) e a fusão híbrida B6a/B6b. Serve somente para exercitar a
        # mecânica do pipeline sobre um parquet monolítico.
        # Para resultados válidos, use `python -m src.train_pipeline`, que treina o
        # FC-DAE e deriva z e o MSE do modelo ajustado.
        print(
            "[AVISO] Modo --data_path: espaço latente e MSE são substitutos sintéticos "
            "(sem FC-DAE treinado). As métricas de B2, B6a e B6b não são válidas. "
            "Use 'python -m src.train_pipeline' para a rotina completa de treino e validação."
        )
        Z_all = df[feature_cols[:m_latent]].to_numpy(dtype=np.float32)
        mse_all = np.zeros(len(df), dtype=np.float32)

    # 2. Particionamento sem vazamento
    partitioner = DataPartitioner(random_state=random_state)
    split_res = partitioner.split(df, label_column="Label", normal_label="BENIGN")

    idx_train = split_res["train"].index.to_numpy()
    idx_val = split_res["val"].index.to_numpy()
    idx_test = split_res["test"].index.to_numpy()
    rare_classes = list(split_res["rare_classes"].keys())

    feat_cols = [c for c in df.columns if c != "Label"]
    X_raw = df[feat_cols].to_numpy(dtype=np.float32)
    y_raw = (df["Label"].to_numpy() != "BENIGN").astype(int)
    classes = df["Label"].to_numpy()

    # Partições de matrizes
    X_train_raw = X_raw[idx_train]
    Z_train = Z_all[idx_train]
    mse_train = mse_all[idx_train]

    X_val_raw = X_raw[idx_val]
    Z_val = Z_all[idx_val]
    mse_val = mse_all[idx_val]
    y_val = y_raw[idx_val]

    X_test_raw = X_raw[idx_test]
    Z_test = Z_all[idx_test]
    mse_test = mse_all[idx_test]
    y_test = y_raw[idx_test]
    test_classes = classes[idx_test]

    # 3. Execução dos Baselines Tabulares
    runner = BaselineRunner(random_state=random_state, n_jobs=1)
    base_results = runner.run_all(
        X_train_raw=X_train_raw,
        Z_train=Z_train,
        mse_train=mse_train,
        X_val_raw=X_val_raw,
        Z_val=Z_val,
        mse_val=mse_val,
        y_val=y_val,
        X_test_raw=X_test_raw,
        Z_test=Z_test,
        mse_test=mse_test,
        y_test=y_test,
        test_classes=test_classes,
    )

    summary_df = base_results["summary_df"]
    stratified_recall = base_results["stratified_recall"]
    thresholds = base_results["thresholds"]
    delta_deas_f1 = base_results["delta_deas_f1"]
    delta_deas_auc = base_results["delta_deas_auc"]

    # 4. Análise de Sensibilidade OFAT (Opcional)
    ofat_results: Dict[str, Any] = {}
    if run_ofat:
        analyzer = OFATSensitivityAnalyzer(random_state=random_state, n_jobs=1)
        fig_dir = os.path.join(output_dir, "figures", "sensitivity")
        sweep_dfs = analyzer.run_sweeps(
            X_train=Z_train,
            X_val=Z_val,
            y_val=y_val,
            X_test=Z_test,
            y_test=y_test,
            output_figure_dir=fig_dir,
        )
        ofat_results["sweeps"] = {k: v.to_dict(orient="records") for k, v in sweep_dfs.items()}
        ofat_results["metadata"] = analyzer.get_metadata()

    # 5. Exportação de Relatórios
    md_content = build_markdown_report(
        summary_df=summary_df,
        stratified_recall=stratified_recall,
        thresholds=thresholds,
        delta_deas_f1=delta_deas_f1,
        delta_deas_auc=delta_deas_auc,
        rare_classes=rare_classes,
    )

    md_path = os.path.join(output_dir, "experimental_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    json_payload = {
        "dataset_name": "CICIDS2017" if not synthetic else "SYNTHETIC_BENCHMARK",
        "partitions": {
            "train_size": len(idx_train),
            "val_size": len(idx_val),
            "test_size": len(idx_test),
            "rare_classes": rare_classes,
        },
        "baselines": summary_df.to_dict(orient="index"),
        "delta_deas_f1": delta_deas_f1,
        "delta_deas_auc": delta_deas_auc,
        "stratified_recall": stratified_recall,
        "calibration": thresholds,
        "ofat": ofat_results,
    }

    json_path = os.path.join(output_dir, "experimental_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    return json_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de Avaliação Experimental e Baselines (Módulo 5)")
    parser.add_argument("--data_path", type=str, default=None, help="Caminho para arquivo parquet processado")
    parser.add_argument("--synthetic", action="store_true", help="Gera dados sintéticos para benchmarking rápido")
    parser.add_argument("--output_dir", type=str, default="reports", help="Diretório de saída de relatórios")
    parser.add_argument("--random_state", type=int, default=42, help="Semente aleatória determinística")
    parser.add_argument("--run_ofat", action="store_true", help="Executa a análise de sensibilidade OFAT")
    parser.add_argument("--run_lstm", action="store_true", help="Executa o baseline LSTM-AE recorrente")
    args = parser.parse_args()

    run_evaluation_pipeline(
        data_path=args.data_path,
        synthetic=args.synthetic or (args.data_path is None),
        output_dir=args.output_dir,
        random_state=args.random_state,
        run_ofat=args.run_ofat,
        run_lstm=args.run_lstm,
    )


if __name__ == "__main__":
    main()
