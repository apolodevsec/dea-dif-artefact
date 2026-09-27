import argparse
import json
import os
import platform
import time
import joblib
import numpy as np
import pandas as pd
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Inferência e Pontuação com Deep Isolation Forest (Módulo 4)")
    parser.add_argument("--latent_parquet", type=str, default="data/processed/latent_space.parquet",
                        help="Caminho para o arquivo parquet com o espaço latente completo")
    parser.add_argument("--model_path", type=str, default="models/dif_ensemble.joblib",
                        help="Caminho do modelo DIF serializado")
    parser.add_argument("--metadata_path", type=str, default="models/dif_metadata.json",
                        help="Caminho dos metadados de calibração")
    parser.add_argument("--output_parquet", type=str, default="data/processed/dif_scores.parquet",
                        help="Caminho para salvar o parquet com os scores calculados")
    parser.add_argument("--label_column", type=str, default="Label",
                        help="Coluna de rótulo no parquet original")
    parser.add_argument("--normal_label", type=str, default="0",
                        help="Valor que identifica o tráfego normal/benigno")
    parser.add_argument("--batch_size", type=int, default=32768,
                        help="Tamanho do mini-batch para projeções neurais")
    parser.add_argument("--lambda_deas", type=float, default=None,
                        help="Sobrescrever o valor de lambda (padrão: valor do modelo)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Número de jobs para paralelização de árvores")
    return parser.parse_args()


def get_latent_columns(df: pd.DataFrame) -> List[str]:
    z_cols = [col for col in df.columns if col.startswith("z_")]
    if not z_cols:
        exclude = {"Label", "label", "LABEL", "mse", "MSE", "index", "id"}
        z_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    return sorted(z_cols)


def main():
    args = parse_args()
    print("=" * 70)
    print("Iniciando Inferência e Extração de Scores — Módulo 4 (DIF)")
    print("=" * 70)

    if not os.path.exists(args.latent_parquet):
        raise FileNotFoundError(f"Dataset latente não encontrado em: '{args.latent_parquet}'")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Modelo DIF não encontrado em: '{args.model_path}'")

    print(f"Carregando modelo DIF de '{args.model_path}'...")
    clf = joblib.load(args.model_path)
    if args.n_jobs != -1:
        clf.n_jobs = args.n_jobs

    print(f"Lendo dataset de '{args.latent_parquet}'...")
    df = pd.read_parquet(args.latent_parquet)
    n_total = len(df)
    print(f"Total de instâncias a pontuar: {n_total:,}")

    z_cols = get_latent_columns(df)
    print(f"Features latentes utilizadas (m={len(z_cols)}): {z_cols}")
    X = df[z_cols].values.astype(np.float32)

    # Telemetria e Hardware
    print(f"\nAmbiente de Execução:")
    print(f"  * Sistema Operacional: {platform.system()} {platform.release()}")
    print(f"  * Processador: {platform.processor() or 'CPU'}")
    print(f"  * Batch size projeção: {args.batch_size}")
    print(f"  * Workers paralelos: {clf.n_jobs}")

    print("\nExecutando inferência e cálculo de scores de isolamento...")
    t_start = time.time()
    scores = clf.score_samples(X, lambda_deas=args.lambda_deas, batch_size=args.batch_size)
    duration = time.time() - t_start
    throughput = n_total / max(duration, 0.001)

    print(f"Inferência concluída em {duration:.2f} segundos ({throughput:,.1f} amostras/segundo).")

    # Montar DataFrame de resultados preservando rótulos originais
    res_df = pd.DataFrame(index=df.index)
    res_df["score_dif_deas"] = scores["score_dif_deas"]
    res_df["score_dif_standard"] = scores["score_dif_standard"]

    if "mse" in df.columns:
        res_df["mse"] = df["mse"].values

    label_col = args.label_column
    if label_col in df.columns:
        res_df["label_multiclass"] = df[label_col].values
        labels_str = df[label_col].astype(str).str.strip()
        is_normal = (
            (labels_str == str(args.normal_label)) |
            (labels_str.str.upper() == "BENIGN") |
            (labels_str.str.upper() == "NORMAL")
        )
        # 0 = Benigno, 1 = Ataque/Anomalia
        res_df["label_binary"] = np.where(is_normal, 0, 1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_parquet)), exist_ok=True)
    res_df.to_parquet(args.output_parquet, index=True)
    print(f"\nScores e metadados persistidos com sucesso em: '{args.output_parquet}'")
    print(f"Resumo estatístico do Score DEAS:")
    print(f"  * Média: {np.mean(scores['score_dif_deas']):.4f}")
    print(f"  * Desvio padrão: {np.std(scores['score_dif_deas']):.4f}")
    print(f"  * Mínimo: {np.min(scores['score_dif_deas']):.4f}")
    print(f"  * Máximo: {np.max(scores['score_dif_deas']):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
