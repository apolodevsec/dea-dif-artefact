import argparse
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from models.dif import DeepIsolationForest


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento do Deep Isolation Forest (Módulo 4 - DIF)")
    parser.add_argument("--latent_parquet", type=str, default="data/processed/latent_space.parquet",
                        help="Caminho para o arquivo parquet com o espaço latente extraído")
    parser.add_argument("--model_out", type=str, default="models/dif_ensemble.joblib",
                        help="Caminho para salvar o ensemble serializado")
    parser.add_argument("--metadata_out", type=str, default="models/dif_metadata.json",
                        help="Caminho para salvar metadados e limiares de referência")
    parser.add_argument("--label_column", type=str, default="Label",
                        help="Nome da coluna de rótulo no dataset")
    parser.add_argument("--normal_label", type=str, default="0",
                        help="Valor que identifica o tráfego normal/benigno")
    parser.add_argument("--trees", "--n_estimators", dest="n_estimators", type=int, default=100,
                        help="Número de árvores de isolamento (t)")
    parser.add_argument("--subsample", "--max_samples", dest="max_samples", type=int, default=256,
                        help="Tamanho da subamostra por árvore (psi)")
    parser.add_argument("--representations", "--n_representations", dest="n_representations", type=int, default=10,
                        help="Número de redes de projeção aleatórias independentes (r)")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Número de camadas densas da rede Phi_i (L)")
    parser.add_argument("--projection_dim", type=int, default=None,
                        help="Dimensão de projeção d (padrão m = bottleneck_dim)")
    parser.add_argument("--lambda_deas", type=float, default=0.5,
                        help="Fator de sensibilidade do desvio contínuo DEAS (lambda)")
    parser.add_argument("--val_percentile", type=float, default=95.0,
                        help="Percentil sobre validação normal para calibrar tau_DIF_ref")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fração de tráfego normal reservada para validação")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Semente para reprodutibilidade")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Número de threads/processos para inferência")
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
    print("Iniciando Treinamento do Componente Deep Isolation Forest (Módulo 4)")
    print("=" * 70)

    if not os.path.exists(args.latent_parquet):
        raise FileNotFoundError(f"Arquivo latente não encontrado em: '{args.latent_parquet}'")

    print(f"Lendo dados latentes de '{args.latent_parquet}'...")
    df = pd.read_parquet(args.latent_parquet)
    print(f"Total de registros carregados: {len(df):,}")

    z_cols = get_latent_columns(df)
    m = len(z_cols)
    print(f"Features do espaço latente detectadas (m={m}): {z_cols}")

    # Filtragem estrita do tráfego benigno (sem contaminação de ataques)
    label_col = args.label_column
    if label_col in df.columns:
        labels_str = df[label_col].astype(str).str.strip()
        is_normal = (
            (labels_str == str(args.normal_label)) |
            (labels_str.str.upper() == "BENIGN") |
            (labels_str.str.upper() == "NORMAL")
        )
        df_normal = df[is_normal].copy()
        print(f"Amostras normais filtradas para ajuste: {len(df_normal):,} de {len(df):,}")
    else:
        print(f"Coluna de rótulo '{label_col}' não encontrada. Assumindo todas as amostras como normais.")
        df_normal = df.copy()

    X_all_normal = df_normal[z_cols].values.astype(np.float32)

    # Divisão em Treino e Validação exclusivamente normais
    if args.val_split > 0.0 and len(X_all_normal) > 10:
        X_train, X_val = train_test_split(
            X_all_normal, test_size=args.val_split, random_state=args.random_state
        )
    else:
        X_train, X_val = X_all_normal, X_all_normal

    print(f"Partição de treino normal: {len(X_train):,} amostras")
    print(f"Partição de validação normal: {len(X_val):,} amostras")

    # Instanciação do Estimador DIF
    proj_dim = args.projection_dim or m
    print(f"\nConfiguração dos Hiperparâmetros:")
    print(f"  * Árvores (t): {args.n_estimators}")
    print(f"  * Subamostra (psi): {args.max_samples}")
    print(f"  * Redes de Projeção (r): {args.n_representations}")
    print(f"  * Camadas por rede (L): {args.n_layers}")
    print(f"  * Dimensão de projeção (d): {proj_dim}")
    print(f"  * Fator DEAS (lambda): {args.lambda_deas}")

    clf = DeepIsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        n_representations=args.n_representations,
        n_layers=args.n_layers,
        projection_dim=proj_dim,
        lambda_deas=args.lambda_deas,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    t0 = time.time()
    clf.fit(X_train)
    fit_duration = time.time() - t0
    print(f"\nTreinamento do ensemble concluído em {fit_duration:.2f} segundos.")

    # Calibração do Limiar Estatístico Não-Supervisionado de Referência (tau_DIF_ref)
    print(f"\nAvaliando partição de validação normal para calibrar tau_DIF_ref (percentil {args.val_percentile}%)...")
    val_scores = clf.score_samples(X_val)
    tau_ref_deas = float(np.percentile(val_scores["score_dif_deas"], args.val_percentile))
    tau_ref_standard = float(np.percentile(val_scores["score_dif_standard"], args.val_percentile))

    print(f"  * tau_DIF_ref (DEAS, p{args.val_percentile}): {tau_ref_deas:.6f}")
    print(f"  * tau_DIF_ref (Standard, p{args.val_percentile}): {tau_ref_standard:.6f}")

    # Persistência do Modelo e Metadados
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    joblib.dump(clf, args.model_out)
    print(f"\nModelo DIF serializado com sucesso em: '{args.model_out}'")

    metadata = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latent_parquet": args.latent_parquet,
        "n_samples_train_normal": int(len(X_train)),
        "n_samples_val_normal": int(len(X_val)),
        "z_features": z_cols,
        "m_dim": m,
        "d_projection_dim": proj_dim,
        "n_estimators": args.n_estimators,
        "max_samples": args.max_samples,
        "n_representations": args.n_representations,
        "n_layers": args.n_layers,
        "lambda_deas": args.lambda_deas,
        "val_percentile": args.val_percentile,
        "tau_dif_ref_deas": tau_ref_deas,
        "tau_dif_ref_standard": tau_ref_standard,
        "fit_duration_seconds": fit_duration,
        "random_state": args.random_state,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.metadata_out)), exist_ok=True)
    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadados e limiares gravados em: '{args.metadata_out}'")
    print("=" * 70)


if __name__ == "__main__":
    main()
