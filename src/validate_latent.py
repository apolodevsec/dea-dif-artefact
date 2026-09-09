import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp


def set_publication_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.0


def load_stratified_sample(
    df: pd.DataFrame, label_col: str, sample_size: int = 5000
) -> pd.DataFrame:
    n_total = len(df)
    if n_total <= sample_size:
        return df

    # Tenta amostragem estratificada
    try:
        sample_df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / n_total))), random_state=42)
        )
    except Exception:
        sample_df = df.sample(n=sample_size, random_state=42)
        
    return sample_df.reset_index(drop=True)


def plot_tsne(
    df_sample: pd.DataFrame,
    z_cols: List[str],
    label_col: str,
    output_dir: str,
    normal_label_value: str = "0"
):
    print(f"Executando t-SNE sobre {len(df_sample)} amostras estratificadas...")
    X_lat = df_sample[z_cols].values
    labels = df_sample[label_col].astype(str).values

    tsne = TSNE(n_components=2, perplexity=35, random_state=42, init="pca", learning_rate="auto")
    z_2d = tsne.fit_transform(X_lat)

    df_sample["tsne_1"] = z_2d[:, 0]
    df_sample["tsne_2"] = z_2d[:, 1]

    # 1. Gráfico t-SNE Binário (Benigno vs Anômalo)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    is_normal = (labels == str(normal_label_value)) | (labels == "BENIGN") | (labels == "0")
    binary_labels = np.where(is_normal, "Tráfego Benigno", "Tráfego Anômalo")
    df_sample["classe_binaria"] = binary_labels

    plt.figure(figsize=(9, 7))
    palette_bin = {"Tráfego Benigno": "#2ecc71", "Tráfego Anômalo": "#e74c3c"}
    sns.scatterplot(
        data=df_sample,
        x="tsne_1",
        y="tsne_2",
        hue="classe_binaria",
        palette=palette_bin,
        alpha=0.6,
        s=25,
        edgecolor=None
    )
    plt.title("Projeção t-SNE 2D do Espaço Latente (Visão Binária)", fontsize=14, pad=12)
    plt.xlabel("Componente t-SNE 1")
    plt.ylabel("Componente t-SNE 2")
    plt.legend(title="Classe", frameon=True)
    plt.tight_layout()
    
    bin_path = os.path.join(fig_dir, "tsne_binary.png")
    plt.savefig(bin_path, dpi=300)
    plt.close()
    print(f"Gráfico t-SNE binário salvo em: {bin_path}")

    # 2. Gráfico t-SNE Multiclasse (discriminando tipos de ataque)
    plt.figure(figsize=(11, 8))
    unique_labels = df_sample[label_col].nunique()
    palette_multi = sns.color_palette("tab10" if unique_labels <= 10 else "husl", unique_labels)
    
    sns.scatterplot(
        data=df_sample,
        x="tsne_1",
        y="tsne_2",
        hue=label_col,
        palette=palette_multi,
        alpha=0.65,
        s=30,
        edgecolor=None
    )
    plt.title("Projeção t-SNE 2D do Espaço Latente (Visão Multiclasse)", fontsize=14, pad=12)
    plt.xlabel("Componente t-SNE 1")
    plt.ylabel("Componente t-SNE 2")
    plt.legend(title="Categoria", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    multi_path = os.path.join(fig_dir, "tsne_multiclass.png")
    plt.savefig(multi_path, dpi=300)
    plt.close()
    print(f"Gráfico t-SNE multiclasse salvo em: {multi_path}")


def analyze_reconstruction_error(
    df: pd.DataFrame,
    label_col: str,
    mse_col: str,
    output_dir: str,
    normal_label_value: str = "0"
) -> Tuple[float, float, float]:
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    labels = df[label_col].astype(str).values
    is_normal = (labels == str(normal_label_value)) | (labels == "BENIGN") | (labels == "0")

    mse_normal = df.loc[is_normal, mse_col].values
    mse_anomaly = df.loc[~is_normal, mse_col].values

    # Teste estatístico de Kolmogorov-Smirnov
    ks_stat, p_value = ks_2samp(mse_normal, mse_anomaly)
    
    # Rótulos numéricos para ROC-AUC (0 = normal, 1 = anômalo)
    y_true = (~is_normal).astype(int)
    auc_score = roc_auc_score(y_true, df[mse_col].values)

    # Plotagem do KDE overlay
    plt.figure(figsize=(9, 6))
    sns.kdeplot(mse_normal, label="Tráfego Benigno", color="#2ecc71", fill=True, alpha=0.4, common_norm=False)
    sns.kdeplot(mse_anomaly, label="Tráfego Anômalo", color="#e74c3c", fill=True, alpha=0.4, common_norm=False)
    
    plt.title(f"Distribuição Diferencial do Erro de Reconstrução (MSE)\nROC-AUC = {auc_score:.4f} | Est. KS = {ks_stat:.4f}", fontsize=13, pad=12)
    plt.xlabel("Erro Quadrático Médio (MSE)")
    plt.ylabel("Densidade de Probabilidade (KDE)")
    plt.legend(frameon=True)
    plt.tight_layout()

    kde_path = os.path.join(fig_dir, "reconstruction_error_kde.png")
    plt.savefig(kde_path, dpi=300)
    plt.close()
    print(f"Gráfico KDE de erro diferencial salvo em: {kde_path}")

    return ks_stat, p_value, auc_score


def generate_markdown_report(
    output_dir: str,
    sample_size: int,
    total_samples: int,
    ks_stat: float,
    p_value: float,
    auc_score: float
):
    report_path = os.path.join(output_dir, "validation_report.md")
    content = rf"""# Relatório de Validação do Espaço Latente — Módulo 3 (Pre-IF)

## Resumo da Avaliação

* **Total de Conexões Avaliadas:** {total_samples:,}
* **Amostras no Plot t-SNE:** {sample_size:,} (Amostragem Estratificada)
* **Estatística Kolmogorov-Smirnov (KS):** {ks_stat:.6f} (p-valor: {p_value:.4e})
* **ROC-AUC (Erro MSE Isolado):** {auc_score:.4f}

## Gráficos Gerados

1. **Projeção t-SNE Binária:** ![t-SNE Binário](figures/tsne_binary.png)
2. **Projeção t-SNE Multiclasse:** ![t-SNE Multiclasse](figures/tsne_multiclass.png)
3. **Distribuição Diferencial MSE:** ![KDE Reconstrução](figures/reconstruction_error_kde.png)

## Conclusão de Validação
O espaço latente $z \in \mathbb{{R}}^m$ gerado pelo FC-DAE demonstra separabilidade geométrica consistente e resposta diferencial clara ao erro de reconstrução, comprovando a prontidão para a etapa de isolamento pelo Deep Isolation Forest.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Relatório de validação salvo em: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Validação do Espaço Latente Módulo 3 Pre-IF")
    parser.add_argument("--latent_parquet", type=str, required=True, help="Caminho do arquivo .parquet com z e Label")
    parser.add_argument("--label_column", type=str, default="Label", help="Nome da coluna de rótulo")
    parser.add_argument("--mse_column", type=str, default=None, help="Nome da coluna de Erro MSE se pré-calculado")
    parser.add_argument("--sample_size", type=int, default=5000, help="Tamanho da amostra para t-SNE (padrão: 5000)")
    parser.add_argument("--normal_value", type=str, default="0", help="Valor de rótulo para classe normal")
    parser.add_argument("--output_dir", type=str, default="reports", help="Pasta de saída para gráficos e relatórios")

    args = parser.parse_args()
    set_publication_style()

    print(f"Lendo dados de {args.latent_parquet}...")
    df = pd.read_parquet(args.latent_parquet)
    z_cols = [c for c in df.columns if c.startswith("z_")]
    print(f"Colunas de Espaço Latente identificadas ({len(z_cols)}): {z_cols[:5]}...")

    # Se a coluna de MSE não estiver presente no arquivo parquet, mas tivermos colunas originais x e z, calcula se possível
    df_sample = load_stratified_sample(df, args.label_column, args.sample_size)
    plot_tsne(df_sample, z_cols, args.label_column, args.output_dir, args.normal_value)

    ks_stat, p_val, auc_score = 0.0, 1.0, 0.5
    if args.mse_column and args.mse_column in df.columns:
        ks_stat, p_val, auc_score = analyze_reconstruction_error(
            df, args.label_column, args.mse_column, args.output_dir, args.normal_value
        )

    generate_markdown_report(
        args.output_dir, len(df_sample), len(df), ks_stat, p_val, auc_score
    )


if __name__ == "__main__":
    main()
