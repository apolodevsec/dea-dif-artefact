import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models.autoencoder import FCDAE
from data.dataset import TrafficParquetDataset


def get_best_device(requested_device: str = "auto") -> torch.device:
    if requested_device != "auto":
        if requested_device == "dml":
            try:
                import torch_directml
                return torch_directml.device()
            except ImportError:
                return torch.device("cpu")
        return torch.device(requested_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass

    os_cpus = os.cpu_count() or 4
    torch.set_num_threads(os_cpus)
    return torch.device("cpu")


def extract_latent_space(
    model_path: str,
    parquet_path: str,
    output_parquet_path: str,
    label_column: str = "Label",
    batch_size: int = 2048,
    device_name: str = "auto"
):
    device = get_best_device(device_name)
    print(f"Carregando modelo do checkpoint '{model_path}' no dispositivo: {device}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    input_dim = checkpoint['input_dim']
    bottleneck_dim = checkpoint['bottleneck_dim']
    feature_columns = checkpoint.get('feature_columns', None)

    model = FCDAE(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Modelo carregado com sucesso. Input Dim: {input_dim}, Bottleneck Dim: {bottleneck_dim}")

    print(f"Carregando dados para extração a partir de '{parquet_path}'...")
    dataset = TrafficParquetDataset(
        parquet_path=parquet_path,
        label_column=label_column,
        only_normal=False,  # Carrega todas as amostras (normais e anômalas)
        feature_columns=feature_columns
    )

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    latent_vectors = []
    mse_errors = []
    all_labels = []

    criterion_unreduced = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            x_hat, z_batch = model(x_batch)
            
            # MSE individual por amostra (média sobre as features)
            sample_mse = torch.mean(criterion_unreduced(x_hat, x_batch), dim=1)
            
            latent_vectors.append(z_batch.cpu().numpy())
            mse_errors.append(sample_mse.cpu().numpy())
            
            if isinstance(y_batch, torch.Tensor):
                all_labels.extend(y_batch.numpy())
            elif isinstance(y_batch, list):
                all_labels.extend(y_batch)
            else:
                all_labels.extend(np.array(y_batch))

    latent_matrix = np.vstack(latent_vectors)
    mse_vector = np.concatenate(mse_errors)

    z_col_names = [f"z_{i}" for i in range(bottleneck_dim)]
    latent_df = pd.DataFrame(latent_matrix, columns=z_col_names)
    latent_df["mse"] = mse_vector
    latent_df[label_column] = all_labels

    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    latent_df.to_parquet(output_parquet_path, index=False)
    print(f"Espaço latente e erros MSE extraídos com sucesso!")
    print(f"Arquivo salvo em '{output_parquet_path}'. Dimensão total: {latent_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extração do Espaço Latente (z) e Erro MSE via FC-DAE")
    parser.add_argument("--model_path", type=str, default="models/fc_dae_best.pt", help="Caminho do modelo treinado .pt")
    parser.add_argument("--parquet_path", type=str, default="src/data/X_train_autoencoder.parquet", help="Caminho do arquivo .parquet de entrada")
    parser.add_argument("--output_parquet_path", type=str, default="data/processed/latent_space.parquet", help="Caminho de saída para o espaço latente .parquet")
    parser.add_argument("--label_column", type=str, default="Label", help="Nome da coluna de rótulo")
    parser.add_argument("--batch_size", type=int, default=2048, help="Tamanho do batch de inferência")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo: auto, cpu, cuda, dml")

    args = parser.parse_args()
    extract_latent_space(
        model_path=args.model_path,
        parquet_path=args.parquet_path,
        output_parquet_path=args.output_parquet_path,
        label_column=args.label_column,
        batch_size=args.batch_size,
        device_name=args.device
    )
